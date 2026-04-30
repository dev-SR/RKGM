"""
Main pipeline runner.

MVP implementation: sequential Python functions with explicit state object.
LangGraph is NOT required for MVP — it adds human-checkpoint state persistence
and can be added in Tier 3 without changing any pipeline logic.

Usage:
    from pipeline import run_phase_a, run_phase_b

Phase A: paper_id → gaps + candidate papers (fast, no PDFs needed)
Phase B: pdf_paths + phase_a state → generated learning document
"""

from core.models import GapExplanation
import os
import re

from core.models import PipelineState, Paper
from phase_a.gap_detection import detect_gaps
from phase_a.candidates import match_candidates
from utils.apis import fetch_paper, resolve_pdf_url, download_pdf

# NOTE: phase_a.graph and all phase_b modules are imported lazily inside
# the functions that need them.  This prevents import-time crashes when
# heavy deps (ChromaDB, sentence-transformers, marker-pdf) are not yet
# installed or have version conflicts — Phase A can still run cleanly.


def run_phase_a(
    paper_input: str,  # arXiv ID, S2 paper ID, DOI, or local PDF path
    user_gaps: list[str] | None = None,
    progress_callback=None,  # callable(str) for UI status
    pdf_path: str | None = None,  # explicit local PDF for the base article
    reference_depth: int = 1,  # 0=no graph, 1=direct refs, 2=refs-of-refs, 3=full
) -> PipelineState:
    """
    Phase A: fetch base article, optionally build reference graph,
    detect knowledge gaps, and match candidate papers.

    Args:
        paper_input:     arXiv ID (2405.20139), DOI (10.1109/...), or S2 ID
        user_gaps:       extra concepts the user always wants explained
        pdf_path:        local PDF of the base article — richer gap detection
        reference_depth: 0 = gap detection only, no reference graph or candidates
                         1 = direct references only (fast, recommended default)
                         2 = references of references
                         3 = full 3-level graph (slow on first run, cached after)
    """
    # Lazy import — prevents import-time crash if phase_b/graph deps not installed
    from phase_a.graph import build_reference_graph, graph_summary

    state = PipelineState()
    cb = progress_callback or (lambda msg: print(f"[pipeline] {msg}"))

    # ── Normalise paper ID ─────────────────────────────────────────────────
    paper_id = _normalise_paper_id(paper_input)
    cb(f"Looking up paper: {paper_id}…")

    # ── Fetch base article metadata ────────────────────────────────────────
    raw = fetch_paper(paper_id)
    if not raw:
        state.errors.append(
            f"Could not fetch paper '{paper_id}' from Semantic Scholar. "
            f"Check the ID format and your internet connection."
        )
        return state

    ext = raw.get("externalIds") or {}
    oap = raw.get("openAccessPdf") or {}
    ba = Paper(
        paper_id=raw["paperId"],
        title=raw.get("title") or paper_id,
        abstract=raw.get("abstract") or "",
        year=raw.get("year") or 0,
        citation_count=raw.get("citationCount") or 0,
        arxiv_id=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        pdf_url=oap.get("url"),
        level=0,
    )
    state.ba_paper = ba

    # ── Extract BA full text ───────────────────────────────────────────────
    ba_text = _get_ba_text(ba, cb, explicit_pdf_path=pdf_path)
    state.ba_text = ba_text

    if not ba_text.strip():
        cb("Warning: no text extracted — falling back to abstract")
        state.ba_text = ba.abstract
        ba_text = ba.abstract

    # ── Reference graph (skipped at depth=0) ──────────────────────────────
    papers: dict = {}
    G = None

    if reference_depth == 0:
        cb("Depth 0: skipping reference graph — gap detection from BA text only")
        # Add the BA itself as the only paper so candidate matching is a no-op
        ba.level = 0
        papers[ba.paper_id] = ba
        from phase_a.graph import _assign_layers

        _assign_layers(papers)
        import networkx as nx

        G = nx.DiGraph()
        G.add_node(ba.paper_id)
    else:
        depth_label = {
            1: "Level 1 only (direct references — fast)",
            2: "Levels 1–2 (refs of refs — medium)",
            3: "Levels 1–3 (full graph — slow first run, cached after)",
        }
        cb(
            f"Building reference graph — {depth_label.get(reference_depth, str(reference_depth))}…"
        )
        try:
            G, papers = build_reference_graph(
                ba.paper_id,
                progress_callback=cb,
                max_depth=reference_depth,
            )
            summary = graph_summary(G, papers)
            cb(
                f"Graph ready: {summary['total_papers']} papers  "
                f"({summary['foundation']} Foundation / "
                f"{summary['development']} Development / "
                f"{summary['frontier']} Frontier)"
            )
        except Exception as e:
            state.errors.append(f"Graph construction failed: {e}")
            return state

    state.reference_graph = G
    state.all_papers = papers

    # ── Gap detection ──────────────────────────────────────────────────────
    cb("Running gap detection with self-consistency (3 runs)…")
    try:
        gaps = detect_gaps(
            paper_text=ba_text,
            abstract=ba.abstract,
            user_gaps=user_gaps,
            progress_callback=cb,
        )
    except Exception as e:
        state.errors.append(f"Gap detection failed: {e}")
        return state

    state.gaps = gaps
    cb(f"Detected {len(gaps)} knowledge gaps")

    # ── Candidate matching (skipped at depth=0) ────────────────────────────
    if reference_depth == 0:
        cb("Depth 0: no reference graph — skipping candidate matching")
        state.candidates = []
        cb(f"Phase A complete (depth=0) — {len(gaps)} gaps, no candidates")
        return state

    cb("Matching candidate papers for each gap…")
    try:
        candidates = match_candidates(
            gaps=gaps,
            papers=papers,
            progress_callback=cb,
        )
    except Exception as e:
        state.errors.append(f"Candidate matching failed: {e}")
        return state

    state.candidates = candidates
    cb(f"Phase A complete — {len(gaps)} gaps, {len(candidates)} candidate links")
    return state


def run_phase_b(
    state: PipelineState,
    pdf_paths: dict[str, str] | None = None,  # paper_id -> local pdf path
    auto_fetch: bool = True,  # try Unpaywall/arXiv automatically
    progress_callback=None,
) -> PipelineState:
    """
    Phase B: ingest PDFs, retrieve, generate, assemble document.
    Expects a PipelineState returned by run_phase_a.

    pdf_paths: manually supplied PDFs (from UI upload)
    auto_fetch: also attempt automatic PDF fetch via Unpaywall/arXiv
    """
    # Lazy imports — phase_b deps (chromadb, marker-pdf, cross-encoder)
    # only loaded when Phase B actually runs
    from utils.ingest import ingest_pdfs
    from phase_b.retrieval import hybrid_retrieve
    from phase_b.ordering import order_gaps, build_dependency_map
    from phase_b.generation import (
        generate_explanation,
        detect_subgaps,
        assemble_document,
    )

    cb = progress_callback or (lambda msg: print(f"[pipeline] {msg}"))

    if not state.ba_paper or not state.gaps:
        state.errors.append("Phase A must complete successfully before Phase B")
        return state

    # ── Collect PDF paths (auto + manual) ─────────────────────────────────
    all_pdf_paths = dict(pdf_paths or {})

    if auto_fetch:
        cb("Auto-fetching open-access PDFs…")
        download_dir = "/tmp/kgms_pdfs"
        os.makedirs(download_dir, exist_ok=True)
        _auto_fetch_pdfs(state, all_pdf_paths, download_dir, cb)

    if not all_pdf_paths:
        cb("No PDFs available — generating abstract-only explanations")

    # ── Ingest PDFs ────────────────────────────────────────────────────────
    cb("Ingesting and indexing PDFs…")
    try:
        chunks, collection = ingest_pdfs(
            pdf_paths=all_pdf_paths,
            papers=state.all_papers,
            progress_callback=cb,
        )
    except Exception as e:
        state.errors.append(f"PDF ingestion failed: {e}")
        chunks, collection = [], None

    state.chunks = chunks
    state.chroma_collection = collection
    cb(f"Indexed {len(chunks)} chunks from {len(all_pdf_paths)} PDFs")

    # ── Determine ordering ─────────────────────────────────────────────────
    cb("Determining chronological learning order…")
    ordered_ids, dependencies = order_gaps(
        gaps=state.gaps,
        candidates=state.candidates,
        progress_callback=cb,
    )
    state.ordered_gap_ids = ordered_ids
    state.dependencies = dependencies

    dep_map = build_dependency_map(dependencies)
    gap_map = {g.gap_id: g for g in state.gaps}

    # ── Generate explanations (with multi-hop) ─────────────────────────────
    known_concepts: set[str] = set()
    explanations: list = []
    gap_queue: list = [gap_map[gid] for gid in ordered_ids if gid in gap_map]
    i = 0

    while i < len(gap_queue):
        gap = gap_queue[i]
        i += 1

        cb(f"Generating explanation {i}/{len(gap_queue)}: {gap.concept}…")

        # Retrieve chunks for this gap
        if collection and chunks:
            retrieved_chunks, is_abstract_only = hybrid_retrieve(
                gap=gap,
                collection=collection,
                all_chunks=chunks,
            )
        else:
            retrieved_chunks, is_abstract_only = [], True

        # Generate explanation
        try:
            exp = generate_explanation(
                gap=gap,
                chunks=retrieved_chunks,
                papers=state.all_papers,
                graph=state.reference_graph,
                ba_title=state.ba_paper.title,
                known_concepts=known_concepts,
                is_abstract_only=is_abstract_only,
                progress_callback=cb,
            )

            # RAGAS gate: if faithfulness low, expand top-k and retry once
            if not is_abstract_only and exp.confidence < 0.55 and collection:
                cb(
                    f"Low confidence ({exp.confidence:.2f}), expanding retrieval for '{gap.concept}'…"
                )
                from core.config import RERANK_TOP_N

                expanded_chunks, _ = hybrid_retrieve(
                    gap=gap,
                    collection=collection,
                    all_chunks=chunks,
                    top_k_override=15,  # expand from 5 to 15
                )
                if len(expanded_chunks) > len(retrieved_chunks):
                    exp = generate_explanation(
                        gap=gap,
                        chunks=expanded_chunks,
                        papers=state.all_papers,
                        graph=state.reference_graph,
                        ba_title=state.ba_paper.title,
                        known_concepts=known_concepts,
                        is_abstract_only=False,
                        progress_callback=cb,
                    )

        except Exception as e:
            cb(f"Warning: explanation generation failed for '{gap.concept}': {e}")
            exp = _fallback_explanation(gap)

        exp.order_position = len(explanations)
        exp.dependency_note = dep_map.get(gap.gap_id, "")
        explanations.append(exp)
        known_concepts.add(gap.concept.lower())

        # Multi-hop: detect sub-gaps introduced by this explanation
        depth = _get_hop_depth(gap.gap_id)
        if depth < 2:
            subgaps = detect_subgaps(
                explanation_text=exp.explanation_text,
                parent_gap=gap,
                known_concepts=known_concepts,
                depth=depth,
                parent_had_pdf=not exp.is_abstract_only,
            )
            # Insert sub-gaps immediately after current position
            for j, sg in enumerate(subgaps):
                if sg.concept.lower() not in known_concepts:
                    gap_queue.insert(i + j, sg)
                    cb(
                        f"Sub-gap detected: {sg.concept} (will explain before continuing)"
                    )

    state.explanations = explanations

    # ── Coverage verification ──────────────────────────────────────────────
    try:
        from eval.coverage import verify_coverage, coverage_badge

        coverage = verify_coverage(state.gaps, explanations)
        badge, label = coverage_badge(coverage.coverage_score)
        cb(
            f"Coverage check: {badge} {label} "
            f"({coverage.covered} covered, {coverage.abstract_only} abstract-only, "
            f"{coverage.missing} missing)"
        )
        if coverage.recommendations:
            for rec in coverage.recommendations:
                cb(f"  ↳ {rec}")
        state.errors.extend(
            [
                f"[coverage] {r}"
                for r in coverage.recommendations
                if "skipped" in r or "ungrounded" in r
            ]
        )
        state._coverage_report = coverage
    except Exception as e:
        cb(f"Coverage check failed (non-critical): {e}")

    # ── Generate document preamble ─────────────────────────────────────────
    cb("Generating document preamble…")
    try:
        from phase_b.generation import generate_document_preamble

        preamble = generate_document_preamble(
            ba_title=state.ba_paper.title,
            ba_abstract=state.ba_paper.abstract,
            gaps=state.gaps,
        )
    except Exception as e:
        cb(f"Preamble generation failed (non-critical): {e}")
        preamble = None

    # ── Assemble document ──────────────────────────────────────────────────
    cb("Assembling learning document…")
    state.final_document = assemble_document(
        ordered_explanations=explanations,
        ba_title=state.ba_paper.title,
        ba_abstract=state.ba_paper.abstract,
        dependency_map=dep_map,
        gaps=state.gaps,
        preamble_text=preamble,
    )

    cb("Phase B complete ✓")
    return state


# ── Helpers ────────────────────────────────────────────────────────────────


def _normalise_paper_id(input_str: str) -> str:
    """
    Accept arXiv IDs (2404.16130 or arXiv:2404.16130),
    DOIs (10.1109/...), or raw S2 paper IDs.
    Semantic Scholar API accepts all of these directly.
    """
    s = input_str.strip()
    if s.lower().startswith("arxiv:"):
        return s  # S2 accepts "arXiv:XXXX.YYYYY"
    if "/" in s and s.startswith("10."):
        return f"DOI:{s}"
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s):
        return f"arXiv:{s}"
    return s


def _get_ba_text(ba: Paper, cb, explicit_pdf_path: str | None = None) -> str:
    """
    Extract full text from the base article.

    Priority:
      1. explicit_pdf_path — user supplied a local file directly
      2. ba.pdf_url        — download from open-access URL
      3. resolve via Unpaywall / arXiv
      4. abstract          — fallback (gap detection still works, just less precise)
    """
    from utils.ingest import parse_pdf_to_markdown

    # 1. Explicit local PDF (most reliable — user chose this file)
    if explicit_pdf_path and os.path.exists(explicit_pdf_path):
        cb(f"Reading base article PDF: {os.path.basename(explicit_pdf_path)}")
        md = parse_pdf_to_markdown(explicit_pdf_path)
        if md:
            word_count = len(md.split())
            cb(f"PDF parsed successfully — {word_count:,} words extracted")
            return md
        else:
            cb("Warning: PDF parsing returned no text, falling back to abstract")

    # 2. Download from known URL
    if ba.pdf_url:
        cached_path = f"/tmp/kgms_pdfs/ba_{ba.paper_id.replace('/', '_')}.pdf"
        os.makedirs("/tmp/kgms_pdfs", exist_ok=True)
        if not os.path.exists(cached_path):
            cb(f"Downloading base article PDF from open-access URL…")
            download_pdf(ba.pdf_url, cached_path)
        if os.path.exists(cached_path):
            md = parse_pdf_to_markdown(cached_path)
            if md:
                cb(f"PDF downloaded and parsed — {len(md.split()):,} words")
                return md

    # 3. Resolve via Unpaywall / arXiv
    raw_data = {
        "paperId": ba.paper_id,
        "externalIds": {"ArXiv": ba.arxiv_id, "DOI": ba.doi},
        "openAccessPdf": {"url": ba.pdf_url} if ba.pdf_url else None,
    }
    url = resolve_pdf_url(raw_data)
    if url:
        cached_path = f"/tmp/kgms_pdfs/ba_{ba.paper_id.replace('/', '_')}.pdf"
        os.makedirs("/tmp/kgms_pdfs", exist_ok=True)
        if download_pdf(url, cached_path):
            md = parse_pdf_to_markdown(cached_path)
            if md:
                cb(f"PDF resolved and parsed — {len(md.split()):,} words")
                return md

    # 4. Fallback to abstract
    cb("No PDF available — using abstract only (gap detection will be less precise)")
    return ba.abstract


def _auto_fetch_pdfs(
    state: PipelineState,
    all_pdf_paths: dict,
    download_dir: str,
    cb,
):
    """Try to fetch PDFs for all candidate papers via Unpaywall/arXiv."""
    seen: set[str] = set()
    for c in state.candidates:
        pid = c.paper.paper_id
        if pid in seen or pid in all_pdf_paths:
            continue
        seen.add(pid)

        url = c.paper.pdf_url
        if not url:
            raw_data = {
                "paperId": pid,
                "externalIds": {
                    "ArXiv": c.paper.arxiv_id,
                    "DOI": c.paper.doi,
                },
                "openAccessPdf": None,
            }
            url = resolve_pdf_url(raw_data)
            c.paper.pdf_url = url
            c.pdf_available = bool(url)

        if url:
            safe_pid = pid.replace("/", "_")
            dest = os.path.join(download_dir, f"{safe_pid}.pdf")
            if os.path.exists(dest) or download_pdf(url, dest):
                all_pdf_paths[pid] = dest


def _get_hop_depth(gap_id: str) -> int:
    """Count the number of '_sub' suffixes to determine multi-hop depth."""
    return gap_id.count("_sub")


def _fallback_explanation(gap) -> GapExplanation:
    return GapExplanation(
        gap_id=gap.gap_id,
        concept=gap.concept,
        explanation_text=f"*Explanation unavailable for '{gap.concept}'.*\n\n{gap.why_needed}",
        source_citations=[],
        confidence=0.0,
        is_abstract_only=True,
    )
