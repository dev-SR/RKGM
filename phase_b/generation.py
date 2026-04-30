"""
Generation pipeline.

For each gap (in ordered sequence):
  1. Writing Agent: generate explanation with inline [cite: paper_id::chunk_id] markers
  2. Evaluation Agent: check for sub-gaps, unsupported citations, hallucinations
  3. Iterate up to MAX_EVAL_LOOPS
  4. Multi-hop: detect sub-gaps introduced by the explanation
  5. Assemble final Markdown document

Model routing:
  Writing Agent → LLM_HEAVY (70B): multi-paragraph coherent generation
  Evaluation Agent → LLM_LIGHT (8B): structured JSON checking task
  Sub-gap detection → LLM_LIGHT (8B): short-list extraction
"""

import json
import re

from core.models import (
    KnowledgeGap,
    GapType,
    Difficulty,
    Layer,
    Chunk,
    GapExplanation,
    Paper,
)
from core.config import MAX_EVAL_LOOPS, MAX_MULTIHOP_DEPTH
from core.prompts import WRITING_SYSTEM, WRITING_USER, EVAL_PROMPT, SUBGAP_PROMPT
from utils.llm import llm_heavy, llm_light, parse_json_response
from phase_b.retrieval import verbalize_path, format_passages


# ── Single gap explanation ─────────────────────────────────────────────────


def generate_explanation(
    gap: KnowledgeGap,
    chunks: list[Chunk],
    papers: dict[str, Paper],
    graph,
    ba_title: str,
    known_concepts: set[str],
    is_abstract_only: bool = False,
    progress_callback=None,
) -> GapExplanation:
    """
    Generate and iteratively refine an explanation for a single gap.
    Returns a GapExplanation with full text and citation markers.
    """
    if is_abstract_only or not chunks:
        return _abstract_only_explanation(gap, papers=papers, ba_title=ba_title)

    path = verbalize_path(gap, chunks, papers)
    passages = format_passages(chunks, papers)

    user_prompt = WRITING_USER.format(
        ba_title=ba_title,
        concept=gap.concept,
        gap_type=gap.gap_type.value,
        difficulty=gap.difficulty.value,
        why_needed=gap.why_needed,
        path_verbalization=path,
        passages=passages,
    )

    explanation_text = ""
    current_chunks = list(chunks)  # may be expanded by RAGAS gate

    for loop in range(MAX_EVAL_LOOPS):
        if progress_callback:
            progress_callback(
                f"Writing explanation for '{gap.concept}' "
                f"(iteration {loop + 1}/{MAX_EVAL_LOOPS})…"
            )

        # Writing Agent (heavy model)
        explanation_text = llm_heavy(
            WRITING_SYSTEM, user_prompt, temperature=0.3, max_tokens=1500
        )

        # Evaluation Agent (light model)
        eval_issues = _evaluate_explanation(
            explanation_text, passages, gap.concept, known_concepts
        )

        if eval_issues.get("approved", True):
            # RAGAS faithfulness gate — check quality before accepting
            scores = _ragas_gate_check(gap.concept, explanation_text, current_chunks)
            if not scores["passes"]:
                if progress_callback:
                    progress_callback(
                        f"Faithfulness gate failed "
                        f"(faithfulness={scores['faithfulness']:.2f}). "
                        f"Expanding retrieval…"
                    )
                # Expand retrieval and rebuild passages
                current_chunks = _expand_retrieval(gap, current_chunks, papers)
                if current_chunks:
                    passages = format_passages(current_chunks, papers)
                    user_prompt = WRITING_USER.format(
                        ba_title=ba_title,
                        concept=gap.concept,
                        gap_type=gap.gap_type.value,
                        difficulty=gap.difficulty.value,
                        why_needed=gap.why_needed,
                        path_verbalization=path,
                        passages=passages,
                    )
                    continue  # re-generate with expanded context
            break  # passed gate or no improvement possible

        # Build correction instruction for next iteration
        corrections = []
        hallucinations = eval_issues.get("hallucinations", [])
        unsupported = eval_issues.get("unsupported_citations", [])

        if hallucinations:
            bad = "; ".join(hallucinations[:2])
            corrections.append(
                f"Do NOT include this claim (not found in source passages): {bad}"
            )
        if unsupported:
            corrections.append(
                f"These claims need proper [cite:] markers from the passages: "
                f"{'; '.join(unsupported[:2])}"
            )

        if corrections:
            user_prompt += "\n\nCORRECTIONS REQUIRED:\n" + "\n".join(corrections)

    # Extract citation IDs from explanation
    citations = list(set(re.findall(r"\[cite:\s*([^\]]+)\]", explanation_text)))

    # Confidence from RAGAS gate score + base gap confidence
    gate_scores = _ragas_gate_check(gap.concept, explanation_text, current_chunks)
    confidence = min(
        0.96, gap.confidence * 0.4 + gate_scores.get("faithfulness", 0.5) * 0.6
    )

    return GapExplanation(
        gap_id=gap.gap_id,
        concept=gap.concept,
        explanation_text=explanation_text,
        source_citations=citations,
        confidence=round(confidence, 3),
        is_abstract_only=False,
    )


def _ragas_gate_check(concept: str, explanation_text: str, chunks: list) -> dict:
    """
    Run the RAGAS faithfulness gate. Imported lazily to avoid circular imports.
    Returns {"passes": bool, "faithfulness": float, "context_recall": float}
    """
    try:
        from eval.evaluate import evaluate_explanation, passes_faithfulness_gate

        scores = evaluate_explanation(concept, explanation_text, chunks)
        return {
            "passes": passes_faithfulness_gate(scores),
            "faithfulness": scores.get("faithfulness", 0.5),
            "context_recall": scores.get("context_recall", 0.5),
            "answer_relevancy": scores.get("answer_relevancy", 0.5),
        }
    except Exception:
        return {"passes": True, "faithfulness": 0.5, "context_recall": 0.5}


def _expand_retrieval(gap, current_chunks: list, papers: dict) -> list:
    """Placeholder — real top-k expansion is handled in pipeline.py."""
    return current_chunks


def _evaluate_explanation(
    explanation: str,
    passages: str,
    concept: str,
    known_concepts: set[str],
) -> dict:
    """
    Evaluation Agent: structured check of explanation quality.
    Light model (8B) is sufficient for this JSON extraction task.
    """
    prompt = EVAL_PROMPT.format(
        concept=concept,
        explanation=explanation[:2000],
        passages=passages[:3000],
        explained_concepts=json.dumps(list(known_concepts)),
    )
    try:
        raw = llm_light(prompt, max_tokens=600)
        return parse_json_response(raw)
    except Exception:
        return {"approved": True}  # fail open — do not block generation


def _abstract_only_explanation(
    gap: KnowledgeGap,
    papers: dict | None = None,  # all_papers from state, for abstract lookup
    ba_title: str = "",
) -> GapExplanation:
    """
    Generate an actual explanation using only abstract-level information.

    Rather than just repeating why_needed (which is just a template string),
    call the LLM with whatever we know: the gap concept, the gap type,
    why the paper needs it, and the abstract of the best matching candidate.
    This always produces real content, clearly labelled as abstract-based.
    """
    # Find the best abstract to use — search all_papers for the closest match
    best_abstract = ""
    if papers:
        from utils.embedder import embed_single
        import numpy as np

        q_emb = embed_single(gap.retrieval_query)
        best_sim = -1.0
        for p in papers.values():
            if not p.abstract or p.paper_id == gap.gap_id:
                continue
            p_emb = embed_single(p.abstract[:300])
            sim = float(np.dot(q_emb, p_emb))
            if sim > best_sim:
                best_sim = sim
                best_abstract = f'"{p.title}" ({p.year}): {p.abstract[:500]}'

    context_block = (
        f"\nBest matching abstract:\n{best_abstract}" if best_abstract else ""
    )

    prompt = f"""You are explaining a prerequisite concept for a reader about to read a research paper.

Target paper: "{ba_title}"
Concept to explain: {gap.concept}
Type: {gap.gap_type.value}
Why the reader needs this: {gap.why_needed}
{context_block}

Write 2-3 clear paragraphs explaining what "{gap.concept}" is, why it matters for 
the target paper, and what the reader should understand before proceeding.

Important:
- Write actual explanatory content, not a summary of who wrote what
- If this is a historical citation, explain what that work CONTRIBUTED, not who wrote it
- Use plain language, be concrete
- Do NOT use [cite:] markers (no PDFs available)
- End with one sentence connecting this to the target paper"""

    try:
        text = llm_light(prompt, max_tokens=500)
        conf = 0.40
    except Exception:
        text = f"{gap.why_needed}"
        conf = 0.1

    note = (
        "*📄 Abstract-level explanation — no PDF available for direct sources. "
        "Core concepts described from available metadata.*\n\n"
    )

    return GapExplanation(
        gap_id=gap.gap_id,
        concept=gap.concept,
        explanation_text=note + text,
        source_citations=[],
        confidence=conf,
        is_abstract_only=True,
    )


# ── Multi-hop sub-gap detection ────────────────────────────────────────────


def detect_subgaps(
    explanation_text: str,
    parent_gap: KnowledgeGap,
    known_concepts: set[str],
    depth: int = 0,
    parent_had_pdf: bool = True,
) -> list[KnowledgeGap]:
    """
    After generating an explanation, check whether it introduces new unexplained terms.
    Returns sub-gap KnowledgeGap objects to retrieve and explain before moving on.

    Guards against cascade:
    - Never spawn sub-gaps from a parent that had no PDF (they'll all be abstract-only)
    - Never spawn at depth >= MAX_MULTIHOP_DEPTH
    - Filter author citations, generic terms, and anything already known
    """
    if depth >= MAX_MULTIHOP_DEPTH:
        return []

    # Never cascade from abstract-only explanations — no PDF means sub-gaps
    # will also have no PDF, producing useless 0% confidence noise
    if not parent_had_pdf:
        return []

    prompt = SUBGAP_PROMPT.format(
        concept=parent_gap.concept,
        explanation=explanation_text[:1500],
        known_concepts=json.dumps(sorted(known_concepts)),
    )
    try:
        raw = llm_light(prompt, max_tokens=150)
        subterms = parse_json_response(raw)
        if not isinstance(subterms, list):
            return []
    except Exception:
        return []

    # Post-filter: catch what the LLM misses
    _GENERIC = {
        "parameter",
        "parameters",
        "model",
        "models",
        "method",
        "methods",
        "approach",
        "approaches",
        "system",
        "systems",
        "task",
        "tasks",
        "data",
        "dataset",
        "training",
        "learning",
        "network",
        "networks",
        "function",
        "functions",
        "layer",
        "layers",
        "feature",
        "features",
        "input",
        "output",
        "result",
        "results",
        "performance",
        "baseline",
    }
    _AUTHOR_PAT = re.compile(r"\bet\s+al\.?$", re.IGNORECASE)

    subgaps = []
    for i, term in enumerate(subterms[:2]):  # hard cap at 2 sub-gaps
        term = str(term).strip()
        if not term:
            continue
        tl = term.lower()
        # Drop if already known
        if tl in known_concepts:
            continue
        # Drop author citations ("Brown et al.", "Kipf and Welling")
        if _AUTHOR_PAT.search(term) or " et al" in tl:
            continue
        # Drop generic single-word CS terms
        if tl in _GENERIC or (len(tl.split()) == 1 and len(tl) < 8):
            continue

        subgaps.append(
            KnowledgeGap(
                gap_id=f"{parent_gap.gap_id}_sub{depth}_{i}",
                concept=term,
                gap_type=GapType.METHODOLOGY,
                difficulty=Difficulty.INTERMEDIATE,
                domain=parent_gap.domain,
                why_needed=(
                    f"Introduced in the explanation of '{parent_gap.concept}' "
                    f"without being defined."
                ),
                layer_hint=Layer.FOUNDATION,
                retrieval_query=f"{term} definition explanation {parent_gap.domain}",
                source_passage=f"Sub-concept of {parent_gap.concept}",
                confidence=0.70,
            )
        )

    return subgaps


# ── Document assembly ──────────────────────────────────────────────────────


def generate_document_preamble(
    ba_title: str,
    ba_abstract: str,
    gaps: list,
) -> str:
    """
    Generate an LLM-written introduction before individual concept sections.
    Called once before Phase B generation loop — cheap single 8B call.
    """
    from core.prompts import PREAMBLE_PROMPT

    gap_list = "\n".join(
        f"- {g.concept} ({g.gap_type.value}): {g.why_needed}"
        for g in gaps[:15]  # cap to avoid token overflow
    )
    prompt = PREAMBLE_PROMPT.format(
        ba_title=ba_title,
        ba_abstract=ba_abstract[:600],
        gap_list=gap_list,
    )
    try:
        return llm_light(prompt, max_tokens=300)
    except Exception:
        return (
            f"This document prepares you to read *{ba_title}* by explaining "
            f"{len(gaps)} prerequisite concepts identified in the paper. "
            f"Work through the sections in order — each concept builds on those before it."
        )


def _get_section_title(exp: GapExplanation, gap: "KnowledgeGap | None") -> str:
    """
    For historical gaps ("Brown et al."), generate a descriptive title
    based on the work's contribution rather than the author citation.
    Cached so we don't re-call for re-runs.
    """
    if gap is None or gap.gap_type.value != "historical":
        return exp.concept

    # Check if concept looks like an author citation
    import re as _re

    is_citation = (
        bool(_re.search(r"\bet\s+al\.?", exp.concept, _re.IGNORECASE))
        or bool(_re.search(r"^[A-Z][a-z]+ (and|&) [A-Z][a-z]+$", exp.concept))
        or bool(_re.search(r"^[A-Z][a-z]+\s+et\s+al", exp.concept))
    )

    if not is_citation:
        return exp.concept

    from core.prompts import HISTORICAL_TITLE_PROMPT
    from utils.cache import get_cached, set_cached

    cache_key = f"hist_title:{exp.gap_id}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    prompt = HISTORICAL_TITLE_PROMPT.format(
        concept=exp.concept,
        why_needed=gap.why_needed if gap else "",
        source_passage=gap.source_passage[:150] if gap else "",
    )
    try:
        title = llm_light(prompt, max_tokens=30).strip().strip('"').strip("'")
        # Sanity check — if LLM returned something weird, fall back
        if len(title) < 5 or len(title.split()) > 8:
            title = exp.concept
        set_cached(cache_key, title)
        return title
    except Exception:
        return exp.concept


def assemble_document(
    ordered_explanations: list[GapExplanation],
    ba_title: str,
    ba_abstract: str,
    dependency_map: dict[str, str],
    gaps: list | None = None,  # full gap list for preamble
    preamble_text: str | None = None,  # pre-generated preamble
) -> str:
    """
    Assemble the final learning document in Markdown.

    Improvements over v1:
    - LLM-generated preamble explaining what the document covers
    - "What You Will Learn" table of contents with gap types
    - Historical gaps titled by contribution, not author name
    - Abstract-only sections at 0% confidence silently dropped
    - Gap type badge on every section heading
    - Dependency notes rendered as blockquotes
    - Cleaner confidence display
    """
    gap_map = {g.gap_id: g for g in (gaps or [])}

    # Drop entries that have no real content at all:
    # - abstract-only with confidence <= 0.1 (LLM call failed entirely)
    # - explanation is literally just a fallback template with no LLM text
    def _keep(exp: GapExplanation) -> bool:
        text = exp.explanation_text.strip()
        # Generated content failed entirely
        if exp.confidence <= 0.05:
            return False
        # Just the template why_needed line with nothing added
        if len(text.split()) < 20:
            return False
        return True

    filtered = [e for e in ordered_explanations if _keep(e)]

    # ── Header ────────────────────────────────────────────────────────────
    lines = [
        f"# Learning Roadmap for: {ba_title}\n",
    ]

    # ── Preamble ──────────────────────────────────────────────────────────
    intro = preamble_text or (
        f"This document prepares you to read *{ba_title}* by explaining "
        f"{len(filtered)} prerequisite concepts. "
        f"Work through the sections in order — each one builds on the previous."
    )
    lines.append(f"{intro}\n")

    # ── Abstract ──────────────────────────────────────────────────────────
    lines.append(f"> **Paper abstract:** {ba_abstract[:350]}…\n")

    # ── Table of Contents ─────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## What You Will Learn\n")

    type_icons = {
        "terminology": "📖",
        "methodology": "⚙️",
        "benchmark": "📊",
        "historical": "📜",
        "mathematical": "🔢",
    }

    for i, exp in enumerate(filtered, 1):
        gap = gap_map.get(exp.gap_id)
        icon = type_icons.get(gap.gap_type.value if gap else "", "•")
        title = _get_section_title(exp, gap)
        diff = f"*{gap.difficulty.value}*" if gap else ""
        conf = (
            f"{int(exp.confidence * 100)}%" if not exp.is_abstract_only else "abstract"
        )
        status = "🟡" if exp.is_abstract_only else "✅"
        lines.append(
            f"{i}. {status} {icon} **[{title}](#{title.lower().replace(' ', '-')})** "
            f"— {diff}, {conf}"
        )
        if gap:
            lines.append(f"   *{gap.why_needed}*")

    lines.append("\n---\n")

    # ── Sections ──────────────────────────────────────────────────────────
    for i, exp in enumerate(filtered, 1):
        gap = gap_map.get(exp.gap_id)
        title = _get_section_title(exp, gap)

        # Section badge line
        type_val = gap.gap_type.value if gap else "concept"
        icon = type_icons.get(type_val, "•")
        layer_val = gap.layer_hint.value if gap else ""
        layer_badge = {
            "foundation": "🏛 Foundation",
            "development": "📈 Development",
            "frontier": "🚀 Frontier",
        }.get(layer_val, "")
        diff_badge = {"beginner": "🌱", "intermediate": "🌿", "advanced": "🌳"}.get(
            gap.difficulty.value if gap else "", ""
        )

        if exp.is_abstract_only:
            conf_str = "🟡 abstract only"
        else:
            conf_str = f"✅ {int(exp.confidence * 100)}% confidence"

        lines.append(f"## {i}. {title}")
        lines.append(
            f"*{icon} {type_val.capitalize()} · {diff_badge} "
            f"{gap.difficulty.value if gap else ''} · "
            f"{layer_badge} · {conf_str}*\n"
        )

        # Why needed callout
        if gap:
            lines.append(f"> **Why you need this:** {gap.why_needed}\n")

        # Main explanation
        lines.append(exp.explanation_text)
        lines.append("")

        # Dependency note
        dep_note = dependency_map.get(exp.gap_id, "")
        if dep_note:
            lines.append(f"> 📚 **Builds on:** {dep_note}\n")

        # Source papers (show titles, not chunk IDs)
        if exp.source_citations and not exp.is_abstract_only:
            cited_papers = set()
            for cid in exp.source_citations[:6]:
                pid = cid.split("::")[0]
                cited_papers.add(pid)
            lines.append(
                f"*Sourced from {len(cited_papers)} reference paper(s) · "
                f"{len(exp.source_citations)} passages cited*"
            )

        lines.append("\n---\n")

    # ── Closing ───────────────────────────────────────────────────────────
    full_count = sum(1 for e in filtered if not e.is_abstract_only)
    abstract_count = sum(1 for e in filtered if e.is_abstract_only)

    lines.append(f"## ✅ You Are Ready to Read: *{ba_title}*\n")
    lines.append(
        f"You have covered {len(filtered)} prerequisite concepts: "
        f"{full_count} with full-text explanations"
        + (f", {abstract_count} from abstracts only" if abstract_count else "")
        + ". The concepts above are ordered from foundational to frontier — "
        "following this sequence gives you the chronological context to understand "
        "how this paper's contributions fit into the research landscape."
    )

    return "\n".join(lines)
