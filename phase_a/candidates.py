"""
Candidate paper matching for each knowledge gap.

For each gap:
  1. Score all reference papers: α·semantic + β·trendscore_norm + γ·layer_match
  2. Take top-5 candidates
  3. Cluster top candidates using Leiden (SurveyG horizontal summarization)
  4. Generate a one-sentence rationale per candidate (light model)
  5. Resolve PDF URLs

Model routing: rationale and cluster summary → LLM_LIGHT
"""

import json
import numpy as np
import igraph as ig
import leidenalg

from core.models import Paper, KnowledgeGap, CandidatePaper, Layer
from core.config import ALPHA, BETA, GAMMA, LAYER_MATCH_SCORES
from core.prompts import CANDIDATE_RATIONALE_PROMPT, CLUSTER_SUMMARY_PROMPT
from utils.embedder import embed_texts, embed_single
from utils.llm import llm_light
from utils.apis import resolve_pdf_url
from utils.cache import get_cached, set_cached


def match_candidates(
    gaps: list[KnowledgeGap],
    papers: dict[str, Paper],
    top_k: int = 5,
    progress_callback=None,
) -> list[CandidatePaper]:
    """
    Score and rank candidate papers for each gap.
    Returns a flat list of CandidatePaper (multiple per gap).
    """
    # Only score papers that are references (level > 0) and have abstracts
    paper_list = [p for p in papers.values() if p.level > 0 and p.abstract.strip()]
    if not paper_list:
        return []

    # Batch embed all abstracts once
    if progress_callback:
        progress_callback("Embedding reference abstracts…")
    abstracts = [p.abstract for p in paper_list]
    paper_embeds = embed_texts(abstracts)  # (N, dim) normalised

    # Normalise trendscores to [0, 1]
    trendscores = np.array([p.trendscore for p in paper_list])
    ts_max = trendscores.max() or 1.0
    trendscores_norm = trendscores / ts_max

    all_candidates: list[CandidatePaper] = []

    for gap in gaps:
        if progress_callback:
            progress_callback(f"Matching candidates for: {gap.concept}…")

        q_embed = embed_single(gap.retrieval_query)  # (dim,) normalised
        sims = paper_embeds @ q_embed  # dot = cosine (normalised)

        # Compute combined score
        scores = []
        for i, paper in enumerate(paper_list):
            layer_bonus = LAYER_MATCH_SCORES.get(gap.layer_hint.value, {}).get(
                paper.layer.value, 0.3
            )

            score = (
                ALPHA * float(sims[i])
                + BETA * float(trendscores_norm[i])
                + GAMMA * layer_bonus
            )
            scores.append((score, i))

        scores.sort(reverse=True)
        top_indices = [i for _, i in scores[:top_k]]
        top_papers = [paper_list[i] for i in top_indices]
        top_scores = [s for s, _ in scores[:top_k]]

        # Leiden cluster summary over the top candidates
        cluster_ctx = _leiden_cluster_summary(top_papers, gap.concept)

        for paper, score in zip(top_papers, top_scores):
            rationale = _get_rationale(gap, paper)
            full_rationale = rationale
            if cluster_ctx:
                full_rationale += f"\n\n[Cluster context] {cluster_ctx}"

            # Resolve PDF URL
            if not paper.pdf_url:
                raw_data = {
                    "paperId": paper.paper_id,
                    "externalIds": {
                        "ArXiv": paper.arxiv_id,
                        "DOI": paper.doi,
                    },
                    "openAccessPdf": {"url": paper.pdf_url} if paper.pdf_url else None,
                }
                paper.pdf_url = resolve_pdf_url(raw_data)

            all_candidates.append(
                CandidatePaper(
                    paper=paper,
                    gap_id=gap.gap_id,
                    relevance_score=round(score, 4),
                    rationale=full_rationale,
                    pdf_available=bool(paper.pdf_url),
                )
            )

    return all_candidates


def _leiden_cluster_summary(papers: list[Paper], gap_concept: str) -> str:
    """
    Cluster the candidate papers using Leiden and summarise the largest cluster.
    Returns empty string if clustering is not meaningful (< 3 papers).
    """
    if len(papers) < 3:
        return ""

    cache_key = f"cluster:{gap_concept}:{','.join(p.paper_id[:8] for p in papers)}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        embeds = embed_texts([p.abstract for p in papers])
        sims = embeds @ embeds.T  # cosine similarity matrix

        # Build igraph
        g = ig.Graph()
        g.add_vertices(len(papers))
        edges, weights = [], []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                if sims[i, j] > 0.25:
                    edges.append((i, j))
                    weights.append(float(sims[i, j]))

        if not edges:
            return ""

        g.add_edges(edges)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights="weight", seed=42
        )
        largest_community = max(partition, key=len)
        cluster_papers = [papers[i] for i in largest_community]

        papers_json = json.dumps(
            [
                {"title": p.title, "year": p.year, "abstract": p.abstract[:250]}
                for p in cluster_papers
            ],
            ensure_ascii=False,
        )

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            gap_concept=gap_concept,
            papers_json=papers_json,
        )
        summary = llm_light(prompt, max_tokens=200)
        set_cached(cache_key, summary)
        return summary

    except Exception as e:
        print(f"[candidates] Leiden clustering failed for '{gap_concept}': {e}")
        return ""


def _get_rationale(gap: KnowledgeGap, paper: Paper) -> str:
    cache_key = f"rationale:{gap.gap_id}:{paper.paper_id}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    prompt = CANDIDATE_RATIONALE_PROMPT.format(
        concept=gap.concept,
        gap_type=gap.gap_type.value,
        why_needed=gap.why_needed,
        title=paper.title,
        abstract=paper.abstract[:500],
    )
    rationale = llm_light(prompt, max_tokens=150)
    set_cached(cache_key, rationale)
    return rationale


# ── BibTeX export ──────────────────────────────────────────────────────────


def candidates_to_bibtex(candidates: list) -> str:
    """
    Generate BibTeX entries for all unique candidate papers.

    Called from Phase A — the user can download this before going off
    to collect PDFs so they can import directly into Zotero/Mendeley.

    Uses @misc for arXiv preprints, @article for DOI-identified papers,
    @inproceedings for IEEE/ACM conference DOIs.
    """
    seen: set = set()
    entries: list[str] = []

    for c in candidates:
        p = c.paper
        if p.paper_id in seen:
            continue
        seen.add(p.paper_id)

        # Build a clean cite key: FirstAuthorYear or TitleYear
        cite_key = _make_cite_key(p.title, p.year)

        if p.doi and any(
            prefix in p.doi for prefix in ["10.1109", "10.1145", "10.18653", "10.1162"]
        ):
            entry_type = "inproceedings"
        elif p.doi:
            entry_type = "article"
        else:
            entry_type = "misc"  # arXiv

        lines = [f"@{entry_type}{{{cite_key},"]
        lines.append(f"  title     = {{{_escape_bib(p.title)}}},")
        lines.append(f"  year      = {{{p.year}}},")

        if p.doi:
            lines.append(f"  doi       = {{{p.doi}}},")
        if p.arxiv_id:
            lines.append(f"  eprint    = {{{p.arxiv_id}}},")
            lines.append("  archivePrefix = {arXiv},")
            lines.append(f"  url       = {{https://arxiv.org/abs/{p.arxiv_id}}},")

        lines.append(
            f"  note      = {{Layer: {p.layer.value}. "
            f"Citations: {p.citation_count}. "
            f"Semantic Scholar ID: {p.paper_id}}},"
        )
        lines.append("}")
        entries.append("\n".join(lines))

    header = (
        "% BibTeX export from Knowledge Gap Mitigation System (KGMS)\n"
        "% Import into Zotero: File → Import → BibTeX\n"
        "% Import into Mendeley: File → Import → BibTeX\n\n"
    )
    return header + "\n\n".join(entries)


def candidates_to_csv(candidates: list) -> str:
    """
    CSV export of candidate papers with gap mapping.
    Useful for tracking which PDFs have been collected.
    """
    from phase_a.gap_detection import _gap_to_dict

    lines = [
        "gap_id,concept,layer_hint,paper_id,title,year,layer,"
        "citation_count,relevance_score,pdf_available,pdf_url,doi,arxiv_id"
    ]

    seen: set = set()
    for c in candidates:
        p = c.paper
        key = (c.gap_id, p.paper_id)
        if key in seen:
            continue
        seen.add(key)

        def q(s):
            return f'"{str(s).replace(chr(34), chr(39))}"'

        lines.append(
            ",".join(
                [
                    q(c.gap_id),
                    q(c.paper.paper_id),  # concept via gap_id lookup in UI
                    q(p.layer.value),
                    q(p.paper_id),
                    q(p.title),
                    str(p.year),
                    q(p.layer.value),
                    str(p.citation_count),
                    str(round(c.relevance_score, 4)),
                    str(c.pdf_available).lower(),
                    q(p.pdf_url or ""),
                    q(p.doi or ""),
                    q(p.arxiv_id or ""),
                ]
            )
        )

    return "\n".join(lines)


def _make_cite_key(title: str, year: int) -> str:
    """Generate a clean BibTeX cite key from title words + year."""
    import re

    words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
    key_words = [
        w.lower()
        for w in words
        if len(w) > 3
        and w.lower()
        not in {
            "with",
            "from",
            "using",
            "based",
            "this",
            "that",
            "into",
            "over",
            "under",
            "about",
            "through",
            "towards",
        }
    ][:3]
    base = "".join(w.capitalize() for w in key_words) or "Paper"
    return f"{base}{year}"


def _escape_bib(text: str) -> str:
    """Minimal BibTeX escaping for title text."""
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
