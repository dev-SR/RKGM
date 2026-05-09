"""
Reference graph construction and hierarchical layer assignment.

Builds a directed NetworkX graph where:
  - Nodes = papers (BA + all references up to Level 3)
  - Edges = citation links, weighted by abstract cosine similarity
  - Node attributes = Paper dataclass + layer assignment

Layer assignment (SurveyG-inspired):
  Foundation  = top-K by trendscore (citation_count / years_old)
  Frontier    = year >= FRONTIER_YEAR_CUTOFF and not Foundation
  Development = everything else
"""

import datetime
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from core.models import Paper, Layer
from core.config import MAX_REFERENCE_LEVEL, FOUNDATION_TOP_K, FRONTIER_YEAR_CUTOFF
from utils.apis import fetch_paper, fetch_references
from utils.embedder import embed_texts


def build_reference_graph(
    ba_paper_id: str,
    progress_callback=None,
    max_depth: int | None = None,  # override MAX_REFERENCE_LEVEL (1, 2, or 3)
) -> tuple[nx.DiGraph, dict[str, Paper]]:
    """
    Build the multi-level reference graph.

    Args:
        ba_paper_id:       Semantic Scholar paper ID of the base article
        progress_callback: optional callable(str) for status messages
        max_depth:         how many reference levels to fetch (1–3).
                           Defaults to MAX_REFERENCE_LEVEL from config.
                           Level 1 = fast (direct references only)
                           Level 3 = thorough but slow on first run (cached after)

    Returns:
        G      — directed NetworkX graph
        papers — dict of paper_id → Paper
    """
    depth = max_depth if max_depth is not None else MAX_REFERENCE_LEVEL
    depth = max(1, min(3, depth))  # clamp to [1, 3]
    G = nx.DiGraph()
    papers: dict[str, Paper] = {}
    queue = [(ba_paper_id, 0)]
    visited: set[str] = set()

    while queue:
        pid, level = queue.pop(0)
        if pid in visited or level > depth:
            continue
        visited.add(pid)

        if progress_callback:
            progress_callback(f"Fetching paper {pid[:12]}... at level {level}")

        raw = fetch_paper(pid)
        if not raw or not raw.get("abstract"):
            continue

        paper = _raw_to_paper(raw, level)
        papers[paper.paper_id] = paper
        G.add_node(paper.paper_id, paper=paper)

        if level < depth:
            refs = fetch_references(pid)
            for ref in refs:
                ref_id = ref.get("paperId")
                if ref_id and ref_id not in visited:
                    G.add_edge(pid, ref_id)
                    queue.append((ref_id, level + 1))

    if not papers:
        raise ValueError(f"Could not fetch paper: {ba_paper_id}")

    # Batch-embed all abstracts and compute edge weights
    _add_edge_weights(G, papers)

    # Assign layers
    _assign_layers(papers)

    return G, papers


def _raw_to_paper(raw: dict, level: int) -> Paper:
    ext = raw.get("externalIds") or {}
    oap = raw.get("openAccessPdf") or {}
    return Paper(
        paper_id=raw["paperId"],
        title=raw.get("title") or "",
        abstract=raw.get("abstract") or "",
        year=raw.get("year") or 0,
        citation_count=raw.get("citationCount") or 0,
        arxiv_id=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        pdf_url=oap.get("url"),
        level=level,
    )


def _add_edge_weights(G: nx.DiGraph, papers: dict[str, Paper]):
    """
    Compute cosine similarity between connected paper abstracts.
    Single batch embed — not per-edge.
    """
    ids = list(papers.keys())
    texts = [papers[i].abstract for i in ids]
    embeds = embed_texts(texts)  # shape: (N, dim)
    embed_map = dict(zip(ids, embeds))

    for u, v in G.edges():
        if u in embed_map and v in embed_map:
            sim = float(np.dot(embed_map[u], embed_map[v]))  # already L2-normalised
            G[u][v]["weight"] = max(0.0, sim)


def _assign_layers(papers: dict[str, Paper]):
    """
    Mutates papers in-place, setting .layer and .trendscore.
    """
    current_year = datetime.datetime.now().year

    for p in papers.values():
        years_old = max(1, current_year - p.year) if p.year > 0 else 10
        p.trendscore = p.citation_count / years_old

    sorted_papers = sorted(papers.values(), key=lambda x: x.trendscore, reverse=True)
    foundation_ids = {p.paper_id for p in sorted_papers[:FOUNDATION_TOP_K]}

    for p in papers.values():
        if p.paper_id in foundation_ids:
            p.layer = Layer.FOUNDATION
        elif p.year >= FRONTIER_YEAR_CUTOFF:
            p.layer = Layer.FRONTIER
        else:
            p.layer = Layer.DEVELOPMENT


def graph_summary(G: nx.DiGraph, papers: dict[str, Paper]) -> dict:
    """Return a dict of stats for display."""
    layers = {Layer.FOUNDATION: 0, Layer.DEVELOPMENT: 0, Layer.FRONTIER: 0}
    levels = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in papers.values():
        layers[p.layer] += 1
        l = min(p.level, 3)
        levels[l] += 1
    return {
        "total_papers": len(papers),
        "total_edges": G.number_of_edges(),
        "foundation": layers[Layer.FOUNDATION],
        "development": layers[Layer.DEVELOPMENT],
        "frontier": layers[Layer.FRONTIER],
        "by_level": levels,
    }
