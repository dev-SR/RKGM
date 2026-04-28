"""
Hybrid retrieval pipeline.

Dense (ChromaDB cosine) + Sparse (BM25) → RRF merge → Cross-encoder rerank

Design decisions:
- Dense captures semantic relevance (concept paraphrases, related vocabulary)
- BM25 captures exact-term relevance (technical jargon, paper titles, acronyms)
- RRF (Reciprocal Rank Fusion): rank-based merge — no score normalisation needed
  because dense and sparse scores are on incomparable scales
- Cross-encoder as final stage: scores (query, passage) jointly, far more accurate
  than bi-encoder cosine for gap-specific relevance. Runs locally, zero API cost.
- Abstract-only fallback: if no chunk passes the confidence threshold, the gap is
  flagged — never silently degraded.
"""

import numpy as np
import bm25s
from sentence_transformers import CrossEncoder

from core.models import KnowledgeGap, Chunk
from core.config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RERANK_TOP_N,
    CROSS_ENCODER_THRESHOLD,
    RERANK_MODEL,
)
from utils.embedder import embed_single

# Singleton cross-encoder (loaded once per process)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def hybrid_retrieve(
    gap: KnowledgeGap,
    collection,  # ChromaDB collection
    all_chunks: list[Chunk],
    bm25_retriever=None,  # pre-built BM25Retriever, built lazily if None
    top_k_override: int | None = None,  # override RERANK_TOP_N (e.g. 15 for gate retry)
) -> tuple[list[Chunk], bool]:
    """
    Full retrieval pipeline for a single gap.

    Returns:
        (chunks, is_abstract_only)
        is_abstract_only=True means no chunk passed the cross-encoder threshold
    """
    if not collection or not all_chunks:
        return [], True

    query = gap.retrieval_query
    chunk_map = {c.chunk_id: c for c in all_chunks}

    # ── 1. Dense retrieval ─────────────────────────────────────────────────
    q_embed = embed_single(query).tolist()
    dense_result = collection.query(
        query_embeddings=[q_embed],
        n_results=min(DENSE_TOP_K, len(all_chunks)),
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_result["ids"][0] if dense_result["ids"] else []

    # ── 2. Sparse BM25 ─────────────────────────────────────────────────────
    sparse_ids = _bm25_retrieve(query, all_chunks, top_k=SPARSE_TOP_K)

    # ── 3. Reciprocal Rank Fusion ──────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf(rank)
    for rank, cid in enumerate(sparse_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf(rank)

    # Take top-30 by RRF for reranking
    top_by_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    candidates = [chunk_map[cid] for cid in top_by_rrf if cid in chunk_map][:30]

    if not candidates:
        return [], True

    # ── 4. Cross-encoder reranking ─────────────────────────────────────────
    reranker = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    ce_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(ce_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    # Filter by threshold
    final_n = top_k_override if top_k_override else RERANK_TOP_N
    passing = [
        (score, chunk) for score, chunk in ranked if score >= CROSS_ENCODER_THRESHOLD
    ]

    if not passing:
        return [c for _, c in ranked[:2]], True

    final_chunks = [c for _, c in passing[:final_n]]
    return final_chunks, False


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank (0-indexed)."""
    return 1.0 / (k + rank + 1)


def _bm25_retrieve(query: str, chunks: list[Chunk], top_k: int) -> list[str]:
    """
    Build a BM25 index over all chunks and retrieve top-k chunk IDs.
    bm25s is pure-Python and fast for hundreds of chunks.
    """
    try:
        corpus = [c.text.lower() for c in chunks]
        tokenized = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokenized)

        query_tokens = bm25s.tokenize([query.lower()], stopwords="en")
        results, _ = retriever.retrieve(query_tokens, k=min(top_k, len(chunks)))

        # results is a 2D array of indices into the corpus
        indices = results[0].tolist()
        return [chunks[i].chunk_id for i in indices if i < len(chunks)]
    except Exception as e:
        print(f"[retrieval] BM25 failed: {e}")
        return []


def verbalize_path(
    gap: KnowledgeGap,
    chunks: list[Chunk],
    papers: dict,
) -> str:
    """
    GNN-RAG-style path verbalization.
    Shows the structural chain from target paper to each source paper.
    Included in the Writing Agent prompt for citation context.
    """
    lines = ["[Target Paper]"]
    seen_papers: set[str] = set()

    for chunk in chunks:
        pid = chunk.paper_id
        if pid in seen_papers:
            continue
        seen_papers.add(pid)

        paper = papers.get(pid)
        if not paper:
            continue

        lines.append(
            f"  → cites ({paper.layer.value} layer, Level {paper.level}): "
            f'"{paper.title[:70]}" ({paper.year})'
        )

    return "\n".join(lines)


def format_passages(chunks: list[Chunk], papers: dict) -> str:
    """
    Format retrieved chunks for inclusion in the Writing Agent prompt.
    Each chunk is prefixed with its citation ID so the LLM can reference it.
    """
    parts = []
    for c in chunks:
        paper = papers.get(c.paper_id)
        title = paper.title if paper else c.paper_id
        year = paper.year if paper else ""
        parts.append(
            f"[cite: {c.chunk_id}]\n"
            f'Source: "{title}" ({year}) — Section: {c.section}\n'
            f"{c.text}"
        )
    return "\n\n" + ("─" * 60) + "\n\n".join(parts)
