"""
PDF ingestion pipeline.

PDF → marker-pdf (Markdown) → section-aware chunking → batch embed → ChromaDB

Design decisions:
- marker-pdf as primary parser: trained on academic papers, handles two-column
  layouts and LaTeX. PyMuPDF as fallback for robustness.
- Section-first chunking: split on ## / ### headers first, then sliding window
  within each section. Ensures chunks are semantically coherent units.
- 400-word chunks with 80-word overlap: prevents key explanations from being
  split across chunk boundaries while keeping chunks short enough for the
  cross-encoder to score accurately.
- ChromaDB in-memory: no server needed, resets cleanly per session.
- Single batch embed call: 10-20x faster than per-chunk embedding.
"""

import os
import re
import subprocess
import hashlib
from pathlib import Path

import fitz  # PyMuPDF fallback
import chromadb
import numpy as np

from core.models import Paper, Chunk
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.embedder import embed_texts
from utils.cache import get_cached, set_cached


# ── PDF → Markdown ─────────────────────────────────────────────────────────


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF to clean Markdown using marker-pdf.
    Falls back to PyMuPDF if marker-pdf is not installed or fails.
    Caches the Markdown by PDF content hash so re-runs are instant.
    """
    content_hash = _file_hash(pdf_path)
    cache_key = f"pdf_md:{content_hash}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    # Try marker-pdf first
    md = _marker_parse(pdf_path)
    if not md:
        md = _pymupdf_parse(pdf_path)

    if md:
        set_cached(cache_key, md)
    return md or ""


def _marker_parse(pdf_path: str) -> str:
    """
    Call marker_single CLI. Outputs Markdown to /tmp/<stem>/<stem>.md
    marker-pdf install: pip install marker-pdf
    """
    try:
        stem = Path(pdf_path).stem
        out_dir = f"/tmp/kgms_marker_{stem}"
        os.makedirs(out_dir, exist_ok=True)

        result = subprocess.run(
            [
                "marker_single",
                pdf_path,
                out_dir,
                "--batch_multiplier",
                "1",
                "--langs",
                "English",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # marker outputs to out_dir/<stem>/<stem>.md
        md_path = Path(out_dir) / stem / f"{stem}.md"
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")

        # Alternate output path (some marker versions)
        md_path2 = Path(out_dir) / f"{stem}.md"
        if md_path2.exists():
            return md_path2.read_text(encoding="utf-8")

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print(f"[ingest] marker-pdf unavailable ({type(e).__name__}), using PyMuPDF")
    return ""


def _pymupdf_parse(pdf_path: str) -> str:
    """
    PyMuPDF fallback. Extracts text page-by-page.
    Inserts page breaks as lightweight section markers.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"## Page {i + 1}\n\n{text}")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[ingest] PyMuPDF failed for {pdf_path}: {e}")
        return ""


# ── Markdown → Chunks ──────────────────────────────────────────────────────


def chunk_markdown(md: str, paper: Paper) -> list[Chunk]:
    """
    Three-pass chunking (Paper Circle SemanticChunker-inspired):

    Pass 1: Extract special elements as standalone chunks BEFORE prose splitting.
      - Block equations ($$...$$)  → chunk type "equation"
      - Markdown tables            → chunk type "table"
      - Figure/table captions      → chunk type "figure"

    This matters for mathematical gaps: an equation and its explanation
    are often split across chunks by a naive sliding window, making the
    cross-encoder unable to score them together. Preserving them intact
    solves this class of retrieval failure.

    Pass 2: Split remaining prose by section headers (## / ###).

    Pass 3: Sliding window within each section (CHUNK_SIZE words, CHUNK_OVERLAP).

    Returns Chunk objects — embeddings are added in batch by build_index().
    """
    chunks: list[Chunk] = []
    counter = [0]  # mutable for inner helper

    def _make_chunk(text: str, section: str, chunk_type: str = "prose") -> Chunk:
        c = Chunk(
            chunk_id=f"{paper.paper_id}::{counter[0]:04d}",
            paper_id=paper.paper_id,
            section=f"{section} [{chunk_type}]" if chunk_type != "prose" else section,
            text=text.strip(),
            page=0,
        )
        counter[0] += 1
        return c

    # ── Pass 1: extract special elements ──────────────────────────────────

    # 1a. Block equations  $$...$$  (possibly multiline)
    eq_pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    for m in eq_pattern.finditer(md):
        eq_text = m.group(0).strip()
        # Include ±2 lines of surrounding context so the chunk is self-contained
        ctx = _surrounding_context(md, m.start(), m.end(), context_chars=300)
        full = f"Equation:\n{eq_text}\n\nContext:\n{ctx}".strip()
        if len(full) > 40:
            chunks.append(_make_chunk(full, "equation", "equation"))
    # Remove equations from the Markdown so they don't also go through Pass 2/3
    md_clean = eq_pattern.sub("[EQUATION_EXTRACTED]", md)

    # 1b. Markdown tables  (lines starting with |)
    table_pattern = re.compile(r"(?:(?:^\|.+\|[ \t]*\n)+)", re.MULTILINE)
    for m in table_pattern.finditer(md_clean):
        table_text = m.group(0).strip()
        if table_text.count("\n") < 1:  # skip single-line non-tables
            continue
        ctx = _surrounding_context(md_clean, m.start(), m.end(), context_chars=200)
        full = f"Table:\n{table_text}\n\nContext:\n{ctx}".strip()
        chunks.append(_make_chunk(full, "table", "table"))
    md_clean = table_pattern.sub("[TABLE_EXTRACTED]", md_clean)

    # 1c. Figure / table captions
    caption_pattern = re.compile(
        r"(?:Figure|Fig\.|Table)\s+\d+[:\.]?\s*.{10,300}", re.IGNORECASE
    )
    for m in caption_pattern.finditer(md_clean):
        caption = m.group(0).strip()
        chunks.append(_make_chunk(caption, "figure", "figure"))
    # Don't remove captions — they're short and also useful in prose context

    # ── Pass 2 & 3: section-aware prose chunking ───────────────────────────
    section_pattern = re.compile(r"\n(?=#{1,3} )", re.MULTILINE)
    raw_sections = section_pattern.split(md_clean)

    for section_text in raw_sections:
        if not section_text.strip():
            continue

        header_match = re.match(r"^(#{1,3})\s+(.+)", section_text.strip())
        section_name = header_match.group(2).strip() if header_match else "body"
        body = (
            section_text[header_match.end() :].strip() if header_match else section_text
        )

        if any(
            kw in section_name.lower()
            for kw in ["reference", "bibliography", "acknowledgment", "appendix"]
        ):
            continue

        # Strip placeholder tokens left from Pass 1
        body = re.sub(r"\[(?:EQUATION|TABLE)_EXTRACTED\]", "", body).strip()

        words = body.split()
        if len(words) < 30:
            continue

        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            if len(chunk_text.strip()) > 80:
                chunks.append(_make_chunk(chunk_text, section_name, "prose"))
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _surrounding_context(
    text: str, start: int, end: int, context_chars: int = 200
) -> str:
    """Return up to context_chars characters before and after [start, end]."""
    before = text[max(0, start - context_chars) : start].strip()
    after = text[end : end + context_chars].strip()
    parts = [p for p in [before, after] if p]
    return " … ".join(parts)


# ── Batch embed + ChromaDB ─────────────────────────────────────────────────


def build_index(
    chunks: list[Chunk],
    progress_callback=None,
) -> tuple[object, list[Chunk]]:
    """
    1. Batch-embed all chunks (single call)
    2. Build ChromaDB in-memory collection

    Returns (chroma_collection, chunks_with_embeddings)
    """
    if not chunks:
        return None, []

    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks…")

    texts = [c.text for c in chunks]
    embeds = embed_texts(texts)  # (N, dim) normalised float32

    for chunk, emb in zip(chunks, embeds):
        chunk.embedding = emb.tolist()

    if progress_callback:
        progress_callback("Building vector index…")

    client = chromadb.Client()  # in-memory
    # Delete if exists (clean slate per session)
    try:
        client.delete_collection("reference_papers")
    except Exception:
        pass

    collection = client.create_collection(
        name="reference_papers", metadata={"hnsw:space": "cosine"}
    )

    # ChromaDB add in batches of 500 to avoid memory issues
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=[c.embedding for c in batch],
            metadatas=[
                {"paper_id": c.paper_id, "section": c.section, "page": c.page}
                for c in batch
            ],
        )

    return collection, chunks


# ── Ingest from file paths ─────────────────────────────────────────────────


def ingest_pdfs(
    pdf_paths: dict[str, str],  # paper_id -> local pdf path
    papers: dict,  # paper_id -> Paper
    progress_callback=None,
) -> tuple[list[Chunk], object]:
    """
    Main entry point for Phase B ingestion.

    Args:
        pdf_paths:  dict mapping paper_id to local PDF file path
        papers:     the all_papers dict from Phase A

    Returns:
        (all_chunks, chroma_collection)
    """
    all_chunks: list[Chunk] = []

    for paper_id, pdf_path in pdf_paths.items():
        paper = papers.get(paper_id)
        if not paper:
            continue
        if not os.path.exists(pdf_path):
            print(f"[ingest] PDF not found: {pdf_path}")
            continue

        if progress_callback:
            progress_callback(f"Parsing: {paper.title[:50]}…")

        md = parse_pdf_to_markdown(pdf_path)
        if not md:
            print(f"[ingest] No text extracted from {pdf_path}")
            continue

        chunks = chunk_markdown(md, paper)
        all_chunks.extend(chunks)

    collection, all_chunks = build_index(all_chunks, progress_callback)
    return all_chunks, collection


# ── Helpers ────────────────────────────────────────────────────────────────


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
