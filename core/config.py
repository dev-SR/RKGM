import os

# ── LLM model routing ──────────────────────────────────────────────────────
LLM_HEAVY = "llama-3.3-70b-versatile"  # Groq: Writing Agent, ordering
LLM_LIGHT = "llama-3.1-8b-instant"  # Groq: structural tasks, cheap calls
EMBED_MODEL = "all-MiniLM-L6-v2"  # local sentence-transformers
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local cross-encoder

# ── Graph construction ─────────────────────────────────────────────────────
MAX_REFERENCE_LEVEL = 3
MAX_REFS_PER_PAPER = 80  # cap per paper to prevent explosion
FOUNDATION_TOP_K = 10  # top-K by trendscore → Foundation layer
FRONTIER_YEAR_CUTOFF = 2022  # papers >= this year → Frontier (if not Foundation)

# ── Gap detection ──────────────────────────────────────────────────────────
SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_MIN = 2  # gap must appear in >= 2/3 runs to be kept
CONFIDENCE_THRESHOLD = 0.5  # below this AND ungrounded → discard

# ── Candidate scoring weights ──────────────────────────────────────────────
ALPHA = 0.5  # semantic similarity (embedding cosine)
BETA = 0.3  # normalised trendscore
GAMMA = 0.2  # layer-match bonus

LAYER_MATCH_SCORES = {
    "foundation": {"foundation": 1.0, "development": 0.5, "frontier": 0.2},
    "development": {"foundation": 0.4, "development": 1.0, "frontier": 0.6},
    "frontier": {"foundation": 0.1, "development": 0.4, "frontier": 1.0},
}

# ── Retrieval ──────────────────────────────────────────────────────────────
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RERANK_TOP_N = 5
CROSS_ENCODER_THRESHOLD = 0.3  # below this → abstract-only fallback

# ── Generation ────────────────────────────────────────────────────────────
MAX_MULTIHOP_DEPTH = 2
MAX_EVAL_LOOPS = 2
FAITHFULNESS_GATE = 0.70
CONTEXT_RECALL_GATE = 0.60
CHUNK_SIZE = 400  # words per chunk
CHUNK_OVERLAP = 80  # word overlap between chunks

# ── Cache ──────────────────────────────────────────────────────────────────
CACHE_DB = os.environ.get("KGMS_CACHE_DB", "kgms_cache.sqlite")

# ── Semantic Scholar ────────────────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,openAccessPdf"
S2_REF_FIELDS = "title,abstract,year,citationCount,externalIds,openAccessPdf"

# ── Unpaywall ──────────────────────────────────────────────────────────────
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "kgms@research.edu")
