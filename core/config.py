import os
from dotenv import load_dotenv

load_dotenv()  # loads .env before any env-var reads below

# ── LLM Provider ───────────────────────────────────────────────────────────
# LLM_PROVIDER: "groq" | "openai" | "auto"
# auto = use Groq if GROQ_API_KEY is set, else OpenAI if OPENAI_API_KEY is set
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").lower().strip()

# ── Model names per provider ───────────────────────────────────────────────
MODELS = {
    "groq": {
        "heavy": "llama-3.3-70b-versatile",  # writing agent, ordering
        "light": "llama-3.1-8b-instant",  # structural tasks, cheap calls
    },
    "openai": {
        "heavy": "gpt-4o",  # writing agent, ordering
        "light": "gpt-4o-mini",  # structural tasks, cheap calls
    },
}


def _resolve_provider() -> str:
    """Determine the active provider from env vars."""
    if LLM_PROVIDER in ("groq", "openai"):
        return LLM_PROVIDER
    # auto: prefer Groq, fall back to OpenAI
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "groq"  # will fail at call time with a clear error


ACTIVE_PROVIDER = _resolve_provider()
LLM_HEAVY = MODELS[ACTIVE_PROVIDER]["heavy"]
LLM_LIGHT = MODELS[ACTIVE_PROVIDER]["light"]

# ── Embeddings + reranking (always local, provider-independent) ────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Graph construction ─────────────────────────────────────────────────────
MAX_REFERENCE_LEVEL = 3
MAX_REFS_PER_PAPER = 80
FOUNDATION_TOP_K = 10
FRONTIER_YEAR_CUTOFF = 2022

# ── Gap detection ──────────────────────────────────────────────────────────
SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_MIN = 2
CONFIDENCE_THRESHOLD = 0.5

# ── Candidate scoring weights ──────────────────────────────────────────────
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2

LAYER_MATCH_SCORES = {
    "foundation": {"foundation": 1.0, "development": 0.5, "frontier": 0.2},
    "development": {"foundation": 0.4, "development": 1.0, "frontier": 0.6},
    "frontier": {"foundation": 0.1, "development": 0.4, "frontier": 1.0},
}

# ── Retrieval ──────────────────────────────────────────────────────────────
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RERANK_TOP_N = 5
CROSS_ENCODER_THRESHOLD = 0.3

# ── Generation ────────────────────────────────────────────────────────────
MAX_MULTIHOP_DEPTH = 2
MAX_EVAL_LOOPS = 2
FAITHFULNESS_GATE = 0.70
CONTEXT_RECALL_GATE = 0.60
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# ── Cache ──────────────────────────────────────────────────────────────────
CACHE_DB = os.environ.get("KGMS_CACHE_DB", "kgms_cache.sqlite")

# ── Unpaywall ──────────────────────────────────────────────────────────────
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "kgms@research.edu")
# ── Semantic Scholar ────────────────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,openAccessPdf"
S2_REF_FIELDS = "title,abstract,year,citationCount,externalIds,openAccessPdf"
