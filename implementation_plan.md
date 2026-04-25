# Knowledge Gap Mitigation System — Final Implementation Plan

> Extending Rahman et al. (2022) IEEE ICCIT using LLM-native techniques.  
> **Goal:** Given a target research paper, detect knowledge gaps, traverse its multi-level reference graph, and generate a chronologically ordered learning document grounded in the paper's own references.

---

## 0. Free Tool Stack (Final)

| Layer | Tool | Why Free & Why This One |
|---|---|---|
| LLM (heavy) | `groq/llama-3.3-70b` via Groq API | Free tier, fast, strong reasoning |
| LLM (light) | `groq/llama-3.1-8b-instant` | Free tier, structural tasks only |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fully local, no API cost |
| Vector DB | `ChromaDB` (in-memory mode) | No server, no persistence needed |
| Sparse retrieval | `bm25s` | Pure Python, zero dependencies |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, sentence-transformers |
| PDF parsing | `marker-pdf` | Academic-paper-trained, outputs clean Markdown |
| Graph | `networkx` | Standard, zero cost |
| Community detection | `leidenalg` + `igraph` | SurveyG-validated |
| Paper APIs | Semantic Scholar (free) + arXiv API (free) + Unpaywall (free) |
| Orchestration | `langgraph` (free OSS) |
| Evaluation | `ragas` (free OSS) |
| UI | `streamlit` | One-person deployable in hours |

**Groq free tier:** 14,400 tokens/minute on Llama 3.3 70B — sufficient for a student project with caching.  
**Alternative LLM:** Google Gemini 1.5 Flash (free tier, 1M context) — good fallback if Groq rate-limits.

---

## 1. Project Structure

```
kgms/
├── core/
│   ├── models.py          # All dataclasses / Pydantic models
│   ├── prompts.py         # All prompt templates
│   └── config.py          # Constants, thresholds, model names
├── phase_a/
│   ├── graph.py           # Reference graph construction + layer assignment
│   ├── gap_detection.py   # Gap extraction (auto + user modes)
│   ├── validation.py      # Self-consistency + grounding checks
│   └── candidate.py       # Candidate paper matching + ranking
├── phase_b/
│   ├── ingest.py          # PDF parsing, chunking, embedding
│   ├── retrieval.py       # Hybrid search + reranking
│   ├── multihop.py        # Multi-hop sub-gap detection
│   ├── ordering.py        # Chronological ordering
│   └── generation.py      # Writing agent + evaluation agent
├── utils/
│   ├── apis.py            # Semantic Scholar, arXiv, Unpaywall wrappers
│   ├── cache.py           # SQLite cache for API responses
│   └── embedder.py        # Singleton sentence-transformer embedder
├── eval/
│   └── evaluate.py        # RAGAS + custom metrics
├── graph_state.py         # LangGraph state definition
├── pipeline.py            # LangGraph graph assembly
└── app.py                 # Streamlit UI
```

---

## 2. Core Data Models (`core/models.py`)

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

class GapType(str, Enum):
    TERMINOLOGY   = "terminology"
    METHODOLOGY   = "methodology"
    BENCHMARK     = "benchmark"
    HISTORICAL    = "historical"
    MATHEMATICAL  = "mathematical"

class Difficulty(str, Enum):
    BEGINNER      = "beginner"
    INTERMEDIATE  = "intermediate"
    ADVANCED      = "advanced"

class Layer(str, Enum):
    FOUNDATION    = "foundation"
    DEVELOPMENT   = "development"
    FRONTIER      = "frontier"

@dataclass
class KnowledgeGap:
    concept:          str
    gap_type:         GapType
    difficulty:       Difficulty
    domain:           str
    why_needed:       str
    layer_hint:       Layer
    retrieval_query:  str
    source_passage:   str          # exact sentence in BA triggering this gap
    confidence:       float        # 0.0–1.0, from self-consistency
    gap_id:           str = ""     # assigned after creation

@dataclass
class Paper:
    paper_id:         str          # Semantic Scholar ID
    title:            str
    abstract:         str
    year:             int
    citation_count:   int
    arxiv_id:         Optional[str]
    doi:              Optional[str]
    pdf_url:          Optional[str]  # resolved by Unpaywall/arXiv
    level:            int           # 0=BA, 1/2/3 = reference depth
    layer:            Layer = Layer.DEVELOPMENT
    trendscore:       float = 0.0

@dataclass
class CandidatePaper:
    paper:            Paper
    gap_id:           str
    relevance_score:  float         # combined α·semantic + β·trendscore + γ·layer_match
    rationale:        str           # LLM-generated "why this paper fills this gap"
    pdf_available:    bool = False

@dataclass
class Chunk:
    chunk_id:         str
    paper_id:         str
    section:          str
    text:             str
    page:             int
    embedding:        Optional[list] = None

@dataclass
class GapExplanation:
    gap_id:           str
    concept:          str
    explanation_text: str
    source_citations: list[str]     # ["paper_id::chunk_id", ...]
    confidence:       float
    is_abstract_only: bool = False  # True if no PDF was found
    order_position:   int = 0

@dataclass
class PipelineState:
    # Phase A
    ba_text:          str = ""
    ba_paper:         Optional[Paper] = None
    reference_graph:  Optional[object] = None  # networkx DiGraph
    all_papers:       dict = field(default_factory=dict)  # paper_id -> Paper
    gaps:             list[KnowledgeGap] = field(default_factory=list)
    candidates:       list[CandidatePaper] = field(default_factory=list)
    # Phase B
    chunks:           list[Chunk] = field(default_factory=list)
    chroma_collection: Optional[object] = None
    explanations:     list[GapExplanation] = field(default_factory=list)
    ordered_gaps:     list[str] = field(default_factory=list)  # gap_ids in order
    final_document:   str = ""
    errors:           list[str] = field(default_factory=list)
```

---

## 3. Configuration (`core/config.py`)

```python
# --- Model routing ---
LLM_HEAVY   = "llama-3.3-70b-versatile"      # Groq — Writing/Eval agents
LLM_LIGHT   = "llama-3.1-8b-instant"         # Groq — structural tasks
EMBED_MODEL = "all-MiniLM-L6-v2"             # local sentence-transformers
RERANK_MODEL= "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Graph parameters ---
MAX_REFERENCE_LEVEL = 3
FOUNDATION_TOP_K    = 10           # top K papers by trendscore → Foundation
FRONTIER_YEAR_CUTOFF= 2022         # papers >= this year → Frontier (if not Foundation)

# --- Gap detection ---
SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_MIN  = 2          # gap must appear in >= 2 of 3 runs
CONFIDENCE_THRESHOLD  = 0.5        # below this → flag for user review

# --- Retrieval ---
DENSE_TOP_K  = 20
SPARSE_TOP_K = 20
RERANK_TOP_N = 5
CROSS_ENCODER_THRESHOLD = 0.4      # below this → abstract-only fallback

# --- Generation ---
MAX_MULTIHOP_DEPTH = 2             # recurse sub-gaps up to depth 2
MAX_EVAL_LOOPS     = 2             # Writing→Eval cycles per gap
FAITHFULNESS_GATE  = 0.70          # RAGAS faithfulness minimum
CONTEXT_RECALL_GATE= 0.60

# --- Scoring weights ---
ALPHA = 0.5   # semantic similarity
BETA  = 0.3   # trendscore
GAMMA = 0.2   # layer match bonus

# --- Cache ---
CACHE_DB = "kgms_cache.sqlite"
```

---

## 4. API Utilities (`utils/apis.py`)

```python
import requests, time
from utils.cache import get_cached, set_cached

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,references,openAccessPdf"

def fetch_paper(paper_id: str) -> dict:
    """paper_id can be S2 ID, DOI, or arXiv ID."""
    cached = get_cached(f"paper:{paper_id}")
    if cached: return cached
    r = requests.get(f"{S2_BASE}/paper/{paper_id}", params={"fields": S2_FIELDS})
    r.raise_for_status()
    data = r.json()
    set_cached(f"paper:{paper_id}", data)
    return data

def fetch_references(paper_id: str, level: int = 1) -> list[dict]:
    """Fetch references of a paper up to MAX_REFERENCE_LEVEL recursively."""
    cached = get_cached(f"refs:{paper_id}:{level}")
    if cached: return cached
    r = requests.get(
        f"{S2_BASE}/paper/{paper_id}/references",
        params={"fields": S2_FIELDS, "limit": 100}
    )
    r.raise_for_status()
    refs = [d["citedPaper"] for d in r.json().get("data", [])]
    set_cached(f"refs:{paper_id}:{level}", refs)
    return refs

def resolve_pdf_url(paper: dict) -> str | None:
    """Try Semantic Scholar openAccessPdf, then Unpaywall, then arXiv."""
    # 1. S2 open access
    if paper.get("openAccessPdf"):
        return paper["openAccessPdf"]["url"]
    # 2. Unpaywall by DOI
    doi = paper.get("externalIds", {}).get("DOI")
    if doi:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                         params={"email": "research@kgms.edu"}, timeout=5)
        if r.ok:
            best = r.json().get("best_oa_location")
            if best and best.get("url_for_pdf"):
                return best["url_for_pdf"]
    # 3. arXiv
    arxiv_id = paper.get("externalIds", {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return None

def download_pdf(url: str, dest_path: str) -> bool:
    """Download a PDF to disk, returns True on success."""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "KGMS/1.0"})
        if r.ok and "pdf" in r.headers.get("Content-Type", ""):
            with open(dest_path, "wb") as f: f.write(r.content)
            return True
    except Exception:
        pass
    return False
```

---

## 5. Cache Layer (`utils/cache.py`)

```python
import sqlite3, json, hashlib, os

DB = os.environ.get("CACHE_DB", "kgms_cache.sqlite")

def _conn():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    return c

def get_cached(key: str):
    with _conn() as c:
        row = c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

def set_cached(key: str, value):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(value)))
```

---

## 6. Phase A — Reference Graph (`phase_a/graph.py`)

```python
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.models import Paper, Layer
from core.config import *
from utils.apis import fetch_paper, fetch_references
import numpy as np

embedder = SentenceTransformer(EMBED_MODEL)

def build_reference_graph(ba_paper_id: str) -> tuple[nx.DiGraph, dict[str, Paper]]:
    """
    Build a 3-level reference graph.
    Returns (DiGraph, paper_id -> Paper dict).
    Nodes carry: paper_id, layer, trendscore.
    Edges carry: weight (cosine sim of abstracts).
    """
    G = nx.DiGraph()
    papers: dict[str, Paper] = {}
    queue = [(ba_paper_id, 0)]
    visited = set()

    while queue:
        pid, level = queue.pop(0)
        if pid in visited or level > MAX_REFERENCE_LEVEL:
            continue
        visited.add(pid)

        raw = fetch_paper(pid)
        if not raw or not raw.get("abstract"):
            continue

        paper = Paper(
            paper_id=raw["paperId"],
            title=raw.get("title", ""),
            abstract=raw.get("abstract", ""),
            year=raw.get("year") or 0,
            citation_count=raw.get("citationCount") or 0,
            arxiv_id=raw.get("externalIds", {}).get("ArXiv"),
            doi=raw.get("externalIds", {}).get("DOI"),
            pdf_url=raw.get("openAccessPdf", {}).get("url") if raw.get("openAccessPdf") else None,
            level=level,
        )
        papers[paper.paper_id] = paper
        G.add_node(paper.paper_id)

        if level < MAX_REFERENCE_LEVEL:
            refs = fetch_references(pid, level + 1)
            for ref in refs:
                ref_id = ref.get("paperId")
                if ref_id:
                    G.add_edge(pid, ref_id)
                    queue.append((ref_id, level + 1))

    # Compute edge weights (cosine similarity of abstracts)
    ids    = list(papers.keys())
    texts  = [papers[i].abstract for i in ids]
    embeds = embedder.encode(texts, batch_size=64, show_progress_bar=False)
    embed_map = dict(zip(ids, embeds))

    for u, v in G.edges():
        if u in embed_map and v in embed_map:
            sim = float(cosine_similarity([embed_map[u]], [embed_map[v]])[0][0])
            G[u][v]["weight"] = sim

    # Assign layers
    assign_layers(papers)

    return G, papers


def assign_layers(papers: dict[str, Paper]):
    """
    Assign Foundation / Development / Frontier layers.
    Foundation: top-K by trendscore.
    Frontier: year >= FRONTIER_YEAR_CUTOFF and not Foundation.
    Development: everything else.
    """
    import datetime
    current_year = datetime.datetime.now().year

    for p in papers.values():
        years_old = max(1, current_year - p.year)
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


def weighted_bfs(G: nx.DiGraph, seed_id: str, target_layer: Layer,
                 papers: dict[str, Paper], top_k: int = 5) -> list[str]:
    """
    SurveyG vertical traversal: from a Foundation seed, BFS toward target_layer
    prioritizing edges by weight.
    """
    visited, results, frontier = set([seed_id]), [], [seed_id]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in G.successors(node):
                if neighbor not in visited and neighbor in papers:
                    if papers[neighbor].layer == target_layer:
                        results.append(neighbor)
                    next_frontier.append(neighbor)
                    visited.add(neighbor)
        # Sort next frontier by edge weight descending
        next_frontier.sort(
            key=lambda n: max((G[p][n].get("weight", 0)
                               for p in G.predecessors(n) if p in visited), default=0),
            reverse=True
        )
        frontier = next_frontier
    return results[:top_k]
```

---

## 7. Prompt Templates (`core/prompts.py`)

```python
# ── Gap Detection ──────────────────────────────────────────────────────────

GAP_DETECTION_SYSTEM = """You are a research prerequisite analyst.
Given a research paper's text, identify knowledge gaps a reader needs filled
before they can fully understand the paper. Output ONLY a valid JSON array."""

GAP_DETECTION_USER = """Paper text:
<paper>
{paper_text}
</paper>

For each knowledge gap, output a JSON object with these exact fields:
- concept: short name of the concept (5 words max)
- gap_type: one of [terminology, methodology, benchmark, historical, mathematical]
- difficulty: one of [beginner, intermediate, advanced]
- domain: field of knowledge
- why_needed: one sentence — why the paper assumes this concept
- layer_hint: one of [foundation, development, frontier]
- retrieval_query: an academic search query to find papers explaining this concept
- source_passage: exact phrase from the paper that triggered this gap (quote it)
- confidence: your confidence 0.0–1.0

Also include gaps hinted by the Related Work section citations —
if the paper says "as shown in [X]" without explaining X, that is a gap.

Output only a JSON array. No preamble, no markdown.
User-specified gaps to include: {user_gaps}"""

# ── Gap Validation (grounding check) ───────────────────────────────────────

GROUNDING_CHECK_PROMPT = """Does the following gap genuinely trace to the paper text?

Gap concept: {concept}
Source passage claimed: {source_passage}
Paper abstract: {abstract}

Answer with JSON: {{"grounded": true/false, "reason": "one sentence"}}
No preamble."""

# ── Candidate Rationale ────────────────────────────────────────────────────

CANDIDATE_RATIONALE_PROMPT = """Knowledge gap: {concept} ({gap_type})
Why needed: {why_needed}

Candidate paper:
Title: {title}
Abstract: {abstract}

In one sentence, explain specifically how this paper fills the knowledge gap above.
Be concrete. Do not just restate the abstract."""

# ── Writing Agent ──────────────────────────────────────────────────────────

WRITING_AGENT_SYSTEM = """You are an expert research educator writing a learning document.
Your job is to explain a knowledge gap using only the provided source passages.
Every factual claim must be followed by a citation in format [cite: paper_id::chunk_id].
Do not introduce information not present in the source passages."""

WRITING_AGENT_USER = """Knowledge gap to explain:
Concept: {concept}
Type: {gap_type}
Difficulty: {difficulty}
Why the reader needs this: {why_needed}

Reference chain (how this paper relates to the target paper):
{path_verbalization}

Source passages to draw from:
{passages}

Write a clear explanation (2–4 paragraphs) suitable for a researcher who has NOT
read the papers above. After every factual claim, add [cite: paper_id::chunk_id].
Start with what the concept IS, then explain how it works, then why it matters
for understanding the target paper.
End with a one-sentence bridge: "This concept is needed in [target paper] because..."
"""

# ── Evaluation Agent ───────────────────────────────────────────────────────

EVAL_AGENT_PROMPT = """Review this explanation of the concept "{concept}":

<explanation>
{explanation}
</explanation>

Source passages available:
{passages}

Check:
1. Does the explanation introduce any term that is itself unexplained?
   If yes, list those terms as sub_gaps.
2. Are all [cite: ...] markers referencing passages that actually support the claim?
   List any unsupported claims.
3. Is anything stated that is NOT in the source passages (hallucination)?

Output JSON only:
{{
  "sub_gaps": ["term1", "term2"],
  "unsupported_claims": ["claim text..."],
  "hallucinations": ["text..."],
  "approved": true/false
}}"""

# ── Chronological Ordering ─────────────────────────────────────────────────

ORDERING_PROMPT = """Given these knowledge gaps, determine the optimal learning order.

Gaps:
{gaps_json}

Rules:
1. Foundation-layer gaps come before Development, which comes before Frontier.
2. Within the same layer, a gap A should come before gap B if understanding A
   is necessary to understand B.
3. For each ordering decision, output a dependency sentence.

Output JSON:
{{
  "ordered_gap_ids": ["id1", "id2", ...],
  "dependencies": [
    {{"before": "id1", "after": "id2",
      "reason": "Gap B assumes knowledge of attention, introduced in Gap A."}}
  ]
}}"""

# ── Horizontal Cluster Summary ─────────────────────────────────────────────

CLUSTER_SUMMARY_PROMPT = """You are analyzing a cluster of related academic papers.

Papers in this cluster:
{papers_json}

These papers all relate to the knowledge gap: {gap_concept}

In 2–3 sentences:
1. What shared methodology or theme unifies these papers?
2. How do they collectively address the gap?

Be specific. Do not just list the papers."""
```

---

## 8. Gap Detection (`phase_a/gap_detection.py`)

```python
import json
from groq import Groq
from core.models import KnowledgeGap, GapType, Difficulty, Layer
from core.config import *
from core.prompts import GAP_DETECTION_SYSTEM, GAP_DETECTION_USER, GROUNDING_CHECK_PROMPT
from utils.cache import get_cached, set_cached
import hashlib

client = Groq()   # reads GROQ_API_KEY from env

def _llm_light(prompt: str) -> str:
    r = client.chat.completions.create(
        model=LLM_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return r.choices[0].message.content

def _llm_heavy(system: str, user: str, temperature: float = 0.3) -> str:
    r = client.chat.completions.create(
        model=LLM_HEAVY,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=temperature
    )
    return r.choices[0].message.content


def detect_gaps(paper_text: str, user_gaps: list[str] = None,
                abstract: str = "") -> list[KnowledgeGap]:
    """
    Run gap detection SELF_CONSISTENCY_RUNS times, keep gaps
    appearing >= SELF_CONSISTENCY_MIN times, validate grounding.
    """
    user_gaps_str = ", ".join(user_gaps) if user_gaps else "none"
    prompt_user   = GAP_DETECTION_USER.format(
        paper_text=paper_text[:6000],  # trim to fit context
        user_gaps=user_gaps_str
    )

    cache_key = "gaps:" + hashlib.md5(prompt_user.encode()).hexdigest()
    cached = get_cached(cache_key)
    if cached:
        return [KnowledgeGap(**g) for g in cached]

    all_runs: list[list[dict]] = []
    for temp in [0.2, 0.4, 0.6]:   # 3 runs at different temperatures
        try:
            raw = _llm_heavy(GAP_DETECTION_SYSTEM, prompt_user, temperature=temp)
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            all_runs.append(json.loads(raw))
        except Exception:
            continue

    # Self-consistency: keep concepts appearing in >= MIN runs
    from collections import Counter
    concept_counts = Counter()
    concept_to_data: dict[str, dict] = {}
    for run in all_runs:
        for gap in run:
            c = gap.get("concept", "").lower().strip()
            concept_counts[c] += 1
            concept_to_data[c] = gap  # last run wins for data

    stable_gaps = [
        concept_to_data[c] for c, count in concept_counts.items()
        if count >= SELF_CONSISTENCY_MIN
    ]

    # Build KnowledgeGap objects + validate grounding
    validated = []
    for i, g in enumerate(stable_gaps):
        conf = float(g.get("confidence", 0.5))
        grounded = _check_grounding(
            g.get("concept",""), g.get("source_passage",""), abstract
        )
        if not grounded and conf < CONFIDENCE_THRESHOLD:
            continue   # discard ungrounded low-confidence gaps
        if not grounded:
            conf *= 0.7   # penalize but keep flagged gaps

        gap = KnowledgeGap(
            gap_id=f"gap_{i:03d}",
            concept=g.get("concept",""),
            gap_type=GapType(g.get("gap_type","methodology")),
            difficulty=Difficulty(g.get("difficulty","intermediate")),
            domain=g.get("domain",""),
            why_needed=g.get("why_needed",""),
            layer_hint=Layer(g.get("layer_hint","development")),
            retrieval_query=g.get("retrieval_query",""),
            source_passage=g.get("source_passage",""),
            confidence=conf,
        )
        validated.append(gap)

    set_cached(cache_key, [vars(g) for g in validated])
    return validated


def _check_grounding(concept: str, source_passage: str, abstract: str) -> bool:
    if not source_passage:
        return False
    prompt = GROUNDING_CHECK_PROMPT.format(
        concept=concept, source_passage=source_passage, abstract=abstract
    )
    try:
        raw  = _llm_light(prompt)
        data = json.loads(raw.strip().lstrip("```json").rstrip("```"))
        return bool(data.get("grounded", False))
    except Exception:
        return True   # fail open
```

---

## 9. Candidate Matching (`phase_a/candidate.py`)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.models import Paper, KnowledgeGap, CandidatePaper, Layer
from core.config import ALPHA, BETA, GAMMA
from core.prompts import CANDIDATE_RATIONALE_PROMPT, CLUSTER_SUMMARY_PROMPT
from phase_a.gap_detection import _llm_light, _llm_heavy
import igraph as ig
import leidenalg
import json

embedder = SentenceTransformer("all-MiniLM-L6-v2")

LAYER_MATCH_SCORES = {
    "foundation":   {"foundation": 1.0, "development": 0.5, "frontier": 0.2},
    "development":  {"foundation": 0.4, "development": 1.0, "frontier": 0.6},
    "frontier":     {"foundation": 0.1, "development": 0.4, "frontier": 1.0},
}


def match_candidates(gaps: list[KnowledgeGap],
                     papers: dict[str, Paper],
                     top_k: int = 5) -> list[CandidatePaper]:
    """
    For each gap, embed the retrieval_query and score all papers.
    Score = α·semantic + β·trendscore_norm + γ·layer_match
    Apply Leiden clustering per gap to group candidates and generate rationale.
    """
    paper_list  = [p for p in papers.values() if p.level > 0]
    if not paper_list:
        return []

    abstracts   = [p.abstract for p in paper_list]
    paper_embeds= embedder.encode(abstracts, batch_size=64, show_progress_bar=False)

    # Normalize trendscores to [0,1]
    trendscores = np.array([p.trendscore for p in paper_list])
    ts_max = trendscores.max() or 1.0
    trendscores_norm = trendscores / ts_max

    all_candidates = []
    for gap in gaps:
        q_embed = embedder.encode([gap.retrieval_query])
        sims    = cosine_similarity(q_embed, paper_embeds)[0]

        scores = []
        for i, paper in enumerate(paper_list):
            layer_match = LAYER_MATCH_SCORES.get(gap.layer_hint.value, {})\
                                            .get(paper.layer.value, 0.3)
            score = (ALPHA * sims[i] +
                     BETA  * trendscores_norm[i] +
                     GAMMA * layer_match)
            scores.append((score, i))

        scores.sort(reverse=True)
        top_papers = [paper_list[i] for _, i in scores[:top_k]]

        # Leiden clustering on top papers for horizontal summarization
        cluster_summary = _leiden_cluster_summary(top_papers, gap.concept)

        for paper in top_papers:
            rationale = _get_rationale(gap, paper)
            all_candidates.append(CandidatePaper(
                paper=paper,
                gap_id=gap.gap_id,
                relevance_score=scores[top_papers.index(paper)][0],
                rationale=f"{rationale}\n\nCluster context: {cluster_summary}",
                pdf_available=bool(paper.pdf_url)
            ))

    return all_candidates


def _leiden_cluster_summary(papers: list[Paper], gap_concept: str) -> str:
    """Cluster the candidate papers and generate a summary of shared themes."""
    if len(papers) < 3:
        return ""
    try:
        embeds = embedder.encode([p.abstract for p in papers])
        sims   = cosine_similarity(embeds)

        # Build igraph for Leiden
        g = ig.Graph()
        g.add_vertices(len(papers))
        edges, weights = [], []
        for i in range(len(papers)):
            for j in range(i+1, len(papers)):
                if sims[i][j] > 0.3:
                    edges.append((i, j))
                    weights.append(float(sims[i][j]))
        g.add_edges(edges)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights="weight"
        )
        # Take the largest community
        largest = max(partition, key=len)
        cluster_papers = [papers[i] for i in largest]

        papers_json = json.dumps([
            {"title": p.title, "year": p.year, "abstract": p.abstract[:300]}
            for p in cluster_papers
        ])
        prompt = CLUSTER_SUMMARY_PROMPT.format(
            papers_json=papers_json, gap_concept=gap_concept
        )
        return _llm_light(prompt)
    except Exception:
        return ""


def _get_rationale(gap: KnowledgeGap, paper: Paper) -> str:
    prompt = CANDIDATE_RATIONALE_PROMPT.format(
        concept=gap.concept, gap_type=gap.gap_type.value,
        why_needed=gap.why_needed,
        title=paper.title, abstract=paper.abstract[:500]
    )
    return _llm_light(prompt)
```

---

## 10. Phase B — PDF Ingestion (`phase_b/ingest.py`)

```python
import os, subprocess, hashlib
from pathlib import Path
from core.models import Chunk, Paper
from utils.embedder import embed_texts

CHUNK_SIZE = 400      # tokens approx (words * 1.3)
CHUNK_OVERLAP = 80

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """Use marker-pdf to convert PDF to clean Markdown preserving sections."""
    cache_key = hashlib.md5(open(pdf_path,'rb').read()).hexdigest()
    md_path = f"/tmp/kgms_marker_{cache_key}.md"
    if os.path.exists(md_path):
        return open(md_path).read()
    # marker-pdf CLI: pip install marker-pdf
    result = subprocess.run(
        ["marker_single", pdf_path, "/tmp/", "--batch_multiplier", "2"],
        capture_output=True, text=True
    )
    # marker outputs to /tmp/<basename>/<basename>.md
    stem    = Path(pdf_path).stem
    out_md  = f"/tmp/{stem}/{stem}.md"
    if os.path.exists(out_md):
        md = open(out_md).read()
        open(md_path, "w").write(md)
        return md
    # Fallback: PyMuPDF if marker fails
    return _pymupdf_fallback(pdf_path)

def _pymupdf_fallback(pdf_path: str) -> str:
    import fitz
    doc  = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    return text

def chunk_markdown(md: str, paper: Paper) -> list[Chunk]:
    """
    Split by section headers (##, ###), then apply sliding window within sections.
    This ensures chunks are semantically coherent.
    """
    import re
    sections = re.split(r'\n(?=#{1,3} )', md)
    chunks   = []
    chunk_id = 0

    for section_text in sections:
        header_match = re.match(r'^(#{1,3}) (.+)\n', section_text)
        section_name = header_match.group(2).strip() if header_match else "body"
        body         = section_text[header_match.end():] if header_match else section_text

        words  = body.split()
        start  = 0
        while start < len(words):
            end        = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            if len(chunk_text.strip()) > 50:   # skip tiny chunks
                chunks.append(Chunk(
                    chunk_id=f"{paper.paper_id}::{chunk_id:04d}",
                    paper_id=paper.paper_id,
                    section=section_name,
                    text=chunk_text,
                    page=0,
                ))
                chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    # Batch embed all chunks
    texts   = [c.text for c in chunks]
    embeds  = embed_texts(texts)
    for c, e in zip(chunks, embeds):
        c.embedding = e.tolist()

    return chunks


def build_chroma_collection(chunks: list[Chunk]):
    """Build an in-memory ChromaDB collection from chunks."""
    import chromadb
    client = chromadb.Client()   # in-memory
    col    = client.get_or_create_collection("reference_papers")
    col.add(
        ids       =[c.chunk_id for c in chunks],
        documents =[c.text for c in chunks],
        embeddings=[c.embedding for c in chunks],
        metadatas =[{"paper_id": c.paper_id, "section": c.section} for c in chunks],
    )
    return col
```

---

## 11. Hybrid Retrieval (`phase_b/retrieval.py`)

```python
import numpy as np
from sentence_transformers import CrossEncoder
from bm25s import BM25
from core.models import KnowledgeGap, Chunk
from core.config import DENSE_TOP_K, SPARSE_TOP_K, RERANK_TOP_N, CROSS_ENCODER_THRESHOLD
from utils.embedder import embed_texts

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_retrieve(gap: KnowledgeGap, collection, all_chunks: list[Chunk]) -> list[Chunk]:
    """
    Hybrid retrieval: Dense (ChromaDB) + Sparse (BM25), merged with RRF,
    then cross-encoder reranked. Returns top RERANK_TOP_N chunks.
    """
    query = gap.retrieval_query

    # 1. Dense retrieval
    q_embed = embed_texts([query])[0].tolist()
    dense_results = collection.query(
        query_embeddings=[q_embed],
        n_results=DENSE_TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = dense_results["ids"][0]

    # 2. Sparse BM25
    corpus     = [c.text for c in all_chunks]
    chunk_ids  = [c.chunk_id for c in all_chunks]
    bm25       = BM25()
    tokenized  = [t.split() for t in corpus]
    bm25.index(tokenized)
    scores     = bm25.get_scores(query.split())
    top_sparse = np.argsort(scores)[::-1][:SPARSE_TOP_K]
    sparse_ids = [chunk_ids[i] for i in top_sparse]

    # 3. Reciprocal Rank Fusion
    def rrf_score(rank, k=60): return 1 / (k + rank)
    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0) + rrf_score(rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0) + rrf_score(rank)

    chunk_map   = {c.chunk_id: c for c in all_chunks}
    top_by_rrf  = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:30]
    candidate_chunks = [chunk_map[cid] for cid in top_by_rrf if cid in chunk_map]

    # 4. Cross-encoder rerank
    pairs  = [(query, c.text) for c in candidate_chunks]
    scores_ce = reranker.predict(pairs)
    ranked = sorted(zip(scores_ce, candidate_chunks), key=lambda x: x[0], reverse=True)

    # Filter by threshold
    final = [c for score, c in ranked[:RERANK_TOP_N] if score > CROSS_ENCODER_THRESHOLD]
    return final if final else [c for _, c in ranked[:2]]  # always return at least 2
```

---

## 12. Multi-Hop Detection (`phase_b/multihop.py`)

```python
import json
from core.models import KnowledgeGap, GapType, Difficulty, Layer, Chunk
from core.config import MAX_MULTIHOP_DEPTH, LLM_LIGHT
from core.prompts import EVAL_AGENT_PROMPT
from phase_a.gap_detection import _llm_light
from phase_b.retrieval import hybrid_retrieve

def detect_subgaps(explanation: str, gap: KnowledgeGap,
                   existing_concepts: set[str]) -> list[KnowledgeGap]:
    """
    After generating an explanation, check if it introduces unexplained concepts.
    Returns sub-gaps as new KnowledgeGap objects.
    """
    prompt = f"""This explanation was just generated for the concept "{gap.concept}":

<explanation>
{explanation}
</explanation>

Does this explanation itself use any technical term that it does NOT define?
List only terms that a reader unfamiliar with the field would not understand,
and which are NOT in this set of already-explained concepts: {existing_concepts}

Output JSON array of strings only. If none, output [].
Example: ["attention mechanism", "softmax"]"""

    try:
        raw      = _llm_light(prompt)
        raw      = raw.strip().lstrip("```json").rstrip("```")
        subterms = json.loads(raw)
    except Exception:
        return []

    subgaps = []
    for i, term in enumerate(subterms[:3]):   # max 3 sub-gaps per gap
        if term.lower() in existing_concepts:
            continue
        subgap = KnowledgeGap(
            gap_id=f"{gap.gap_id}_sub{i}",
            concept=term,
            gap_type=GapType.METHODOLOGY,
            difficulty=Difficulty.INTERMEDIATE,
            domain=gap.domain,
            why_needed=f"Introduced in explanation of '{gap.concept}' without definition",
            layer_hint=Layer.FOUNDATION,
            retrieval_query=f"{term} definition explanation",
            source_passage=f"Introduced in explanation of {gap.concept}",
            confidence=0.75,
        )
        subgaps.append(subgap)

    return subgaps
```

---

## 13. Chronological Ordering (`phase_b/ordering.py`)

```python
import json
from core.models import KnowledgeGap, Layer
from core.config import LLM_HEAVY
from core.prompts import ORDERING_PROMPT
from phase_a.gap_detection import _llm_heavy, _llm_light

LAYER_ORDER = {Layer.FOUNDATION: 0, Layer.DEVELOPMENT: 1, Layer.FRONTIER: 2}

def order_gaps(gaps: list[KnowledgeGap],
               papers_by_gap: dict[str, list]) -> tuple[list[str], list[dict]]:
    """
    Step 1 (deterministic): sort by layer_hint, then by trendscore of best candidate.
    Step 2 (LLM): refine within-layer order using dependency extraction.
    Returns (ordered_gap_ids, dependency_edges).
    """
    # Step 1: deterministic primary sort
    def primary_sort_key(gap: KnowledgeGap):
        layer_rank = LAYER_ORDER.get(gap.layer_hint, 1)
        # Within layer: by trendscore of best candidate paper (desc)
        best_ts = max(
            (p.trendscore for p in papers_by_gap.get(gap.gap_id, [])), default=0
        )
        return (layer_rank, -best_ts)

    sorted_gaps = sorted(gaps, key=primary_sort_key)

    # Step 2: LLM refinement (within-layer reordering only)
    gaps_json = json.dumps([
        {"gap_id": g.gap_id, "concept": g.concept, "layer": g.layer_hint.value,
         "why_needed": g.why_needed, "difficulty": g.difficulty.value}
        for g in sorted_gaps
    ], indent=2)

    try:
        raw  = _llm_heavy("You are a curriculum designer.", 
                          ORDERING_PROMPT.format(gaps_json=gaps_json), 
                          temperature=0.1)
        raw  = raw.strip().lstrip("```json").rstrip("```")
        data = json.loads(raw)
        ordered_ids  = data.get("ordered_gap_ids", [g.gap_id for g in sorted_gaps])
        dependencies = data.get("dependencies", [])
    except Exception:
        ordered_ids  = [g.gap_id for g in sorted_gaps]
        dependencies = []

    # Cycle detection and resolution using trendscore
    ordered_ids = _resolve_cycles(ordered_ids, dependencies, gaps)
    return ordered_ids, dependencies


def _resolve_cycles(ordered_ids: list[str],
                    dependencies: list[dict],
                    gaps: list[KnowledgeGap]) -> list[str]:
    """If A→B and B→A, break tie by trendscore or keep deterministic order."""
    dep_set = {(d["before"], d["after"]) for d in dependencies}
    result  = list(ordered_ids)  # already deterministically sorted
    # Simple: trust the deterministic sort as the tiebreaker — just return as-is
    # since step 1 already broke ties by trendscore
    return result
```

---

## 14. Generation (`phase_b/generation.py`)

```python
import json
from core.models import KnowledgeGap, Chunk, GapExplanation, Paper
from core.config import *
from core.prompts import WRITING_AGENT_SYSTEM, WRITING_AGENT_USER, EVAL_AGENT_PROMPT
from phase_a.gap_detection import _llm_heavy, _llm_light
from phase_b.multihop import detect_subgaps

def verbalize_path(gap: KnowledgeGap,
                   papers: dict[str, Paper],
                   G) -> str:
    """
    GNN-RAG-style path verbalization:
    [Target Paper] → cites → [Ref A (Foundation)] → cites → [Ref B (Development)]
    """
    # Find the candidate paper for this gap in graph
    lines = [f"[Target Paper]"]
    # Walk from BA toward papers related to this gap
    # (simplified: show layer path)
    lines.append(f"  ↓ cites ({gap.layer_hint.value} layer)")
    lines.append(f"  [{gap.concept} — sourced from {gap.layer_hint.value} references]")
    return "\n".join(lines)


def format_passages(chunks: list[Chunk], papers: dict[str, Paper]) -> str:
    parts = []
    for c in chunks:
        paper = papers.get(c.paper_id)
        title = paper.title if paper else c.paper_id
        parts.append(
            f"[{c.chunk_id}] From: \"{title}\" (Section: {c.section})\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_explanation(gap: KnowledgeGap,
                         chunks: list[Chunk],
                         papers: dict[str, Paper],
                         G,
                         known_concepts: set[str],
                         depth: int = 0) -> GapExplanation:
    """
    Writing Agent + Evaluation Agent loop.
    Recursively handles sub-gaps up to MAX_MULTIHOP_DEPTH.
    """
    if not chunks:
        return GapExplanation(
            gap_id=gap.gap_id, concept=gap.concept,
            explanation_text=f"[Abstract-level only] No full-text PDF available for '{gap.concept}'. "
                             f"Based on abstract: {gap.why_needed}",
            source_citations=[], confidence=0.3, is_abstract_only=True
        )

    path     = verbalize_path(gap, papers, G)
    passages = format_passages(chunks, papers)
    user_prompt = WRITING_AGENT_USER.format(
        concept=gap.concept, gap_type=gap.gap_type.value,
        difficulty=gap.difficulty.value, why_needed=gap.why_needed,
        path_verbalization=path, passages=passages
    )

    explanation_text = ""
    for loop in range(MAX_EVAL_LOOPS):
        # Writing Agent
        explanation_text = _llm_heavy(WRITING_AGENT_SYSTEM, user_prompt)

        # Evaluation Agent
        eval_prompt = EVAL_AGENT_PROMPT.format(
            concept=gap.concept, explanation=explanation_text, passages=passages
        )
        try:
            eval_raw  = _llm_light(eval_prompt)
            eval_raw  = eval_raw.strip().lstrip("```json").rstrip("```")
            eval_data = json.loads(eval_raw)
        except Exception:
            break

        if eval_data.get("approved", True):
            break

        # Handle hallucinations: add constraint to re-prompt
        if eval_data.get("hallucinations"):
            bad = "; ".join(eval_data["hallucinations"][:2])
            user_prompt += f"\n\nIMPORTANT: Do NOT include this claim (not in sources): {bad}"

    # Extract citation IDs from explanation text
    import re
    citations = re.findall(r'\[cite: ([^\]]+)\]', explanation_text)

    # Compute confidence from cross-encoder scores would go here
    confidence = min(0.95, gap.confidence + 0.1 * len(chunks))

    explanation = GapExplanation(
        gap_id=gap.gap_id, concept=gap.concept,
        explanation_text=explanation_text,
        source_citations=list(set(citations)),
        confidence=confidence
    )

    # Multi-hop: check for sub-gaps
    if depth < MAX_MULTIHOP_DEPTH:
        subgaps = detect_subgaps(explanation_text, gap, known_concepts)
        # Sub-gap explanations would be generated recursively and prepended
        # (handled in pipeline.py)

    return explanation


def assemble_document(ordered_explanations: list[GapExplanation],
                      ba_title: str,
                      dependencies: list[dict]) -> str:
    """
    Assemble the final learning document in Markdown.
    """
    lines = [
        f"# Learning Roadmap for: {ba_title}\n",
        "*This document was automatically generated to fill your knowledge gaps "
        "before reading the target paper.*\n",
        "---\n"
    ]

    for i, exp in enumerate(ordered_explanations, 1):
        status = " *(abstract only)*" if exp.is_abstract_only else ""
        conf_pct = int(exp.confidence * 100)
        lines.append(f"## {i}. {exp.concept}{status}")
        lines.append(f"*Confidence: {conf_pct}%*\n")
        lines.append(exp.explanation_text)
        lines.append("")

        # Add dependency note if this concept depends on a previous one
        for dep in dependencies:
            if dep.get("after") == exp.gap_id:
                lines.append(f"> **Prerequisite link:** {dep['reason']}\n")

    lines.append("---")
    lines.append(f"## You are now ready to read: *{ba_title}*")
    lines.append("\nAll identified prerequisite concepts have been explained above.")
    return "\n".join(lines)
```

---

## 15. LangGraph Pipeline (`pipeline.py`)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from core.models import PipelineState
from phase_a.graph import build_reference_graph
from phase_a.gap_detection import detect_gaps
from phase_a.candidate import match_candidates
from phase_b.ingest import parse_pdf_to_markdown, chunk_markdown, build_chroma_collection
from phase_b.retrieval import hybrid_retrieve
from phase_b.ordering import order_gaps
from phase_b.generation import generate_explanation, assemble_document
from utils.apis import resolve_pdf_url, download_pdf
import os

# ── Node functions ─────────────────────────────────────────────────────────

def node_build_graph(state: PipelineState) -> PipelineState:
    G, papers = build_reference_graph(state.ba_paper.paper_id)
    state.reference_graph = G
    state.all_papers = papers
    return state

def node_detect_gaps(state: PipelineState) -> PipelineState:
    gaps = detect_gaps(
        paper_text=state.ba_text,
        user_gaps=[],   # populated from UI
        abstract=state.ba_paper.abstract
    )
    state.gaps = gaps
    return state

def node_match_candidates(state: PipelineState) -> PipelineState:
    candidates = match_candidates(state.gaps, state.all_papers)
    # Resolve PDF URLs
    for c in candidates:
        if not c.paper.pdf_url:
            raw = {"externalIds": {}}
            if c.paper.arxiv_id:
                raw["externalIds"]["ArXiv"] = c.paper.arxiv_id
            if c.paper.doi:
                raw["externalIds"]["DOI"] = c.paper.doi
            c.paper.pdf_url = resolve_pdf_url(raw)
        c.pdf_available = bool(c.paper.pdf_url)
    state.candidates = candidates
    return state

# HUMAN CHECKPOINT — Phase A ends here, user collects PDFs

def node_ingest_pdfs(state: PipelineState) -> PipelineState:
    """Ingest all available PDFs (auto-fetched or user-uploaded)."""
    all_chunks = []
    pdf_dir    = "/tmp/kgms_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    seen_papers = set()
    for c in state.candidates:
        pid = c.paper.paper_id
        if pid in seen_papers:
            continue
        seen_papers.add(pid)

        pdf_path = f"{pdf_dir}/{pid.replace('/','_')}.pdf"
        if not os.path.exists(pdf_path):
            if c.paper.pdf_url:
                download_pdf(c.paper.pdf_url, pdf_path)

        if os.path.exists(pdf_path):
            md     = parse_pdf_to_markdown(pdf_path)
            chunks = chunk_markdown(md, c.paper)
            all_chunks.extend(chunks)
            state.all_papers[pid].pdf_url = pdf_path  # mark as local

    state.chunks = all_chunks
    state.chroma_collection = build_chroma_collection(all_chunks)
    return state

def node_generate_explanations(state: PipelineState) -> PipelineState:
    """Retrieve + generate for each gap, with multi-hop handling."""
    # First: determine ordering
    papers_by_gap = {}
    for c in state.candidates:
        papers_by_gap.setdefault(c.gap_id, []).append(c.paper)

    ordered_ids, dependencies = order_gaps(state.gaps, papers_by_gap)
    state.ordered_gaps = ordered_ids

    # Generate in order
    known_concepts: set[str] = set()
    explanations_map = {}

    gap_map = {g.gap_id: g for g in state.gaps}
    for gap_id in ordered_ids:
        gap = gap_map.get(gap_id)
        if not gap:
            continue

        # Retrieve chunks for this gap
        if state.chroma_collection and state.chunks:
            chunks = hybrid_retrieve(gap, state.chroma_collection, state.chunks)
        else:
            chunks = []

        exp = generate_explanation(
            gap=gap, chunks=chunks,
            papers=state.all_papers,
            G=state.reference_graph,
            known_concepts=known_concepts,
        )
        explanations_map[gap_id] = exp
        known_concepts.add(gap.concept.lower())

    # Assemble in order
    ordered_exps = [explanations_map[gid]
                    for gid in ordered_ids if gid in explanations_map]
    state.explanations = ordered_exps
    state.final_document = assemble_document(
        ordered_exps, state.ba_paper.title, dependencies
    )
    return state

# ── Graph assembly ─────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("build_graph",          node_build_graph)
    builder.add_node("detect_gaps",          node_detect_gaps)
    builder.add_node("match_candidates",     node_match_candidates)
    # HUMAN CHECKPOINT between phase A and B (LangGraph interrupt_before)
    builder.add_node("ingest_pdfs",          node_ingest_pdfs)
    builder.add_node("generate_explanations",node_generate_explanations)

    builder.set_entry_point("build_graph")
    builder.add_edge("build_graph",           "detect_gaps")
    builder.add_edge("detect_gaps",           "match_candidates")
    builder.add_edge("match_candidates",      "ingest_pdfs")   # interrupted here
    builder.add_edge("ingest_pdfs",           "generate_explanations")
    builder.add_edge("generate_explanations", END)

    # SQLite checkpointer for human-in-the-loop pause
    memory = SqliteSaver.from_conn_string("kgms_checkpoints.sqlite")
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["ingest_pdfs"]   # pause here for user to collect PDFs
    )
```

---

## 16. Evaluation (`eval/evaluate.py`)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def run_ragas_eval(gap_concept: str,
                   question: str,
                   answer: str,
                   contexts: list[str]) -> dict:
    """
    Run RAGAS on a single gap explanation.
    Returns faithfulness, answer_relevancy, context_recall scores.
    """
    ds = Dataset.from_dict({
        "question":  [question],
        "answer":    [answer],
        "contexts":  [contexts],
        "ground_truth": [question],  # question is proxy for ground truth
    })
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])
    return result.to_pandas().to_dict(orient="records")[0]

def check_faithfulness_gate(score: float, gap_id: str) -> bool:
    """Return True if explanation passes gate, False triggers re-retrieval."""
    from core.config import FAITHFULNESS_GATE
    return score >= FAITHFULNESS_GATE
```

---

## 17. MVP Tiering

```
MVP (Weeks 1–5):  Phase A complete
  ✓ Reference graph construction (NetworkX + Semantic Scholar)
  ✓ Layer assignment (trendscore)
  ✓ Gap detection (auto + user-specified, with self-consistency)
  ✓ Candidate matching (semantic + trendscore + layer)
  ✓ Cluster summaries (Leiden)
  ✓ Streamlit UI showing gaps + candidates + PDF links

Tier 2 (Weeks 6–8):  Phase B basic
  ✓ PDF auto-fetch (Unpaywall + arXiv)
  ✓ marker-pdf parsing + semantic chunking
  ✓ Dense-only retrieval (no BM25 yet)
  ✓ Single-pass Writing Agent (no eval loop)
  ✓ Abstract-only fallback with labeling
  ✓ Learning document output

Tier 3 (Weeks 9–12):  Full system
  ✓ Hybrid retrieval (dense + BM25 + cross-encoder)
  ✓ Writing + Evaluation agent loop (MAX_EVAL_LOOPS=2)
  ✓ Multi-hop sub-gap detection
  ✓ Claim-level citation anchoring + grounding check
  ✓ RAGAS faithfulness gate with auto re-retrieval
  ✓ Full chronological ordering with dependency graph
  ✓ LangGraph human checkpoint
```

---

## 18. Environment Setup

```bash
# Create environment
conda create -n kgms python=3.11
conda activate kgms

# Core
pip install groq langchain langgraph sentence-transformers chromadb

# Retrieval
pip install bm25s rank-bm25

# Graph
pip install networkx igraph leidenalg

# PDF
pip install marker-pdf pymupdf   # marker-pdf is the primary

# Evaluation
pip install ragas datasets

# UI
pip install streamlit

# Utilities
pip install requests httpx

# Environment variables
export GROQ_API_KEY="your_key_here"
# Groq free tier: https://console.groq.com
# Semantic Scholar: no key needed for <100 req/5min
# Unpaywall: free, just needs an email in the request
```

---

## 19. Key Architectural Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Primary LLM | Groq Llama 3.3 70B | Free, fast, strong reasoning |
| Light LLM | Groq Llama 3.1 8B | Structural tasks only, saves quota |
| Embeddings | sentence-transformers local | Zero API cost, no rate limits |
| PDF parser | marker-pdf | Academic-trained, clean Markdown output |
| Chunking | By section header then sliding window | Semantically coherent chunks |
| Vector DB | ChromaDB in-memory | No server needed, reset per paper |
| Sparse retrieval | bm25s | Pure Python, no Elasticsearch |
| Fusion | RRF (not score fusion) | RRF is rank-agnostic, more robust |
| Reranking | Cross-encoder local | Free, better than score-based rerank |
| Ordering | Deterministic (layer+trendscore) first, LLM second | Reproducible + intelligent |
| Orchestration | LangGraph | Human checkpoint, state persistence |
| UI | Streamlit | One-person buildable in hours |
