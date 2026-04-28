# Research Knowledge Gaps Mitigation (RKGM)

Automatically identifies what you need to know before reading a research paper, then generates a personalised, chronologically ordered learning document — sourced from the paper's own reference hierarchy.

**Based on:** Rahman et al. (2022), *An Article Recommendation Technique from a Multi-Layer Reference Article Graph for Facilitating Chronological Learning*, IEEE ICCIT. [DOI: 10.1109/ICCIT57492.2022.10103286](https://ieeexplore.ieee.org/abstract/document/10103286)

---

## What It Does

1. **Phase A** — Give it any paper (arXiv ID, DOI, Semantic Scholar ID or PDF file). It fetches the paper's 3-level reference graph, detects knowledge gaps using structured LLM prompting, and returns a ranked list of candidate papers per gap with rationale.

2. **Human checkpoint** — Review the gap list, toggle off concepts you already know, add your own. The system auto-fetches open-access PDFs via Unpaywall and arXiv.

3. **Phase B** — Ingests the PDFs, retrieves the most relevant passages per gap (hybrid dense + BM25 + cross-encoder reranking), generates a grounded explanation for each gap with inline citations, orders them chronologically, and assembles a complete learning document.

---

## Project Structure

```
kgms/
├── core/
│   ├── models.py         All dataclasses (Paper, KnowledgeGap, GapExplanation, …)
│   ├── config.py         All constants, thresholds, model names — tune here
│   └── prompts.py        All LLM prompt templates — tune here
│
├── utils/
│   ├── ingest.py         marker-pdf → section chunking → ChromaDB indexing
│   ├── cache.py          SQLite cache — all API calls cached on first fetch
│   ├── embedder.py       sentence-transformers (local) + TF-IDF fallback
│   ├── llm.py            Groq client with model routing + rate-limit retry
│   └── apis.py           Semantic Scholar + Unpaywall + arXiv + PDF download
│
├── phase_a/
│   ├── graph.py          BFS reference graph + layer assignment (trendscore)
│   ├── gap_detection.py  3-run self-consistency + grounding check
│   └── candidates.py     α/β/γ scoring + Leiden clustering + rationale
│
├── phase_b/
│   ├── retrieval.py      Dense + BM25 + RRF + cross-encoder reranking
│   ├── ordering.py       Deterministic layer sort + LLM dependency refinement
│   └── generation.py     Writing Agent + Eval Agent + multi-hop + doc assembly
│
├── eval/
│   └── evaluate.py       RAGAS metrics, faithfulness gate, citation grounding
│
├── pipeline.py           Phase A + Phase B orchestration
├── app.py                Streamlit UI (4 views)
└── requirements.txt
```

---

## Architecture

### Model Routing

| Task                                                                         | Model                         | Why                                               |
| ---------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------- |
| Gap detection (×3), grounding checks, eval agent, rationale, cluster summary | Groq Llama 3.1 8B             | Structured extraction — 8B is reliable and fast   |
| Writing Agent, ordering, sub-gap detection                                   | Groq Llama 3.3 70B            | Multi-paragraph generation needs the larger model |
| Embeddings, reranking                                                        | Local (sentence-transformers) | Zero API cost, no rate limits                     |

### Retrieval Pipeline

```
Gap query
  → Dense (ChromaDB cosine, top-20)  ─┐
  → Sparse (BM25, top-20)            ─┤→ RRF merge → Cross-encoder rerank → Top-5
```

### Ordering Strategy

1. **Stage 1 (deterministic):** Foundation → Development → Frontier, then by trendscore within layer
2. **Stage 2 (LLM):** Refine within-layer order using dependency sentences
3. **Cycle resolution:** Mutual dependency → higher trendscore paper goes first

### Gap Detection

- Runs 3× at temperatures [0.1, 0.35, 0.6]
- Keeps concepts appearing in ≥ 2/3 runs (self-consistency)
- Validates each gap traces back to a specific passage in the BA

---

## Setup & Usage

### 1. Get a Groq API key

Free at [console.groq.com](https://console.groq.com). The free tier (14,400 tokens/min on Llama 3.3 70B) is sufficient for this project.

Add the key to the `.env` file:

```bash
GROQ_API_KEY="your_key_here"
```

### 2. Install dependencies

```bash
git clone https://github.com/sharukhn32/Research-Knowledge-Gaps.git
cd Research-Knowledge-Gaps
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
# pip install -r requirements.txt
```

For better PDF parsing (recommended — handles academic two-column layouts):

```bash
pip install marker-pdf
# Note: downloads ~1.5 GB of models on first use
```

### 3. Running Tests

```bash
# base paper: https://arxiv.org/pdf/2405.20139
python test_phase_a.py --paper 2405.20139 --pdf path/to/2405.20139v1.pdf --depth 1

# # Minimal
# python test_phase_a.py --paper 2405.20139

# # Recommended — supply the PDF
# python test_phase_a.py --paper 2405.20139 --pdf gnn_rag.pdf

# # With depth control and extra gaps
# python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --depth 2 \
#   --gaps "SPARQL" "KG embedding" --show-papers 15

# # Export everything
# python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --export ./results/
        
python test_phase_b.py --paper 2405.20139 --pdf /home/sharukh/Downloads/temp/2405.20139v1.pdf --depth 1

# more examples:
# Full pipeline (Phase A → Phase B → document)
# python test_phase_b.py --paper 2405.20139 --pdf gnn_rag.pdf

# # Phase A already ran — re-use cached state
# python test_phase_b.py --paper 2405.20139

# # Force Phase A to re-run
# python test_phase_b.py --paper 2405.20139 --pdf p.pdf --rerun-a

# # Load an explicit state file
# python test_phase_b.py --state ./output/phase_a_state_2405.20139.json

# # Only generate explanations for specific gaps
# python test_phase_b.py --paper 2405.20139 --only-gaps "KGQA" "GNN" "retrieval"

# Custom output directory
python test_phase_b.py --paper 2405.20139 --pdf p.pdf --out ./results/
```

All 12 tests run with mocked API calls — no API key or internet required.

---

## Configuration

All tunable parameters are in `core/config.py`:

```python
FOUNDATION_TOP_K     = 10     # how many papers qualify as Foundation layer
FRONTIER_YEAR_CUTOFF = 2022   # papers from this year onwards → Frontier

SELF_CONSISTENCY_MIN = 2      # gap must appear in ≥ N/3 runs
CONFIDENCE_THRESHOLD = 0.5    # discard ungrounded gaps below this

ALPHA = 0.5    # candidate scoring: semantic similarity weight
BETA  = 0.3    # candidate scoring: trendscore weight  
GAMMA = 0.2    # candidate scoring: layer-match weight

MAX_MULTIHOP_DEPTH  = 2       # max recursion depth for sub-gap detection
MAX_EVAL_LOOPS      = 2       # Writing↔Eval agent iterations per gap
FAITHFULNESS_GATE   = 0.70    # RAGAS faithfulness minimum before retry
```

---

## Supported Paper Input Formats

| Format              | Example                                    |
| ------------------- | ------------------------------------------ |
| arXiv ID            | `2405.20139` or `arXiv:2405.20139`         |
| DOI                 | `10.1109/ICCIT57492.2022.10103286`         |
| Semantic Scholar ID | `649def34f8be52c8b66281af98ae884c09aef38b` |

---

## Citation

If you use this project, please cite the original paper it extends:

```bibtex
@inproceedings{rahman2022article,
  title     = {An Article Recommendation Technique from a Multi-Layer Reference
               Article Graph for Facilitating Chronological Learning},
  author    = {Rahman, Sharukh and Emad, Kazi Hasnayeen and Azad, Saiful
               and Mahmud, Mufti and Kaiser, M. Shamim},
  booktitle = {2022 25th International Conference on Computer and Information
               Technology (ICCIT)},
  year      = {2022},
  publisher = {IEEE},
  doi       = {10.1109/ICCIT57492.2022.10103286}
}
```
