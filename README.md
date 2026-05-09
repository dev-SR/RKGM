# Research Knowledge Gaps Mitigation (RKGM)

Automatically identifies what you need to know before reading a research paper, then generates a personalised, chronologically ordered learning document — sourced from the paper's own reference hierarchy.

**Based on:** Rahman et al. (2022), *An Article Recommendation Technique from a Multi-Layer Reference Article Graph for Facilitating Chronological Learning*, IEEE ICCIT. [DOI: 10.1109/ICCIT57492.2022.10103286](https://ieeexplore.ieee.org/abstract/document/10103286)

---

## What It Does

1. **Phase A** — Give it any paper (arXiv ID, DOI, Semantic Scholar ID or PDF file). It fetches the paper's 3-level reference graph, detects knowledge gaps using structured LLM prompting, and returns a ranked list of candidate papers per gap with rationale.

2. **Human checkpoint** — Review the gap list, toggle off concepts you already know, add your own. The system auto-fetches open-access PDFs via Unpaywall and arXiv.

3. **Phase B** — Ingests the PDFs, retrieves the most relevant passages per gap (hybrid dense + BM25 + cross-encoder reranking), generates a grounded explanation for each gap with inline citations, orders them chronologically, and assembles a complete learning document.

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

### 3. Running Tests

```bash
  # Minimal — paper ID only, abstract-level gap detection
  python run.py --paper 2405.20139 --depth 0 --out ./output/  # for paper: https://arxiv.org/pdf/2405.20139

  # # With PDF for richer gap detection
  # python run.py --paper 2405.20139 --pdf /path/to/gnn_rag.pdf

  # # Control depth and add custom gaps
  # python run.py --paper 2405.20139 --pdf paper.pdf --depth 2 \
  #               --gaps "knowledge graph" "SPARQL"

  # # Skip Phase B (Phase A only — gaps + candidate list)
  # python run.py --paper 2405.20139 --pdf paper.pdf --phase-a-only

  # # Custom output directory
  # python run.py --paper 2405.20139 --pdf paper.pdf --out ./results/

  # # Full verbose logging
  # python run.py --paper 2405.20139 --pdf paper.pdf --verbose

  # # Clear API cache (force fresh Semantic Scholar calls)
  # python run.py --paper 2405.20139 --clear-cache

# Outputs written to --out directory (default: current directory):
#   learning_roadmap_<paper_id>.md   — the learning document
#   phase_a_state_<paper_id>.json    — saved Phase A state (re-usable)
#   candidates_<paper_id>.bib        — BibTeX for all candidate papers
#   candidates_<paper_id>.csv        — CSV with PDF availability status
```

For better PDF parsing (recommended — handles academic two-column layouts):

```bash
pip install marker-pdf
# Note: downloads ~1.5 GB of models on first use
```

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
