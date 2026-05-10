#!/usr/bin/env python3
"""
run.py — KGMS Full Pipeline CLI

Runs Phase A and Phase B end-to-end from the terminal.
No UI required. Outputs a Markdown learning document.

Usage:
  # Minimal — paper ID only, abstract-level gap detection
  python run.py --paper 2405.20139 --depth 0 --out ./output/ 

  # With PDF for richer gap detection
  python run.py --paper 2405.20139 --pdf /path/to/gnn_rag.pdf

  # Control depth and add custom gaps
  python run.py --paper 2405.20139 --pdf paper.pdf --depth 2 \
                --gaps "knowledge graph" "SPARQL"

  # Skip Phase B (Phase A only — gaps + candidate list)
  python run.py --paper 2405.20139 --pdf paper.pdf --phase-a-only

  # Custom output directory
  python run.py --paper 2405.20139 --pdf paper.pdf --out ./results/

  # Full verbose logging
  python run.py --paper 2405.20139 --pdf paper.pdf --verbose

  # Clear API cache (force fresh Semantic Scholar calls)
  python run.py --paper 2405.20139 --clear-cache

Outputs written to --out directory (default: current directory):
  learning_roadmap_<paper_id>.md   — the learning document
  phase_a_state_<paper_id>.json    — saved Phase A state (re-usable)
  candidates_<paper_id>.bib        — BibTeX for all candidate papers
  candidates_<paper_id>.csv        — CSV with PDF availability status
"""

import argparse
import json
import os
import sys
import time
import textwrap
import traceback

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm import RateLimitError


# ── Colour helpers ─────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    GRAY = "\033[90m"


def _c(t, col):
    return f"{col}{t}{C.RESET}"


def ok(t):
    print(f"  {C.GREEN}✓{C.RESET} {t}")


def warn(t):
    print(f"  {C.YELLOW}⚠{C.RESET} {t}")


def err(t):
    print(f"  {C.RED}✗{C.RESET} {t}")


def info(t):
    print(f"  {C.GRAY}{t}{C.RESET}")


def head(t):
    print(f"\n{C.BOLD}{C.CYAN}── {t} {'─' * (58 - len(t))}{C.RESET}")


def kv(k, v, w=26):
    print(f"  {C.DIM}{k:<{w}}{C.RESET}{v}")


# ── Progress callback ──────────────────────────────────────────────────────


def make_callback(verbose: bool):
    def cb(msg: str):
        if verbose:
            print(f"  {C.DIM}  {msg}{C.RESET}")
        else:
            truncated = msg[:70] + "…" if len(msg) > 70 else msg
            print(f"\r  {C.DIM}⏳ {truncated:<73}{C.RESET}", end="", flush=True)

    return cb


# ── Phase A state save/load ────────────────────────────────────────────────


def save_phase_a_state(state, path: str):
    """Save Phase A output to JSON for re-use in future runs."""
    data = {
        "ba_paper": {
            "paper_id": state.ba_paper.paper_id,
            "title": state.ba_paper.title,
            "abstract": state.ba_paper.abstract,
            "year": state.ba_paper.year,
            "citation_count": state.ba_paper.citation_count,
            "arxiv_id": state.ba_paper.arxiv_id,
            "doi": state.ba_paper.doi,
            "pdf_url": state.ba_paper.pdf_url,
            "layer": state.ba_paper.layer.value,
            "trendscore": state.ba_paper.trendscore,
        },
        "ba_text_words": len(state.ba_text.split()) if state.ba_text else 0,
        "gaps": [
            {
                "gap_id": g.gap_id,
                "concept": g.concept,
                "gap_type": g.gap_type.value,
                "difficulty": g.difficulty.value,
                "domain": g.domain,
                "why_needed": g.why_needed,
                "layer_hint": g.layer_hint.value,
                "retrieval_query": g.retrieval_query,
                "source_passage": g.source_passage,
                "confidence": round(g.confidence, 4),
            }
            for g in state.gaps
        ],
        "candidates": [
            {
                "gap_id": c.gap_id,
                "paper_id": c.paper.paper_id,
                "title": c.paper.title,
                "year": c.paper.year,
                "citation_count": c.paper.citation_count,
                "layer": c.paper.layer.value,
                "trendscore": round(c.paper.trendscore, 3),
                "arxiv_id": c.paper.arxiv_id,
                "doi": c.paper.doi,
                "pdf_url": c.paper.pdf_url,
                "relevance_score": round(c.relevance_score, 4),
                "rationale": c.rationale,
                "pdf_available": c.pdf_available,
            }
            for c in state.candidates
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main ───────────────────────────────────────────────────────────────────


def run(args):
    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        err("GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then:  export GROQ_API_KEY='gsk_...'")
        print("  Or add it to a .env file in this directory.")
        sys.exit(1)

    if args.pdf and not os.path.exists(args.pdf):
        err(f"PDF not found: {args.pdf}")
        sys.exit(1)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    safe_id = args.paper.replace("/", "_").replace(":", "_")[:24]
    state_path = os.path.join(out_dir, f"phase_a_state_{safe_id}.json")
    doc_path = os.path.join(out_dir, f"learning_roadmap_{safe_id}.md")
    bib_path = os.path.join(out_dir, f"candidates_{safe_id}.bib")
    csv_path = os.path.join(out_dir, f"candidates_{safe_id}.csv")

    # ── Banner ─────────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.BLUE}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  KGMS — Knowledge Gap Mitigation System{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'═' * 70}{C.RESET}")
    kv("Paper", args.paper)
    kv("PDF", args.pdf or "(none — will try auto-download)")
    kv(
        "Reference depth",
        {
            0: "0 (gap detection only)",
            1: "1 (direct refs)",
            2: "2 (refs of refs)",
            3: "3 (full graph)",
        }.get(args.depth, str(args.depth)),
    )
    kv("Custom gaps", ", ".join(args.gaps) if args.gaps else "(none)")
    kv("Output dir", out_dir)
    kv("Mode", "Phase A only" if args.phase_a_only else "Phase A + Phase B")

    if args.clear_cache:
        from utils.cache import clear_cache

        clear_cache()
        ok("Cache cleared")

    cb = make_callback(args.verbose)
    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A
    # ══════════════════════════════════════════════════════════════════════
    head("PHASE A — Reference Graph + Gap Detection + Candidate Matching")
    t_a = time.time()

    from pipeline import run_phase_a

    try:
        state = run_phase_a(
            paper_input=args.paper,
            user_gaps=args.gaps or [],
            progress_callback=cb,
            pdf_path=args.pdf,
            reference_depth=args.depth,
        )
    except RateLimitError as e:
        if not args.verbose:
            print()
        err(f"Groq daily token limit reached for model '{e.model}'.")
        if e.wait_time:
            warn(f"Please try again in {e.wait_time}.")
        info("Upgrade at: https://console.groq.com/settings/billing")
        sys.exit(1)
    except Exception as e:
        if not args.verbose:
            print()
        err(f"Phase A failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    if not args.verbose:
        print()

    if state.errors:
        for e in state.errors:
            err(e)
        sys.exit(1)

    elapsed_a = time.time() - t_a

    # ── Phase A results ────────────────────────────────────────────────────
    ba = state.ba_paper
    ok(f"Paper: {ba.title} ({ba.year})")
    kv(
        "  Text source",
        "full PDF" if len(state.ba_text.split()) > 500 else "abstract only",
        22,
    )
    kv("  Words", f"{len(state.ba_text.split()):,}", 22)
    kv("  Papers in graph", str(len(state.all_papers)), 22)
    kv("  Gaps detected", _c(str(len(state.gaps)), C.GREEN), 22)

    if state.gaps:
        from collections import Counter

        by_type = Counter(g.gap_type.value for g in state.gaps)
        kv("  Gap types", "  ".join(f"{t}:{n}" for t, n in sorted(by_type.items())), 22)

    unique_papers = {c.paper.paper_id: c for c in state.candidates}
    pdf_avail = sum(1 for c in unique_papers.values() if c.pdf_available)
    kv("  Candidate papers", str(len(unique_papers)), 22)
    kv("  PDF available", f"{pdf_avail}/{len(unique_papers)}", 22)
    kv("  Phase A time", f"{elapsed_a:.1f}s", 22)

    # ── Print gap list ─────────────────────────────────────────────────────
    print()
    print(
        f"  {'#':<4} {'Concept':<28} {'Type':<14} {'Diff':<14} {'Conf':>5}  {'Layer'}"
    )
    print(f"  {'─' * 4} {'─' * 28} {'─' * 14} {'─' * 14} {'─' * 5}  {'─' * 12}")
    type_icons = {
        "terminology": "📖",
        "methodology": "⚙️",
        "benchmark": "📊",
        "historical": "📜",
        "mathematical": "🔢",
    }
    diff_icons = {"beginner": "🌱", "intermediate": "🌿", "advanced": "🌳"}
    layer_cols = {"foundation": C.GREEN, "development": C.YELLOW, "frontier": C.CYAN}
    for i, gap in enumerate(state.gaps, 1):
        lc = layer_cols.get(gap.layer_hint.value, C.RESET)
        icon = type_icons.get(gap.gap_type.value, "")
        diff = diff_icons.get(gap.difficulty.value, "")
        print(
            f"  {i:<4} {gap.concept[:28]:<28} "
            f"{icon}{gap.gap_type.value[:12]:<14} "
            f"{diff}{gap.difficulty.value[:12]:<14} "
            f"{int(gap.confidence * 100):>4}%  "
            f"{_c(gap.layer_hint.value, lc)}"
        )

    # ── Save Phase A outputs ───────────────────────────────────────────────
    save_phase_a_state(state, state_path)
    ok(f"State saved → {state_path}")

    if state.candidates:
        from phase_a.candidates import candidates_to_bibtex, candidates_to_csv

        with open(bib_path, "w") as f:
            f.write(candidates_to_bibtex(state.candidates))
        with open(csv_path, "w") as f:
            f.write(candidates_to_csv(state.candidates))
        ok(f"BibTeX   → {bib_path}")
        ok(f"CSV      → {csv_path}")

    if args.phase_a_only:
        head("DONE (Phase A only)")
        kv("Total time", f"{time.time() - t_total:.1f}s")
        print()
        print(
            f"  To run Phase B:  python run.py --paper {args.paper}"
            + (f" --pdf {args.pdf}" if args.pdf else "")
        )
        return

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B
    # ══════════════════════════════════════════════════════════════════════
    head("PHASE B — PDF Ingestion + Retrieval + Generation")
    t_b = time.time()

    from pipeline import run_phase_b

    try:
        state = run_phase_b(
            state=state,
            pdf_paths={},
            auto_fetch=True,
            progress_callback=cb,
        )
    except RateLimitError as e:
        if not args.verbose:
            print()
        err(f"Groq daily token limit reached for model '{e.model}'.")
        if e.wait_time:
            warn(f"Please try again in {e.wait_time}.")
        info("Upgrade at: https://console.groq.com/settings/billing")
        sys.exit(1)
    except Exception as e:
        if not args.verbose:
            print()
        err(f"Phase B failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    if not args.verbose:
        print()

    elapsed_b = time.time() - t_b

    # Non-fatal warnings (coverage issues etc.)
    for e in state.errors:
        warn(e)

    if not state.final_document:
        err("Document generation produced no output.")
        sys.exit(1)

    # ── Write document ─────────────────────────────────────────────────────
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(state.final_document)

    # ── Phase B results ────────────────────────────────────────────────────
    n_exps = len(state.explanations)
    n_full = sum(1 for e in state.explanations if not e.is_abstract_only)
    n_abstract = sum(1 for e in state.explanations if e.is_abstract_only)
    avg_conf = sum(e.confidence for e in state.explanations) / max(n_exps, 1)

    ok(f"Learning document → {doc_path}")
    kv("  Explanations", str(n_exps), 22)
    kv("  Full-text", _c(str(n_full), C.GREEN), 22)
    kv("  Abstract-only", _c(str(n_abstract), C.YELLOW if n_abstract else C.GREEN), 22)
    kv("  Avg confidence", f"{int(avg_conf * 100)}%", 22)
    kv("  Phase B time", f"{elapsed_b:.1f}s", 22)

    # Coverage report
    if hasattr(state, "_coverage_report") and state._coverage_report:
        cov = state._coverage_report
        from eval.coverage import coverage_badge

        badge, label = coverage_badge(cov.coverage_score)
        kv("  Coverage", f"{badge} {label}", 22)
        for rec in cov.recommendations:
            if "excellent" not in rec.lower():
                warn(rec)

    # ── Learning order ─────────────────────────────────────────────────────
    if state.ordered_gap_ids and state.explanations:
        print()
        print(f"  {'#':<4} {'Concept':<32} {'Type':<6}  {'Conf':>5}  Source")
        print(f"  {'─' * 4} {'─' * 32} {'─' * 6}  {'─' * 5}  {'─' * 14}")
        gap_map = {g.gap_id: g for g in state.gaps}
        exp_map = {e.gap_id: e for e in state.explanations}
        for i, gid in enumerate(state.ordered_gap_ids, 1):
            gap = gap_map.get(gid)
            exp = exp_map.get(gid)
            if not gap or not exp:
                continue
            icon = type_icons.get(gap.gap_type.value, "")
            source = (
                "📄 abstract"
                if exp.is_abstract_only
                else f"✅ {len(exp.source_citations)} cite(s)"
            )
            conf_c = (
                C.GREEN
                if exp.confidence >= 0.75
                else (C.YELLOW if exp.confidence >= 0.5 else C.RED)
            )
            print(
                f"  {i:<4} {gap.concept[:32]:<32} {icon:<6}  "
                f"{_c(f'{int(exp.confidence * 100)}%', conf_c):>5}  {source}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    head("COMPLETE")
    kv(
        "Total time",
        f"{time.time() - t_total:.1f}s  (A: {elapsed_a:.0f}s + B: {elapsed_b:.0f}s)",
    )
    kv("Learning document", doc_path)
    kv("State (reusable)", state_path)
    kv("BibTeX", bib_path)
    kv("CSV", csv_path)
    print()
    print(f"  {C.DIM}Open the document:{C.RESET}")
    print(f"    {C.CYAN}cat {doc_path}{C.RESET}")
    print()
    print(f"  {C.DIM}Evaluate the output:{C.RESET}")
    print(
        f"    {C.CYAN}python evaluate.py --state {state_path} --doc {doc_path}{C.RESET}"
    )
    print()


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="KGMS — Generate a learning document for any research paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Quick start:
          python run.py --paper 2405.20139 --pdf gnn_rag.pdf

        Phase A only (gaps + candidate list, no generation):
          python run.py --paper 2405.20139 --pdf gnn_rag.pdf --phase-a-only

        With custom gaps and deeper reference graph:
          python run.py --paper 2405.20139 --pdf paper.pdf \\
                        --depth 2 --gaps "SPARQL" "entity linking"

        After running, evaluate the document:
          python evaluate.py \\
            --state ./output/phase_a_state_2405.20139.json \\
            --doc   ./output/learning_roadmap_2405.20139.md
        """),
    )

    parser.add_argument(
        "--paper",
        "-p",
        required=True,
        metavar="ID",
        help="Paper identifier: arXiv ID (2405.20139), DOI (10.1109/...), or Semantic Scholar ID",
    )

    parser.add_argument(
        "--pdf",
        "-f",
        metavar="FILE",
        help="Local PDF of the target paper. Strongly recommended — "
        "gap detection from full text is far more accurate than abstract-only",
    )

    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Reference graph depth  "
        "[0=no graph (gap detection only)  "
        "1=direct refs (default, fast)  "
        "2=refs-of-refs  "
        "3=full 3-level graph (slow first run, cached after)]",
    )

    parser.add_argument(
        "--gaps",
        "-g",
        nargs="*",
        default=[],
        metavar="CONCEPT",
        help="Extra concepts to always explain, e.g: --gaps 'attention mechanism' 'SPARQL'",
    )

    parser.add_argument(
        "--out",
        "-o",
        default=".",
        metavar="DIR",
        help="Output directory for document, state, BibTeX, CSV (default: current dir)",
    )

    parser.add_argument(
        "--phase-a-only",
        action="store_true",
        help="Run Phase A only: detect gaps and list candidate papers. Skip generation.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all progress messages (default: single updating line)",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the SQLite API cache before running (forces fresh Semantic Scholar calls)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
