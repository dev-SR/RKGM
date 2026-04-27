#!/usr/bin/env python3
"""
test_phase_a.py — CLI test script for KGMS Phase A

Tests the full Phase A pipeline with detailed printed output:
  - Base article fetch and PDF parsing
  - Reference graph construction
  - Gap detection with source tracing
  - Candidate paper matching per gap
  - BibTeX / CSV export

Usage examples:
  # Minimal — just a paper ID, fetch PDF automatically
  python test_phase_a.py --paper 2405.20139

  # Supply a local PDF for richer gap detection
  python test_phase_a.py --paper 2405.20139 --pdf /path/to/gnn_rag.pdf

  # Choose reference depth (1=fast, 2=medium, 3=thorough)
  python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --depth 2

  # Add concepts you specifically want explained
  python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --gaps "GNN" "SPARQL"

  # Skip candidate matching (gap detection only)
  python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --no-candidates

  # Export results to files
  python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --export ./output/
"""

import argparse
import json
import os
import sys
import time
import textwrap
from dotenv import load_dotenv

load_dotenv()
# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Terminal colours ──────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def _c(text, colour):
    return f"{colour}{text}{C.RESET}"


def banner(text, colour=C.BLUE):
    width = 70
    print()
    print(_c("─" * width, colour))
    print(_c(f"  {text}", colour + C.BOLD))
    print(_c("─" * width, colour))


def section(text):
    print(f"\n{C.BOLD}{C.CYAN}▶ {text}{C.RESET}")


def ok(text):
    print(f"  {C.GREEN}✓{C.RESET} {text}")


def info(text, indent=4):
    prefix = " " * indent
    for line in textwrap.wrap(
        text, width=74, initial_indent=prefix, subsequent_indent=prefix + "  "
    ):
        print(_c(line, C.GRAY))


def warn(text):
    print(f"  {C.YELLOW}⚠{C.RESET} {text}")


def err(text):
    print(f"  {C.RED}✗{C.RESET} {text}")


def kv(key, value, key_width=28):
    print(f"  {C.DIM}{key:<{key_width}}{C.RESET}{value}")


def gap_type_colour(gt):
    return {
        "terminology": C.CYAN,
        "methodology": C.BLUE,
        "benchmark": C.YELLOW,
        "historical": C.WHITE,
        "mathematical": C.GREEN,
    }.get(gt, C.RESET)


def layer_colour(layer):
    return {
        "foundation": C.GREEN,
        "development": C.YELLOW,
        "frontier": C.CYAN,
    }.get(layer, C.RESET)


def conf_bar(conf):
    filled = int(conf * 10)
    bar = "█" * filled + "░" * (10 - filled)
    colour = C.GREEN if conf >= 0.75 else (C.YELLOW if conf >= 0.5 else C.RED)
    return _c(f"[{bar}] {int(conf * 100):3d}%", colour)


# ── Progress callback ─────────────────────────────────────────────────────


def make_callback(verbose: bool):
    log = []

    def cb(msg):
        log.append(msg)
        if verbose:
            print(f"  {C.DIM}ℹ {msg}{C.RESET}")
        else:
            # Single updating line
            truncated = msg[:65] + "…" if len(msg) > 65 else msg
            print(f"\r  {C.DIM}⏳ {truncated:<68}{C.RESET}", end="", flush=True)

    return cb, log


# ── Main test ─────────────────────────────────────────────────────────────


def run_test(args):
    banner("KGMS Phase A — Test Run", C.BLUE)

    # ── Check API key ─────────────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        err("GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com, then:")
        print("  export GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    # ── Validate PDF path if given ────────────────────────────────────────
    pdf_path = args.pdf
    if pdf_path:
        if not os.path.exists(pdf_path):
            err(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        size_kb = os.path.getsize(pdf_path) / 1024
        ok(f"PDF file: {os.path.basename(pdf_path)} ({size_kb:.0f} KB)")
    else:
        warn("No PDF supplied — will try auto-download, then fall back to abstract")
        warn("Use --pdf <path> for richer gap detection")

    kv("Paper input", args.paper)
    kv(
        "Reference depth",
        {
            0: "Level 0  (gap detection only — no graph)",
            1: "Level 1  (direct references — fast)",
            2: "Level 2  (refs of refs — medium)",
            3: "Level 3  (full 3-level graph — slow first run)",
        }.get(args.depth, str(args.depth)),
    )
    kv("User-specified gaps", ", ".join(args.gaps) if args.gaps else "(none)")
    kv("Candidate matching", "enabled" if not args.no_candidates else "disabled")

    # ── Run Phase A ───────────────────────────────────────────────────────
    from pipeline import run_phase_a
    from phase_a.candidates import candidates_to_bibtex, candidates_to_csv

    cb, log = make_callback(verbose=args.verbose)
    t_start = time.time()

    section("Running Phase A pipeline…")
    state = run_phase_a(
        paper_input=args.paper,
        user_gaps=args.gaps or [],
        progress_callback=cb,
        pdf_path=pdf_path,
        reference_depth=args.depth,
    )
    if not args.verbose:
        print()  # newline after progress line

    elapsed = time.time() - t_start

    if state.errors:
        print()
        for e in state.errors:
            err(e)
        sys.exit(1)

    # ── Base article ──────────────────────────────────────────────────────
    section("Base Article")
    ba = state.ba_paper
    kv("Title", ba.title)
    kv("Year", str(ba.year))
    kv("Citations", f"{ba.citation_count:,}")
    kv("arXiv ID", ba.arxiv_id or "—")
    kv("DOI", ba.doi or "—")
    kv("S2 paper ID", ba.paper_id)
    kv("PDF URL", ba.pdf_url or "—")

    word_count = len(state.ba_text.split()) if state.ba_text else 0
    source = "full PDF" if word_count > 500 else "abstract only"
    kv("Text available", f"{word_count:,} words  [{source}]")

    if word_count < 500:
        warn(
            "Only abstract text available — gap detection may miss internal "
            "gaps. Supply a PDF with --pdf for better results."
        )

    # ── Reference graph ───────────────────────────────────────────────────
    section(f"Reference Graph  (depth={args.depth})")
    from phase_a.graph import graph_summary

    summary = graph_summary(state.reference_graph, state.all_papers)

    kv("Total papers", str(summary["total_papers"]))
    kv("Total edges", str(summary["total_edges"]))
    kv("Foundation", _c(str(summary["foundation"]), C.GREEN))
    kv("Development", _c(str(summary["development"]), C.YELLOW))
    kv("Frontier", _c(str(summary["frontier"]), C.CYAN))

    print(f"\n  {'Level':<8} {'Papers':>8}")
    print(f"  {'─' * 8} {'─' * 8}")
    for lvl, count in sorted(summary["by_level"].items()):
        bar = "▓" * min(count // 2, 30)
        print(f"  Level {lvl:<4} {count:>8}  {_c(bar, C.BLUE)}")

    if args.show_papers:
        print()
        print(f"  {'Title':<50} {'Year':>5} {'Cit':>7} {'Layer':<12}")
        print(f"  {'─' * 50} {'─' * 5} {'─' * 7} {'─' * 12}")
        sorted_papers = sorted(
            state.all_papers.values(), key=lambda p: p.trendscore, reverse=True
        )
        for p in sorted_papers[: args.show_papers]:
            lc = layer_colour(p.layer.value)
            print(
                f"  {p.title[:50]:<50} {p.year:>5} "
                f"{p.citation_count:>7,} "
                f"{_c(p.layer.value, lc):<12}"
            )

    # ── Knowledge gaps ────────────────────────────────────────────────────
    gaps = state.gaps
    section(f"Knowledge Gaps  ({len(gaps)} detected)")

    if not gaps:
        err("No gaps detected.")
        warn("Possible causes:")
        print("    • Paper text was abstract-only (use --pdf for full text)")
        print("    • max_tokens was too low (now fixed to 4096)")
        print("    • Self-consistency threshold too high (≥2/3 runs required)")
        print("    • Try adding explicit gaps with --gaps 'concept name'")
        return state

    # Group by type
    by_type = {}
    for g in gaps:
        by_type.setdefault(g.gap_type.value, []).append(g)

    type_summary = "  ".join(
        f"{_c(t, gap_type_colour(t))}:{len(gs)}" for t, gs in by_type.items()
    )
    print(f"  Types: {type_summary}")
    print()

    for i, gap in enumerate(gaps, 1):
        tc = gap_type_colour(gap.gap_type.value)
        lc = layer_colour(gap.layer_hint.value)
        diff_icon = {"beginner": "🌱", "intermediate": "🌿", "advanced": "🌳"}.get(
            gap.difficulty.value, ""
        )

        print(f"  {C.BOLD}Gap {i:02d}: {gap.concept}{C.RESET}")
        print(f"    Type:       {_c(gap.gap_type.value, tc)}")
        print(f"    Difficulty: {diff_icon} {gap.difficulty.value}")
        print(f"    Domain:     {gap.domain}")
        print(f"    Layer hint: {_c(gap.layer_hint.value, lc)}")
        print(f"    Confidence: {conf_bar(gap.confidence)}")
        print(f"    Why needed:")
        info(gap.why_needed, indent=6)
        if gap.source_passage:
            print(f"    Source in paper:")
            info(f'"{gap.source_passage[:200]}"', indent=6)
        print(f"    Retrieval query: {_c(gap.retrieval_query, C.CYAN)}")

        if i < len(gaps):
            print()

    # ── Candidate papers ──────────────────────────────────────────────────
    if args.no_candidates or not state.candidates:
        if args.no_candidates:
            section("Candidate matching skipped (--no-candidates)")
        else:
            warn("No candidates generated")
        _print_summary(state, elapsed, args)
        return state

    section(f"Candidate Papers  ({len(state.candidates)} total)")

    gap_map = {g.gap_id: g for g in gaps}
    cands_by_gap = {}
    for c in state.candidates:
        cands_by_gap.setdefault(c.gap_id, []).append(c)

    for gap_id, cands in cands_by_gap.items():
        gap = gap_map.get(gap_id)
        if not gap:
            continue
        tc = gap_type_colour(gap.gap_type.value)
        print(
            f"\n  {C.BOLD}Gap: {gap.concept}{C.RESET}  [{_c(gap.gap_type.value, tc)}]"
        )
        print(f"  {'─' * 65}")

        for j, c in enumerate(cands[: args.top_k], 1):
            pdf_icon = (
                _c("✅ PDF", C.GREEN) if c.pdf_available else _c("❌ No PDF", C.RED)
            )
            lc = layer_colour(c.paper.layer.value)
            score_bar = "▓" * int(c.relevance_score * 20)
            print(f"    {j}. {C.BOLD}{c.paper.title[:62]}{C.RESET}")
            print(
                f"       Year: {c.paper.year}  "
                f"Citations: {c.paper.citation_count:,}  "
                f"Layer: {_c(c.paper.layer.value, lc)}  "
                f"Score: {_c(f'{c.relevance_score:.3f}', C.YELLOW)}  "
                f"{pdf_icon}"
            )
            if c.paper.arxiv_id:
                print(f"       arXiv: https://arxiv.org/abs/{c.paper.arxiv_id}")
            if c.paper.doi:
                print(f"       DOI:   {c.paper.doi}")
            print(f"       Relevance bar: [{_c(f'{score_bar:<20}', C.BLUE)}]")
            print(f"       Rationale:")
            info(c.rationale.split("\n\n")[0][:250], indent=8)

    # ── Summary + export ──────────────────────────────────────────────────
    _print_summary(state, elapsed, args)

    if args.export:
        _export_results(state, args.export)

    return state


def _print_summary(state, elapsed, args):
    banner("Summary", C.GREEN)
    gaps = state.gaps
    cands = state.candidates
    pdf_avail = sum(1 for c in cands if c.pdf_available)
    unique_p = len({c.paper.paper_id for c in cands})

    kv("Total time", f"{elapsed:.1f}s")
    kv(
        "BA text source",
        "full PDF" if len(state.ba_text.split()) > 500 else "abstract only",
    )
    kv("Reference depth", f"Level {args.depth}")
    kv("Papers in graph", str(len(state.all_papers)))
    kv("Gaps detected", _c(str(len(gaps)), C.GREEN if gaps else C.RED))

    if gaps:
        by_type = {}
        for g in gaps:
            by_type[g.gap_type.value] = by_type.get(g.gap_type.value, 0) + 1
        type_str = ", ".join(f"{t}:{n}" for t, n in sorted(by_type.items()))
        kv("  by type", type_str)
        avg_conf = sum(g.confidence for g in gaps) / len(gaps)
        kv("  avg confidence", f"{int(avg_conf * 100)}%")

    if cands:
        kv(
            "Candidate papers",
            f"{unique_p} unique papers ({len(cands)} gap↔paper links)",
        )
        kv(
            "  PDF available",
            f"{pdf_avail}/{unique_p} ({int(100 * pdf_avail / max(unique_p, 1))}%)",
        )

    print()
    print(_c("  Next steps:", C.BOLD))
    print(f"    1. Review gaps above — disable any you already know")
    if cands:
        print(
            f"    2. Collect PDFs — {unique_p - pdf_avail} papers need manual download"
        )
        if pdf_avail > 0:
            print(
                f"       {pdf_avail} papers have open-access PDFs (auto-downloaded in Phase B)"
            )
    print(f"    3. Run Phase B:  state = run_phase_b(state, auto_fetch=True)")
    if args.export:
        print(f"    4. BibTeX & CSV saved to: {args.export}")


def _export_results(state, export_dir: str):
    from phase_a.candidates import candidates_to_bibtex, candidates_to_csv

    os.makedirs(export_dir, exist_ok=True)
    safe_id = state.ba_paper.paper_id.replace("/", "_")[:20]

    # BibTeX
    bib_path = os.path.join(export_dir, f"candidates_{safe_id}.bib")
    with open(bib_path, "w") as f:
        f.write(candidates_to_bibtex(state.candidates))
    ok(f"BibTeX → {bib_path}")

    # CSV
    csv_path = os.path.join(export_dir, f"candidates_{safe_id}.csv")
    with open(csv_path, "w") as f:
        f.write(candidates_to_csv(state.candidates))
    ok(f"CSV    → {csv_path}")

    # Gap JSON
    gap_path = os.path.join(export_dir, f"gaps_{safe_id}.json")
    gaps_data = [
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
            "confidence": round(g.confidence, 3),
        }
        for g in state.gaps
    ]
    with open(gap_path, "w") as f:
        json.dump(gaps_data, f, indent=2, ensure_ascii=False)
    ok(f"Gaps   → {gap_path}")

    # Summary text
    summary_path = os.path.join(export_dir, f"summary_{safe_id}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Paper: {state.ba_paper.title}\n")
        f.write(f"Year:  {state.ba_paper.year}\n\n")
        f.write(f"Gaps detected: {len(state.gaps)}\n\n")
        for g in state.gaps:
            f.write(
                f"[{g.gap_type.value.upper()}] {g.concept}  "
                f"(conf: {int(g.confidence * 100)}%)\n"
            )
            f.write(f"  Why: {g.why_needed}\n")
            f.write(f'  Source: "{g.source_passage[:150]}"\n\n')
    ok(f"Summary→ {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Test KGMS Phase A — gap detection and candidate matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python test_phase_a.py --paper 2405.20139
          python test_phase_a.py --paper 2405.20139 --pdf gnn_rag.pdf
          python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --depth 2
          python test_phase_a.py --paper 2405.20139 --gaps "GNN" "SPARQL" "KG embedding"
          python test_phase_a.py --paper 2405.20139 --pdf paper.pdf --export ./results/
          python test_phase_a.py --paper 2405.20139 --verbose --show-papers 20
        """),
    )

    parser.add_argument(
        "--paper",
        "-p",
        required=True,
        help="Paper identifier: arXiv ID (2405.20139), DOI (10.1109/...), or S2 ID",
    )
    parser.add_argument(
        "--pdf",
        "-f",
        help="Local PDF file of the base article (strongly recommended for "
        "richer gap detection)",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help=(
            "Reference graph depth.\n"
            "  0 = gap detection only from BA text (no graph, no candidates, fastest)\n"
            "  1 = direct references only (default, fast)\n"
            "  2 = refs-of-refs (medium)\n"
            "  3 = full 3-level graph (slow first run, cached after)"
        ),
    )
    parser.add_argument(
        "--gaps",
        "-g",
        nargs="*",
        default=[],
        help="Extra concepts you want explained (space-separated, quote multi-word)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of candidate papers to show per gap (default: 3)",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Skip candidate matching (test gap detection only)",
    )
    parser.add_argument(
        "--show-papers",
        type=int,
        default=0,
        metavar="N",
        help="Show top-N papers in the reference graph by trendscore",
    )
    parser.add_argument(
        "--export",
        "-e",
        metavar="DIR",
        help="Export gap JSON, BibTeX, CSV, and summary to this directory",
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
        help="Clear the SQLite cache before running (forces fresh API calls)",
    )

    args = parser.parse_args()

    if args.clear_cache:
        from utils.cache import clear_cache

        clear_cache()
        print(_c("  Cache cleared", C.YELLOW))

    run_test(args)


if __name__ == "__main__":
    main()
