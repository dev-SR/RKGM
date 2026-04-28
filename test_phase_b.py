#!/usr/bin/env python3
"""
test_phase_b.py — CLI test script for KGMS Phase B

Runs the full Phase B pipeline and writes a Markdown learning document.
If Phase A has not been run yet (or --rerun-a is passed), runs Phase A first.

Usage examples:
  # Full run — Phase A then Phase B
  python test_phase_b.py --paper 2405.20139 --pdf gnn_rag.pdf

  # Re-use a saved Phase A state (skip re-running A)
  python test_phase_b.py --state phase_a_state.json

  # Control which gaps to generate (by gap_id or concept substring)
  python test_phase_b.py --paper 2405.20139 --pdf p.pdf --only-gaps "KGQA" "GNN"

  # Override output document path
  python test_phase_b.py --paper 2405.20139 --pdf p.pdf --out ./results/

  # Verbose: print every progress event
  python test_phase_b.py --paper 2405.20139 --pdf p.pdf --verbose

  # Depth 0 (no references) — gaps from BA only, phase B generates from abstract
  python test_phase_b.py --paper 2405.20139 --pdf p.pdf --depth 0
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

# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Reuse colour helpers from test_phase_a ────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
    GREEN  = "\033[92m"; YELLOW= "\033[93m"; BLUE  = "\033[94m"
    CYAN   = "\033[96m"; RED   = "\033[91m"; WHITE = "\033[97m"
    GRAY   = "\033[90m"; MAGENTA="\033[95m"

def _c(text, colour): return f"{colour}{text}{C.RESET}"

def banner(text, colour=C.BLUE):
    w = 70
    print(); print(_c("─"*w, colour))
    print(_c(f"  {text}", colour+C.BOLD))
    print(_c("─"*w, colour))

def section(text):
    print(f"\n{C.BOLD}{C.CYAN}▶ {text}{C.RESET}")

def ok(text):   print(f"  {C.GREEN}✓{C.RESET} {text}")
def warn(text): print(f"  {C.YELLOW}⚠{C.RESET} {text}")
def err(text):  print(f"  {C.RED}✗{C.RESET} {text}")
def kv(key, value, w=28): print(f"  {C.DIM}{key:<{w}}{C.RESET}{value}")

def info(text, indent=4):
    prefix = " " * indent
    for line in textwrap.wrap(text, 74, initial_indent=prefix,
                              subsequent_indent=prefix+"  "):
        print(_c(line, C.GRAY))

def conf_bar(conf):
    filled = int(conf * 10)
    bar    = "█"*filled + "░"*(10-filled)
    colour = C.GREEN if conf >= 0.75 else (C.YELLOW if conf >= 0.5 else C.RED)
    return _c(f"[{bar}] {int(conf*100):3d}%", colour)

def gap_type_colour(gt):
    return {"terminology":C.CYAN,"methodology":C.BLUE,"benchmark":C.YELLOW,
            "historical":C.WHITE,"mathematical":C.GREEN}.get(gt, C.RESET)

def layer_colour(layer):
    return {"foundation":C.GREEN,"development":C.YELLOW,"frontier":C.CYAN}.get(layer, C.RESET)


# ── State serialisation helpers ───────────────────────────────────────────

def save_phase_a_state(state, path: str):
    """Serialise Phase A state to JSON for re-use."""
    from core.models import Layer, GapType, Difficulty

    data = {
        "ba_paper": {
            "paper_id":      state.ba_paper.paper_id,
            "title":         state.ba_paper.title,
            "abstract":      state.ba_paper.abstract,
            "year":          state.ba_paper.year,
            "citation_count":state.ba_paper.citation_count,
            "arxiv_id":      state.ba_paper.arxiv_id,
            "doi":           state.ba_paper.doi,
            "pdf_url":       state.ba_paper.pdf_url,
            "layer":         state.ba_paper.layer.value,
            "trendscore":    state.ba_paper.trendscore,
        },
        "ba_text_snippet": state.ba_text[:500] if state.ba_text else "",
        "ba_text_words":   len(state.ba_text.split()) if state.ba_text else 0,
        "gaps": [
            {
                "gap_id":          g.gap_id,
                "concept":         g.concept,
                "gap_type":        g.gap_type.value,
                "difficulty":      g.difficulty.value,
                "domain":          g.domain,
                "why_needed":      g.why_needed,
                "layer_hint":      g.layer_hint.value,
                "retrieval_query": g.retrieval_query,
                "source_passage":  g.source_passage,
                "confidence":      round(g.confidence, 4),
            }
            for g in state.gaps
        ],
        "candidates": [
            {
                "gap_id":          c.gap_id,
                "paper_id":        c.paper.paper_id,
                "title":           c.paper.title,
                "year":            c.paper.year,
                "citation_count":  c.paper.citation_count,
                "layer":           c.paper.layer.value,
                "trendscore":      round(c.paper.trendscore, 3),
                "arxiv_id":        c.paper.arxiv_id,
                "doi":             c.paper.doi,
                "pdf_url":         c.paper.pdf_url,
                "relevance_score": round(c.relevance_score, 4),
                "rationale":       c.rationale,
                "pdf_available":   c.pdf_available,
            }
            for c in state.candidates
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_phase_a_state(path: str):
    """Restore a minimal PipelineState from saved JSON."""
    from core.models import (
        PipelineState, Paper, KnowledgeGap, CandidatePaper,
        GapType, Difficulty, Layer
    )

    with open(path) as f:
        data = json.load(f)

    # Reconstruct BA
    bp = data["ba_paper"]
    ba = Paper(
        paper_id      = bp["paper_id"],
        title         = bp["title"],
        abstract      = bp["abstract"],
        year          = bp["year"],
        citation_count= bp["citation_count"],
        arxiv_id      = bp.get("arxiv_id"),
        doi           = bp.get("doi"),
        pdf_url       = bp.get("pdf_url"),
        level         = 0,
        layer         = Layer(bp.get("layer", "development")),
        trendscore    = bp.get("trendscore", 0.0),
    )

    # Reconstruct gaps
    gaps = []
    for g in data["gaps"]:
        gaps.append(KnowledgeGap(
            gap_id          = g["gap_id"],
            concept         = g["concept"],
            gap_type        = GapType(g["gap_type"]),
            difficulty      = Difficulty(g["difficulty"]),
            domain          = g["domain"],
            why_needed      = g["why_needed"],
            layer_hint      = Layer(g["layer_hint"]),
            retrieval_query = g["retrieval_query"],
            source_passage  = g["source_passage"],
            confidence      = g["confidence"],
        ))

    # Reconstruct candidate papers
    candidates = []
    paper_cache = {}
    for c in data["candidates"]:
        pid = c["paper_id"]
        if pid not in paper_cache:
            paper_cache[pid] = Paper(
                paper_id      = pid,
                title         = c["title"],
                abstract      = "",
                year          = c["year"],
                citation_count= c["citation_count"],
                layer         = Layer(c.get("layer", "development")),
                trendscore    = c.get("trendscore", 0.0),
                arxiv_id      = c.get("arxiv_id"),
                doi           = c.get("doi"),
                pdf_url       = c.get("pdf_url"),
                level         = 1,
            )
        candidates.append(CandidatePaper(
            paper           = paper_cache[pid],
            gap_id          = c["gap_id"],
            relevance_score = c["relevance_score"],
            rationale       = c["rationale"],
            pdf_available   = c["pdf_available"],
        ))

    state = PipelineState()
    state.ba_paper   = ba
    state.ba_text    = data.get("ba_text_snippet", ba.abstract)
    state.gaps       = gaps
    state.candidates = candidates
    return state


# ── Progress callback with section tracking ────────────────────────────────

def make_callback(verbose: bool):
    log = []
    current = [""]
    def cb(msg):
        log.append(msg)
        if verbose:
            print(f"  {C.DIM}ℹ {msg}{C.RESET}")
        else:
            truncated = msg[:66]+"…" if len(msg) > 66 else msg
            print(f"\r  {C.DIM}⏳ {truncated:<70}{C.RESET}", end="", flush=True)
    return cb, log


# ── Print Phase A recap ────────────────────────────────────────────────────

def print_phase_a_recap(state):
    section("Phase A Recap — What Phase B will work with")
    kv("Base article",    state.ba_paper.title)
    kv("BA text",         f"{len(state.ba_text.split()):,} words")
    kv("Gaps to explain", str(len(state.gaps)))
    kv("Candidate links", str(len(state.candidates)))

    unique_papers = {c.paper.paper_id: c for c in state.candidates}
    pdf_avail     = sum(1 for c in unique_papers.values() if c.pdf_available)
    kv("Unique ref papers", str(len(unique_papers)))
    kv("PDFs auto-available",
       f"{pdf_avail}/{len(unique_papers)}  "
       f"({int(100*pdf_avail/max(len(unique_papers),1))}%)")

    print()
    print(f"  {'#':<4} {'Concept':<30} {'Type':<14} {'Conf':>5}  {'Layer':<12}  {'Best candidate (top-1)'}")
    print(f"  {'─'*4} {'─'*30} {'─'*14} {'─'*5}  {'─'*12}  {'─'*40}")

    gap_map   = {g.gap_id: g for g in state.gaps}
    cands_map = {}
    for c in state.candidates:
        cands_map.setdefault(c.gap_id, []).append(c)

    for i, gap in enumerate(state.gaps, 1):
        tc       = gap_type_colour(gap.gap_type.value)
        lc       = layer_colour(gap.layer_hint.value)
        top_cand = cands_map.get(gap.gap_id, [{}])[0] if cands_map.get(gap.gap_id) else None
        cand_str = ""
        if top_cand and hasattr(top_cand, 'paper'):
            pdf_icon = _c("✅","") if top_cand.pdf_available else _c("❌","")
            cand_str = f"{pdf_icon} {top_cand.paper.title[:38]} ({top_cand.paper.year})"
        print(
            f"  {i:<4} "
            f"{gap.concept[:30]:<30} "
            f"{_c(gap.gap_type.value[:14], tc):<23} "
            f"{int(gap.confidence*100):>4}%  "
            f"{_c(gap.layer_hint.value[:12], lc):<21}  "
            f"{cand_str}"
        )


# ── Print Phase B results ─────────────────────────────────────────────────

def print_phase_b_results(state, doc_path: str, elapsed: float):
    section("Generated Explanations")

    for i, exp in enumerate(state.explanations, 1):
        status_icon = "🟡 abstract" if exp.is_abstract_only else "✅ full-text"
        print(f"\n  {C.BOLD}{i:02d}. {exp.concept}{C.RESET}  [{status_icon}]")
        print(f"      Confidence:   {conf_bar(exp.confidence)}")
        print(f"      Citations:    {len(exp.source_citations)}")
        if exp.dependency_note:
            print(f"      Dependency:   {_c(exp.dependency_note[:70], C.GRAY)}")

        # Show first 200 chars of explanation
        preview = exp.explanation_text[:200].replace("\n", " ")
        print(f"      Preview:")
        info(f'"{preview}…"', indent=6)

        # Show citation breakdown
        if exp.source_citations and not exp.is_abstract_only:
            paper_ids = {cid.split("::")[0] for cid in exp.source_citations}
            papers_str = ", ".join(list(paper_ids)[:3])
            if len(paper_ids) > 3:
                papers_str += f" +{len(paper_ids)-3} more"
            print(f"      Sourced from: {_c(papers_str, C.CYAN)}")

    # Coverage report
    if hasattr(state, "_coverage_report") and state._coverage_report:
        cov = state._coverage_report
        from eval.coverage import coverage_badge
        badge, label = coverage_badge(cov.coverage_score)
        section(f"Coverage Report — {badge} {label}")
        kv("Covered (full-text)", _c(str(cov.covered),       C.GREEN))
        kv("Abstract-only",       _c(str(cov.abstract_only), C.YELLOW))
        kv("Uncited",             _c(str(cov.uncited),       C.YELLOW))
        kv("Missing",             _c(str(cov.missing),       C.RED))
        if cov.recommendations:
            print()
            for rec in cov.recommendations:
                icon = "✅" if "excellent" in rec.lower() else "ℹ️"
                print(f"  {icon}  {rec}")

    # Ordering
    if state.ordered_gap_ids:
        section("Chronological Learning Order")
        gap_map = {g.gap_id: g for g in state.gaps}
        for i, gid in enumerate(state.ordered_gap_ids, 1):
            gap = gap_map.get(gid)
            if not gap:
                continue
            lc  = layer_colour(gap.layer_hint.value)
            exp = next((e for e in state.explanations if e.gap_id == gid), None)
            status = "✅" if exp and not exp.is_abstract_only else "🟡"
            print(f"  {i:>2}. {status} {gap.concept:<35} "
                  f"[{_c(gap.layer_hint.value, lc)}]")

        if state.dependencies:
            print(f"\n  Dependency links:")
            for dep in state.dependencies[:8]:
                before = gap_map.get(dep['before'])
                after  = gap_map.get(dep['after'])
                if before and after:
                    print(f"    {before.concept} → {after.concept}")
                    info(dep.get('reason',''), indent=6)

    # Document written
    section("Output Document")
    kv("Path",        doc_path)
    kv("Size",        f"{os.path.getsize(doc_path):,} bytes")
    kv("Total time",  f"{elapsed:.1f}s")

    # First 600 chars preview
    with open(doc_path) as f:
        preview = f.read(600)
    print(f"\n  {C.DIM}{'─'*65}{C.RESET}")
    for line in preview.splitlines()[:18]:
        print(f"  {C.GRAY}{line}{C.RESET}")
    print(f"  {C.DIM}… (see full document at {doc_path}){C.RESET}")


# ── Main ──────────────────────────────────────────────────────────────────

def run_test(args):
    banner("KGMS Phase B — Test Run", C.MAGENTA)

    if not os.environ.get("GROQ_API_KEY"):
        err("GROQ_API_KEY not set.")
        print("  export GROQ_API_KEY='gsk_...'  (free at console.groq.com)")
        sys.exit(1)

    # ── Resolve output dir ────────────────────────────────────────────────
    out_dir   = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    safe_id   = (args.paper or "paper").replace("/","_").replace(":","_")[:20]
    state_path = os.path.join(out_dir, f"phase_a_state_{safe_id}.json")
    doc_path   = os.path.join(out_dir, f"learning_roadmap_{safe_id}.md")

    kv("Output directory",  out_dir)
    kv("Document path",     doc_path)
    kv("State cache",       state_path)

    # ── Load or run Phase A ───────────────────────────────────────────────
    state = None

    if args.state:
        # Explicit state file
        if not os.path.exists(args.state):
            err(f"State file not found: {args.state}")
            sys.exit(1)
        section("Loading Phase A state from file")
        state = load_phase_a_state(args.state)
        ok(f"Loaded: {len(state.gaps)} gaps, {len(state.candidates)} candidates")

    elif not args.rerun_a and os.path.exists(state_path):
        # Cached state from a previous run
        section("Loading cached Phase A state")
        state = load_phase_a_state(state_path)
        ok(f"Cached state: {len(state.gaps)} gaps, {len(state.candidates)} candidates")
        warn("Use --rerun-a to force a fresh Phase A run")

    if state is None:
        # Run Phase A fresh
        if not args.paper:
            err("--paper is required when no saved state is available")
            sys.exit(1)

        if args.pdf and not os.path.exists(args.pdf):
            err(f"PDF file not found: {args.pdf}")
            sys.exit(1)

        section("Running Phase A first…")
        cb, log = make_callback(verbose=args.verbose)
        from pipeline import run_phase_a
        try:
            state = run_phase_a(
                paper_input    = args.paper,
                user_gaps      = args.gaps or [],
                progress_callback = cb,
                pdf_path       = args.pdf,
                reference_depth= args.depth,
            )
        except Exception as e:
            print()
            err(f"Phase A crashed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not args.verbose:
            print()  # newline after progress line

        if state.errors:
            for e in state.errors:
                err(e)
            sys.exit(1)

        # Save state for re-use
        save_phase_a_state(state, state_path)
        ok(f"Phase A state saved → {state_path}")

    # ── Filter gaps if --only-gaps specified ──────────────────────────────
    if args.only_gaps:
        keywords = [k.lower() for k in args.only_gaps]
        original_count = len(state.gaps)
        state.gaps = [
            g for g in state.gaps
            if any(kw in g.concept.lower() or kw in g.gap_type.value for kw in keywords)
        ]
        state.candidates = [
            c for c in state.candidates
            if c.gap_id in {g.gap_id for g in state.gaps}
        ]
        ok(f"Gap filter applied: {len(state.gaps)}/{original_count} gaps selected")
        if not state.gaps:
            err(f"No gaps match filter: {args.only_gaps}")
            sys.exit(1)

    # ── Print Phase A recap ───────────────────────────────────────────────
    print_phase_a_recap(state)

    # ── Run Phase B ───────────────────────────────────────────────────────
    section("Running Phase B — PDF ingestion + retrieval + generation…")
    cb, log = make_callback(verbose=args.verbose)

    # Restore full ba_text if loaded from state (snippet only)
    if len(state.ba_text.split()) < 200 and args.pdf:
        from utils.ingest import parse_pdf_to_markdown
        md = parse_pdf_to_markdown(args.pdf)
        if md:
            state.ba_text = md
            ok(f"Restored full BA text from PDF: {len(md.split()):,} words")

    t_start = time.time()
    from pipeline import run_phase_b
    try:
        state = run_phase_b(
            state             = state,
            pdf_paths         = {},    # auto-fetch handles it
            auto_fetch        = True,
            progress_callback = cb,
        )
    except Exception as e:
        print()
        err(f"Phase B crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not args.verbose:
        print()

    elapsed = time.time() - t_start

    if state.errors:
        for e in state.errors:
            warn(e)   # non-fatal: coverage warnings etc.

    if not state.final_document:
        err("Document generation failed — no output produced")
        sys.exit(1)

    # ── Write document ────────────────────────────────────────────────────
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(state.final_document)
    ok(f"Learning document written → {doc_path}")

    # ── Print results ─────────────────────────────────────────────────────
    print_phase_b_results(state, doc_path, elapsed)

    # ── Final summary ─────────────────────────────────────────────────────
    banner("Summary", C.GREEN)
    kv("Paper",           state.ba_paper.title[:60])
    kv("Phase A gaps",    str(len(state.gaps)))
    kv("Explanations",    str(len(state.explanations)))
    full_text = sum(1 for e in state.explanations if not e.is_abstract_only)
    abstract  = sum(1 for e in state.explanations if e.is_abstract_only)
    kv("  Full-text",     _c(str(full_text), C.GREEN))
    kv("  Abstract-only", _c(str(abstract),  C.YELLOW if abstract else C.GREEN))
    avg_conf = sum(e.confidence for e in state.explanations) / max(len(state.explanations), 1)
    kv("  Avg confidence",f"{int(avg_conf*100)}%")
    kv("Total time",      f"{elapsed:.1f}s")
    kv("Document",        doc_path)

    if hasattr(state, "_coverage_report") and state._coverage_report:
        from eval.coverage import coverage_badge
        badge, label = coverage_badge(state._coverage_report.coverage_score)
        kv("Coverage",    f"{badge} {label}")

    print()
    ok("Phase B complete. Open the document:")
    print(f"    {C.CYAN}cat {doc_path}{C.RESET}")
    print(f"    {C.CYAN}# or open in any Markdown viewer{C.RESET}")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test KGMS Phase B — generate a learning document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full pipeline — Phase A then Phase B
          python test_phase_b.py --paper 2405.20139 --pdf gnn_rag.pdf

          # Skip Phase A if already cached in output dir
          python test_phase_b.py --paper 2405.20139

          # Force re-run Phase A
          python test_phase_b.py --paper 2405.20139 --pdf p.pdf --rerun-a

          # Load explicit Phase A state file
          python test_phase_b.py --state ./output/phase_a_state_2405.20139.json

          # Only generate explanations for specific gaps
          python test_phase_b.py --paper 2405.20139 --only-gaps "KGQA" "GNN"

          # Depth 0 — gaps from BA only, no reference candidates
          python test_phase_b.py --paper 2405.20139 --pdf p.pdf --depth 0
        """)
    )
    parser.add_argument("--paper", "-p",
        help="Paper ID (arXiv, DOI, S2). Required if no --state is given.")
    parser.add_argument("--pdf", "-f",
        help="Local PDF of base article (strongly recommended)")
    parser.add_argument("--state", "-s",
        help="Path to a saved Phase A state JSON (skips running Phase A)")
    parser.add_argument("--rerun-a", action="store_true",
        help="Force Phase A to re-run even if a cached state exists")
    parser.add_argument("--depth", "-d", type=int, choices=[0,1,2,3], default=1,
        help="Reference depth for Phase A (0=no graph, 1=direct refs, 2-3=deeper). "
             "Ignored if loading from --state.")
    parser.add_argument("--gaps", "-g", nargs="*", default=[],
        help="Extra concepts to always include in gap detection")
    parser.add_argument("--only-gaps", nargs="*", metavar="KEYWORD",
        help="Only generate explanations for gaps matching these keywords")
    parser.add_argument("--out", "-o", default=".",
        help="Output directory for document + state cache (default: current dir)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Print all progress messages")
    parser.add_argument("--clear-cache", action="store_true",
        help="Clear SQLite cache before running")

    args = parser.parse_args()

    if not args.paper and not args.state:
        parser.error("Provide either --paper <id> or --state <path>")

    if args.clear_cache:
        from utils.cache import clear_cache
        clear_cache()
        print(_c("  Cache cleared", C.YELLOW))

    run_test(args)


if __name__ == "__main__":
    main()
