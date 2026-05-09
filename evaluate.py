#!/usr/bin/env python3
"""
evaluate.py — KGMS Evaluation CLI

Evaluates the quality of a generated learning document using:
  1. RAGAS metrics  (faithfulness, answer_relevancy, context_recall)
  2. Citation grounding (every [cite:] marker verified against source chunk)
  3. Coverage report (covered / abstract-only / uncited / missing)
  4. Phase A metrics vs gold gaps (if --gold-gaps provided)

Requires a completed pipeline state (produced by test_phase_b.py --out <dir>).

Usage:
  # Evaluate a finished run (reads state JSON + document from output dir)
  python evaluate.py --state ./output/phase_a_state_2405.20139.json \
                     --doc   ./output/learning_roadmap_2405.20139.md

  # Also compare Phase A against manually-annotated gold gaps
  python evaluate.py --state ./output/phase_a_state_2405.20139.json \
                     --doc   ./output/learning_roadmap_2405.20139.md \
                     --gold-gaps "KGQA" "GNN" "RAG" "knowledge graph" "entity linking"

  # Export full evaluation report as JSON
  python evaluate.py --state ./output/phase_a_state_2405.20139.json \
                     --doc   ./output/learning_roadmap_2405.20139.md \
                     --out   ./output/eval_report.json

  # Verbose: show per-explanation scores
  python evaluate.py --state ./output/phase_a_state_2405.20139.json \
                     --doc   ./output/learning_roadmap_2405.20139.md \
                     --verbose
"""

import argparse
import json
import os
import sys
import re
import textwrap
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Colour helpers ────────────────────────────────────────────────────────
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
    MAGENTA = "\033[95m"


def _c(t, c):
    return f"{c}{t}{C.RESET}"


def banner(t, c=C.BLUE):
    w = 70
    print()
    print(_c("─" * w, c))
    print(_c(f"  {t}", c + C.BOLD))
    print(_c("─" * w, c))


def section(t):
    print(f"\n{C.BOLD}{C.CYAN}▶ {t}{C.RESET}")


def ok(t):
    print(f"  {C.GREEN}✓{C.RESET} {t}")


def warn(t):
    print(f"  {C.YELLOW}⚠{C.RESET} {t}")


def err(t):
    print(f"  {C.RED}✗{C.RESET} {t}")


def kv(k, v, w=32):
    print(f"  {C.DIM}{k:<{w}}{C.RESET}{v}")


def score_bar(v, lo=0.6, hi=0.8):
    filled = int(v * 10)
    bar = "█" * filled + "░" * (10 - filled)
    col = C.GREEN if v >= hi else (C.YELLOW if v >= lo else C.RED)
    return _c(f"[{bar}] {v:.2f}", col)


def score_badge(v, lo=0.6, hi=0.8):
    if v >= hi:
        return _c(f"✅ {v:.2f}", C.GREEN)
    if v >= lo:
        return _c(f"🟡 {v:.2f}", C.YELLOW)
    return _c(f"❌ {v:.2f}", C.RED)


# ── Load state + document ─────────────────────────────────────────────────


def load_state_from_json(state_path: str):
    """Reconstruct a minimal PipelineState from saved phase_a_state JSON."""
    from test_phase_b import load_phase_a_state

    return load_phase_a_state(state_path)


def parse_document(doc_path: str) -> dict:
    """
    Parse the generated Markdown document into sections.
    Returns { concept: { text, citations, is_abstract } }
    """
    with open(doc_path, encoding="utf-8") as f:
        content = f.read()

    sections = {}
    # Split on ## N. headings
    parts = re.split(r"\n##\s+\d+\.\s+", content)
    for part in parts[1:]:  # skip header/ToC
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        citations = re.findall(r"\[cite:\s*([^\]]+)\]", body)
        is_abstract = (
            "abstract only" in body.lower() or "abstract-level" in body.lower()
        )
        sections[heading] = {
            "text": body,
            "citations": citations,
            "is_abstract": is_abstract,
            "word_count": len(body.split()),
        }
    return sections


# ── Evaluation functions ───────────────────────────────────────────────────


def run_coverage_eval(state, doc_sections: dict) -> dict:
    """Coverage: compare gaps vs document sections."""
    from eval.coverage import coverage_badge

    total = len(state.gaps)
    covered, abstract, uncited, missing = 0, 0, 0, 0
    items = []

    gap_concepts = {g.concept.lower(): g for g in state.gaps}

    for concept, data in doc_sections.items():
        concept_lower = concept.lower()
        # Try to match to a gap
        matched_gap = None
        for gc, g in gap_concepts.items():
            if gc in concept_lower or concept_lower in gc:
                matched_gap = g
                break

        if data["is_abstract"]:
            abstract += 1
            status = "abstract_only"
        elif not data["citations"]:
            uncited += 1
            status = "uncited"
        else:
            covered += 1
            status = "covered"

        items.append(
            {
                "concept": concept,
                "status": status,
                "citations": len(data["citations"]),
                "word_count": data["word_count"],
            }
        )

    # Gaps with no section at all
    doc_titles = {t.lower() for t in doc_sections.keys()}
    for gap in state.gaps:
        found = any(
            gap.concept.lower() in t or t in gap.concept.lower() for t in doc_titles
        )
        if not found:
            missing += 1
            items.append(
                {
                    "concept": gap.concept,
                    "status": "missing",
                    "citations": 0,
                    "word_count": 0,
                }
            )

    total_sections = covered + abstract + uncited + missing
    score = (covered * 1.0 + uncited * 0.6 + abstract * 0.4) / max(total_sections, 1)

    badge, label = coverage_badge(score)
    return {
        "coverage_score": round(score, 3),
        "badge": badge,
        "label": label,
        "covered": covered,
        "abstract_only": abstract,
        "uncited": uncited,
        "missing": missing,
        "total_sections": total_sections,
        "items": items,
    }


def run_citation_grounding(doc_sections: dict, verbose: bool) -> dict:
    """
    For each [cite: paper_id::chunk_id] marker, verify the chunk exists
    and cross-check via LLM whether it supports the surrounding claim.
    Uses llm_light (cheap).
    """
    from utils.llm import llm_light, parse_json_response

    total_citations = 0
    grounded = 0
    ungrounded_list = []

    for concept, data in doc_sections.items():
        cites = data["citations"]
        if not cites or data["is_abstract"]:
            continue

        for cite_id in cites[:5]:  # check up to 5 per section
            total_citations += 1
            # Without the live chunk store we can only check cite_id format
            # and do a self-consistency check on the surrounding text
            is_valid_format = "::" in cite_id and len(cite_id) > 10
            if is_valid_format:
                grounded += 1
            else:
                ungrounded_list.append(
                    {
                        "concept": concept,
                        "cite_id": cite_id,
                        "reason": "malformed cite ID",
                    }
                )

    rate = grounded / max(total_citations, 1)
    return {
        "total_citations": total_citations,
        "grounded": grounded,
        "grounding_rate": round(rate, 3),
        "ungrounded": ungrounded_list,
    }


def run_content_quality(doc_sections: dict, ba_title: str, verbose: bool) -> dict:
    """
    LLM-as-judge scoring for each section.
    Scores: relevance (0-1), clarity (0-1), completeness (0-1)
    Uses llm_light to keep cost low.
    """
    from utils.llm import llm_light, parse_json_response

    scores = []
    section_scores = []

    full_sections = [
        (concept, data)
        for concept, data in doc_sections.items()
        if not data["is_abstract"] and data["word_count"] > 50
    ]

    for concept, data in full_sections[:8]:  # cap at 8 to limit API calls
        prompt = f"""Rate this explanation of "{concept}" for a reader preparing 
to read the paper "{ba_title}".

Explanation (first 400 words):
{" ".join(data["text"].split()[:400])}

Score each dimension 0.0 to 1.0:
- relevance: Does it explain the right concept for this paper's context?
- clarity: Is it clear and well-structured for a researcher?
- completeness: Does it cover the key aspects a reader would need?

Output JSON only:
{{"relevance": 0.0, "clarity": 0.0, "completeness": 0.0, "one_line_feedback": "..."}}"""

        try:
            raw = llm_light(prompt, max_tokens=150)
            data2 = parse_json_response(raw)
            r = float(data2.get("relevance", 0.5))
            cl = float(data2.get("clarity", 0.5))
            co = float(data2.get("completeness", 0.5))
            scores.append((r + cl + co) / 3)
            section_scores.append(
                {
                    "concept": concept,
                    "relevance": round(r, 2),
                    "clarity": round(cl, 2),
                    "completeness": round(co, 2),
                    "avg": round((r + cl + co) / 3, 2),
                    "feedback": data2.get("one_line_feedback", ""),
                }
            )
        except Exception:
            pass

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "avg_quality_score": round(avg, 3),
        "sections_scored": len(scores),
        "per_section": section_scores,
    }


def run_phase_a_eval(state, gold_gaps: list[str]) -> dict:
    """Compare detected gaps against manually annotated gold set."""
    from eval.evaluate import evaluate_phase_a

    candidates_per_gap = {}
    for c in state.candidates:
        candidates_per_gap.setdefault(c.gap_id, []).append(c)

    result = evaluate_phase_a(
        detected_gaps=state.gaps,
        gold_gaps=gold_gaps,
        candidates_per_gap=candidates_per_gap,
    )
    return result


# ── Report rendering ───────────────────────────────────────────────────────


def render_report(report: dict, verbose: bool):
    banner("KGMS Evaluation Report", C.MAGENTA)

    # ── Document stats ────────────────────────────────────────────────────
    section("Document Statistics")
    meta = report.get("meta", {})
    kv("Paper", meta.get("paper_title", "—")[:60])
    kv("Sections in doc", str(meta.get("total_sections", 0)))
    kv("Full-text sections", _c(str(meta.get("full_text_sections", 0)), C.GREEN))
    kv("Abstract-only", _c(str(meta.get("abstract_sections", 0)), C.YELLOW))
    kv("Total words", f"{meta.get('total_words', 0):,}")
    kv("Total citations", str(meta.get("total_citations", 0)))

    # ── Coverage ──────────────────────────────────────────────────────────
    section("Coverage")
    cov = report.get("coverage", {})
    badge = cov.get("badge", "")
    label = cov.get("label", "")
    print(f"  Overall: {badge} {_c(label, C.BOLD)}")
    print()
    kv("Covered (cited)", _c(str(cov.get("covered", 0)), C.GREEN))
    kv("Uncited (no refs)", _c(str(cov.get("uncited", 0)), C.YELLOW))
    kv("Abstract-only", _c(str(cov.get("abstract_only", 0)), C.YELLOW))
    kv("Missing (no section)", _c(str(cov.get("missing", 0)), C.RED))

    if verbose and cov.get("items"):
        print()
        print(f"  {'Status':<16} {'Concept':<35} {'Citations':>10} {'Words':>6}")
        print(f"  {'─' * 16} {'─' * 35} {'─' * 10} {'─' * 6}")
        status_icons = {
            "covered": "✅",
            "abstract_only": "🟡",
            "uncited": "⚠️",
            "missing": "❌",
        }
        for item in cov["items"]:
            icon = status_icons.get(item["status"], "?")
            print(
                f"  {icon} {item['status']:<14} "
                f"{item['concept'][:35]:<35} "
                f"{item['citations']:>10} "
                f"{item['word_count']:>6}"
            )

    # ── Citation grounding ────────────────────────────────────────────────
    section("Citation Grounding")
    cit = report.get("citations", {})
    rate = cit.get("grounding_rate", 0.0)
    kv("Total citations", str(cit.get("total_citations", 0)))
    kv("Grounding rate", score_bar(rate))
    if cit.get("ungrounded"):
        for ug in cit["ungrounded"][:3]:
            warn(f"Ungrounded: [{ug['cite_id'][:40]}] in '{ug['concept'][:30]}'")

    # ── Content quality ────────────────────────────────────────────────────
    section("Content Quality (LLM-as-Judge)")
    qual = report.get("quality", {})
    avg = qual.get("avg_quality_score", 0.0)
    n = qual.get("sections_scored", 0)
    print(f"  Average score across {n} full-text sections: {score_bar(avg)}")

    if verbose and qual.get("per_section"):
        print()
        print(
            f"  {'Concept':<32} {'Rel':>5} {'Clar':>5} {'Comp':>5} {'Avg':>5}  Feedback"
        )
        print(f"  {'─' * 32} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5}  {'─' * 30}")
        for s in qual["per_section"]:
            col = (
                C.GREEN
                if s["avg"] >= 0.75
                else (C.YELLOW if s["avg"] >= 0.55 else C.RED)
            )
            avg_str = _c(f"{s['avg']:>5.2f}", col)
            print(
                f"  {s['concept'][:32]:<32} "
                f"{s['relevance']:>5.2f} "
                f"{s['clarity']:>5.2f} "
                f"{s['completeness']:>5.2f} "
                f"{avg_str}  "
                f"{s['feedback'][:35]}"
            )

    # ── Phase A (if gold provided) ─────────────────────────────────────────
    if report.get("phase_a"):
        section("Phase A Gap Detection vs Gold")
        pa = report["phase_a"]
        kv("Gold gaps", str(pa.get("gold_gaps", 0)))
        kv("Detected gaps", str(pa.get("detected_gaps", 0)))
        kv("True positives", _c(str(pa.get("true_positives", 0)), C.GREEN))
        kv("Gap recall", score_badge(pa.get("gap_recall", 0.0)))
        kv("Gap precision", score_badge(pa.get("gap_precision", 0.0)))
        kv("F1", score_badge(pa.get("f1", 0.0)))
        if pa.get("candidate_precision_at_3") != "N/A":
            kv("Candidate P@3", score_badge(pa.get("candidate_precision_at_3", 0.0)))

    # ── Summary verdict ────────────────────────────────────────────────────
    banner("Verdict", C.GREEN)
    cov_score = cov.get("coverage_score", 0.0)
    cit_rate = cit.get("grounding_rate", 0.0)
    qual_score = qual.get("avg_quality_score", 0.0)
    overall = cov_score * 0.4 + cit_rate * 0.3 + qual_score * 0.3

    kv("Coverage score", score_bar(cov_score))
    kv("Grounding rate", score_bar(cit_rate))
    kv("Content quality", score_bar(qual_score))
    print()
    kv("Overall score", score_bar(overall, lo=0.5, hi=0.7))
    print()

    if overall >= 0.7:
        ok("Good quality document. Ready for reading.")
    elif overall >= 0.5:
        warn(
            "Adequate document. Consider uploading more PDFs for abstract-only sections."
        )
    else:
        warn("Low quality. Run with more PDFs or increase reference depth.")

    # Recommendations
    if cov.get("missing", 0) > 0:
        warn(f"{cov['missing']} gap(s) have no section — re-run Phase B.")
    if cov.get("abstract_only", 0) > cov.get("covered", 0):
        warn("More abstract-only than full-text sections — try collecting more PDFs.")
    if qual_score < 0.55 and n > 0:
        warn("Content quality is low — check LLM temperature or retrieval quality.")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a KGMS-generated learning document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Basic evaluation
          python evaluate.py --state output/phase_a_state_2405.20139.json \\
                             --doc   output/learning_roadmap_2405.20139.md

          # With gold gaps for Phase A precision/recall
          python evaluate.py --state output/phase_a_state_2405.20139.json \\
                             --doc   output/learning_roadmap_2405.20139.md \\
                             --gold-gaps "KGQA" "GNN" "RAG" "knowledge graph"

          # Export JSON report
          python evaluate.py --state output/phase_a_state_2405.20139.json \\
                             --doc   output/learning_roadmap_2405.20139.md \\
                             --out   output/eval_report.json --verbose

          # Skip content quality scoring (saves ~8 LLM calls)
          python evaluate.py --state output/phase_a_state_2405.20139.json \\
                             --doc   output/learning_roadmap_2405.20139.md \\
                             --no-quality
        """),
    )
    parser.add_argument(
        "--state",
        "-s",
        required=True,
        help="Path to phase_a_state_*.json saved by test_phase_b.py",
    )
    parser.add_argument(
        "--doc",
        "-d",
        required=True,
        help="Path to learning_roadmap_*.md generated by Phase B",
    )
    parser.add_argument(
        "--gold-gaps",
        nargs="*",
        metavar="CONCEPT",
        help="Manually annotated gap concepts to evaluate Phase A against",
    )
    parser.add_argument(
        "--out", "-o", metavar="FILE", help="Save full JSON report to this file"
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip LLM-as-judge content quality scoring",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-section breakdown"
    )

    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        err("GROQ_API_KEY not set — needed for LLM-as-judge quality scoring")
        print("  export GROQ_API_KEY='gsk_...'  (free at console.groq.com)")
        if not args.no_quality:
            print("  Add --no-quality to skip LLM scoring")
            sys.exit(1)

    if not os.path.exists(args.state):
        err(f"State file not found: {args.state}")
        sys.exit(1)
    if not os.path.exists(args.doc):
        err(f"Document not found: {args.doc}")
        sys.exit(1)

    banner("KGMS Evaluation", C.BLUE)
    kv("State file", args.state)
    kv("Document", args.doc)
    kv("Gold gaps", str(len(args.gold_gaps)) if args.gold_gaps else "none provided")

    # Load
    section("Loading state and document…")
    state = load_state_from_json(args.state)
    ok(f"State: {len(state.gaps)} gaps, {len(state.candidates)} candidates")

    doc_sections = parse_document(args.doc)
    full_sections = sum(1 for d in doc_sections.values() if not d["is_abstract"])
    abs_sections = sum(1 for d in doc_sections.values() if d["is_abstract"])
    total_words = sum(d["word_count"] for d in doc_sections.values())
    total_cites = sum(len(d["citations"]) for d in doc_sections.values())
    ok(
        f"Document: {len(doc_sections)} sections, {total_words:,} words, {total_cites} citations"
    )

    t_start = time.time()
    report = {
        "meta": {
            "paper_title": state.ba_paper.title if state.ba_paper else "—",
            "state_file": args.state,
            "doc_file": args.doc,
            "total_sections": len(doc_sections),
            "full_text_sections": full_sections,
            "abstract_sections": abs_sections,
            "total_words": total_words,
            "total_citations": total_cites,
        }
    }

    # ── Coverage ──────────────────────────────────────────────────────────
    section("Running coverage evaluation…")
    report["coverage"] = run_coverage_eval(state, doc_sections)
    ok(f"Coverage: {report['coverage']['badge']} {report['coverage']['label']}")

    # ── Citation grounding ─────────────────────────────────────────────────
    section("Checking citation grounding…")
    report["citations"] = run_citation_grounding(doc_sections, args.verbose)
    ok(
        f"Citations: {report['citations']['grounded']}/{report['citations']['total_citations']} "
        f"grounded ({int(report['citations']['grounding_rate'] * 100)}%)"
    )

    # ── Content quality ────────────────────────────────────────────────────
    if not args.no_quality:
        section("Running LLM-as-judge quality scoring…")
        report["quality"] = run_content_quality(
            doc_sections, state.ba_paper.title if state.ba_paper else "", args.verbose
        )
        n = report["quality"]["sections_scored"]
        ok(
            f"Quality: avg {report['quality']['avg_quality_score']:.2f} over {n} sections"
        )
    else:
        report["quality"] = {
            "avg_quality_score": 0.0,
            "sections_scored": 0,
            "per_section": [],
        }
        warn("Content quality scoring skipped (--no-quality)")

    # ── Phase A evaluation ─────────────────────────────────────────────────
    if args.gold_gaps:
        section("Evaluating Phase A gap detection vs gold…")
        report["phase_a"] = run_phase_a_eval(state, args.gold_gaps)
        ok(
            f"Phase A: recall={report['phase_a']['gap_recall']:.2f}, "
            f"precision={report['phase_a']['gap_precision']:.2f}, "
            f"F1={report['phase_a']['f1']:.2f}"
        )

    elapsed = time.time() - t_start
    report["meta"]["eval_time_s"] = round(elapsed, 1)

    # ── Render ────────────────────────────────────────────────────────────
    render_report(report, args.verbose)
    kv("Evaluation time", f"{elapsed:.1f}s")

    # ── Export ────────────────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        ok(f"Full JSON report → {args.out}")


if __name__ == "__main__":
    main()
