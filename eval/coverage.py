"""
Coverage verification.

After Phase B generation, checks whether every active gap was actually
explained with meaningful content. Silently missed gaps are the most
user-unfriendly failure mode: the document looks complete but a whole
concept was skipped or produced only an abstract-only placeholder.

Three coverage checks:
  1. Completeness  — every active gap has an explanation entry
  2. Quality gate  — explanation is not just the abstract-only placeholder
  3. Citation gate — explanation has at least one [cite:] marker (for full-text gaps)

Output is a CoverageReport shown in the Streamlit UI and logged to the state.
"""

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class GapCoverageItem:
    gap_id: str
    concept: str
    status: str  # "covered" | "abstract_only" | "missing" | "uncited"
    confidence: float
    citations: int
    note: str = ""


@dataclass
class CoverageReport:
    total_gaps: int = 0
    covered: int = 0
    abstract_only: int = 0
    missing: int = 0
    uncited: int = 0
    coverage_score: float = 0.0  # 0–1, shown as % in UI
    items: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


def verify_coverage(
    gaps: list,  # list[KnowledgeGap]  — active gaps passed to Phase B
    explanations: list,  # list[GapExplanation] — what Phase B produced
) -> CoverageReport:
    """
    Compare the active gap list against generated explanations.

    Returns a CoverageReport with per-gap status and aggregate score.
    The coverage_score weights:
      covered (full-text, cited)  → 1.0 point
      abstract_only               → 0.4 point  (partial credit)
      uncited (no [cite:] markers)→ 0.6 point  (generated but ungrounded)
      missing                     → 0.0 points
    """
    report = CoverageReport(total_gaps=len(gaps))
    exp_map = {e.gap_id: e for e in explanations}

    total_score = 0.0

    for gap in gaps:
        exp = exp_map.get(gap.gap_id)

        if exp is None:
            # Gap has no explanation at all — silently dropped
            report.missing += 1
            report.items.append(
                GapCoverageItem(
                    gap_id=gap.gap_id,
                    concept=gap.concept,
                    status="missing",
                    confidence=0.0,
                    citations=0,
                    note="No explanation generated. Re-run Phase B or check retrieval.",
                )
            )
            total_score += 0.0

        elif exp.is_abstract_only:
            report.abstract_only += 1
            report.items.append(
                GapCoverageItem(
                    gap_id=gap.gap_id,
                    concept=gap.concept,
                    status="abstract_only",
                    confidence=exp.confidence,
                    citations=0,
                    note="No PDF found. Find a PDF for one of the candidate papers.",
                )
            )
            total_score += 0.4

        else:
            # Check for inline citations
            cites = re.findall(r"\[cite:\s*[^\]]+\]", exp.explanation_text)
            if not cites:
                report.uncited += 1
                report.items.append(
                    GapCoverageItem(
                        gap_id=gap.gap_id,
                        concept=gap.concept,
                        status="uncited",
                        confidence=exp.confidence,
                        citations=0,
                        note="Explanation generated but no [cite:] markers — "
                        "claims may be ungrounded. Consider re-generating.",
                    )
                )
                total_score += 0.6
            else:
                report.covered += 1
                report.items.append(
                    GapCoverageItem(
                        gap_id=gap.gap_id,
                        concept=gap.concept,
                        status="covered",
                        confidence=exp.confidence,
                        citations=len(cites),
                        note="",
                    )
                )
                total_score += 1.0

    report.coverage_score = round(total_score / max(report.total_gaps, 1), 3)

    # Recommendations
    if report.missing > 0:
        report.recommendations.append(
            f"{report.missing} gap(s) were silently skipped. "
            "Re-run Phase B with auto_fetch=True to retry."
        )
    if report.abstract_only > 0:
        report.recommendations.append(
            f"{report.abstract_only} gap(s) have abstract-only explanations. "
            "Manually upload PDFs for their candidate papers to improve quality."
        )
    if report.uncited > 0:
        report.recommendations.append(
            f"{report.uncited} explanation(s) contain no citation markers. "
            "These may include ungrounded claims — review carefully."
        )
    if report.coverage_score >= 0.9:
        report.recommendations.append(
            "Coverage is excellent. The learning document is well-sourced."
        )

    return report


def coverage_badge(score: float) -> str:
    """Return an emoji badge and label for the coverage score."""
    if score >= 0.9:
        return "🟢", f"Excellent ({int(score * 100)}%)"
    if score >= 0.7:
        return "🟡", f"Good ({int(score * 100)}%)"
    if score >= 0.5:
        return "🟠", f"Partial ({int(score * 100)}%)"
    return "🔴", f"Poor ({int(score * 100)}%)"
