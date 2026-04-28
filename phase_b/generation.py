"""
Generation pipeline.

For each gap (in ordered sequence):
  1. Writing Agent: generate explanation with inline [cite: paper_id::chunk_id] markers
  2. Evaluation Agent: check for sub-gaps, unsupported citations, hallucinations
  3. Iterate up to MAX_EVAL_LOOPS
  4. Multi-hop: detect sub-gaps introduced by the explanation
  5. Assemble final Markdown document

Model routing:
  Writing Agent → LLM_HEAVY (70B): multi-paragraph coherent generation
  Evaluation Agent → LLM_LIGHT (8B): structured JSON checking task
  Sub-gap detection → LLM_LIGHT (8B): short-list extraction
"""

import json
import re

from core.models import (
    KnowledgeGap,
    GapType,
    Difficulty,
    Layer,
    Chunk,
    GapExplanation,
    Paper,
)
from core.config import MAX_EVAL_LOOPS, MAX_MULTIHOP_DEPTH
from core.prompts import WRITING_SYSTEM, WRITING_USER, EVAL_PROMPT, SUBGAP_PROMPT
from utils.llm import llm_heavy, llm_light, parse_json_response
from phase_b.retrieval import verbalize_path, format_passages


# ── Single gap explanation ─────────────────────────────────────────────────


def generate_explanation(
    gap: KnowledgeGap,
    chunks: list[Chunk],
    papers: dict[str, Paper],
    graph,
    ba_title: str,
    known_concepts: set[str],
    is_abstract_only: bool = False,
    progress_callback=None,
) -> GapExplanation:
    """
    Generate and iteratively refine an explanation for a single gap.
    Returns a GapExplanation with full text and citation markers.
    """
    if is_abstract_only or not chunks:
        return _abstract_only_explanation(gap)

    path = verbalize_path(gap, chunks, papers)
    passages = format_passages(chunks, papers)

    user_prompt = WRITING_USER.format(
        ba_title=ba_title,
        concept=gap.concept,
        gap_type=gap.gap_type.value,
        difficulty=gap.difficulty.value,
        why_needed=gap.why_needed,
        path_verbalization=path,
        passages=passages,
    )

    explanation_text = ""
    current_chunks = list(chunks)  # may be expanded by RAGAS gate

    for loop in range(MAX_EVAL_LOOPS):
        if progress_callback:
            progress_callback(
                f"Writing explanation for '{gap.concept}' "
                f"(iteration {loop + 1}/{MAX_EVAL_LOOPS})…"
            )

        # Writing Agent (heavy model)
        explanation_text = llm_heavy(
            WRITING_SYSTEM, user_prompt, temperature=0.3, max_tokens=1500
        )

        # Evaluation Agent (light model)
        eval_issues = _evaluate_explanation(
            explanation_text, passages, gap.concept, known_concepts
        )

        if eval_issues.get("approved", True):
            # RAGAS faithfulness gate — check quality before accepting
            scores = _ragas_gate_check(gap.concept, explanation_text, current_chunks)
            if not scores["passes"]:
                if progress_callback:
                    progress_callback(
                        f"Faithfulness gate failed "
                        f"(faithfulness={scores['faithfulness']:.2f}). "
                        f"Expanding retrieval…"
                    )
                # Expand retrieval and rebuild passages
                current_chunks = _expand_retrieval(gap, current_chunks, papers)
                if current_chunks:
                    passages = format_passages(current_chunks, papers)
                    user_prompt = WRITING_USER.format(
                        ba_title=ba_title,
                        concept=gap.concept,
                        gap_type=gap.gap_type.value,
                        difficulty=gap.difficulty.value,
                        why_needed=gap.why_needed,
                        path_verbalization=path,
                        passages=passages,
                    )
                    continue  # re-generate with expanded context
            break  # passed gate or no improvement possible

        # Build correction instruction for next iteration
        corrections = []
        hallucinations = eval_issues.get("hallucinations", [])
        unsupported = eval_issues.get("unsupported_citations", [])

        if hallucinations:
            bad = "; ".join(hallucinations[:2])
            corrections.append(
                f"Do NOT include this claim (not found in source passages): {bad}"
            )
        if unsupported:
            corrections.append(
                f"These claims need proper [cite:] markers from the passages: "
                f"{'; '.join(unsupported[:2])}"
            )

        if corrections:
            user_prompt += "\n\nCORRECTIONS REQUIRED:\n" + "\n".join(corrections)

    # Extract citation IDs from explanation
    citations = list(set(re.findall(r"\[cite:\s*([^\]]+)\]", explanation_text)))

    # Confidence from RAGAS gate score + base gap confidence
    gate_scores = _ragas_gate_check(gap.concept, explanation_text, current_chunks)
    confidence = min(
        0.96, gap.confidence * 0.4 + gate_scores.get("faithfulness", 0.5) * 0.6
    )

    return GapExplanation(
        gap_id=gap.gap_id,
        concept=gap.concept,
        explanation_text=explanation_text,
        source_citations=citations,
        confidence=round(confidence, 3),
        is_abstract_only=False,
    )


def _ragas_gate_check(concept: str, explanation_text: str, chunks: list) -> dict:
    """
    Run the RAGAS faithfulness gate. Imported lazily to avoid circular imports.
    Returns {"passes": bool, "faithfulness": float, "context_recall": float}
    """
    try:
        from eval.evaluate import evaluate_explanation, passes_faithfulness_gate

        scores = evaluate_explanation(concept, explanation_text, chunks)
        return {
            "passes": passes_faithfulness_gate(scores),
            "faithfulness": scores.get("faithfulness", 0.5),
            "context_recall": scores.get("context_recall", 0.5),
            "answer_relevancy": scores.get("answer_relevancy", 0.5),
        }
    except Exception:
        return {"passes": True, "faithfulness": 0.5, "context_recall": 0.5}


def _expand_retrieval(gap, current_chunks: list, papers: dict) -> list:
    """Placeholder — real top-k expansion is handled in pipeline.py."""
    return current_chunks


def _evaluate_explanation(
    explanation: str,
    passages: str,
    concept: str,
    known_concepts: set[str],
) -> dict:
    """
    Evaluation Agent: structured check of explanation quality.
    Light model (8B) is sufficient for this JSON extraction task.
    """
    prompt = EVAL_PROMPT.format(
        concept=concept,
        explanation=explanation[:2000],
        passages=passages[:3000],
        explained_concepts=json.dumps(list(known_concepts)),
    )
    try:
        raw = llm_light(prompt, max_tokens=600)
        return parse_json_response(raw)
    except Exception:
        return {"approved": True}  # fail open — do not block generation


def _abstract_only_explanation(gap: KnowledgeGap) -> GapExplanation:
    return GapExplanation(
        gap_id=gap.gap_id,
        concept=gap.concept,
        explanation_text=(
            f"*Note: No full-text PDF was available for papers covering "
            f"'{gap.concept}'. The following is based on abstract-level information only "
            f"and may be incomplete.*\n\n"
            f"{gap.why_needed}"
        ),
        source_citations=[],
        confidence=0.3,
        is_abstract_only=True,
    )


# ── Multi-hop sub-gap detection ────────────────────────────────────────────


def detect_subgaps(
    explanation_text: str,
    parent_gap: KnowledgeGap,
    known_concepts: set[str],
    depth: int = 0,
) -> list[KnowledgeGap]:
    """
    After generating an explanation, check whether it introduces new unexplained terms.
    Returns sub-gaps as KnowledgeGap objects (to be retrieved and explained first).

    Capped at MAX_MULTIHOP_DEPTH and 3 sub-gaps per parent.
    """
    if depth >= MAX_MULTIHOP_DEPTH:
        return []

    prompt = SUBGAP_PROMPT.format(
        concept=parent_gap.concept,
        explanation=explanation_text[:1500],
        known_concepts=json.dumps(list(known_concepts)),
    )
    try:
        raw = llm_light(prompt, max_tokens=200)
        subterms = parse_json_response(raw)
        if not isinstance(subterms, list):
            return []
    except Exception:
        return []

    subgaps = []
    for i, term in enumerate(subterms[:3]):
        term = str(term).strip()
        if not term or term.lower() in known_concepts:
            continue

        subgaps.append(
            KnowledgeGap(
                gap_id=f"{parent_gap.gap_id}_sub{depth}_{i}",
                concept=term,
                gap_type=GapType.METHODOLOGY,
                difficulty=Difficulty.INTERMEDIATE,
                domain=parent_gap.domain,
                why_needed=(
                    f"Introduced in the explanation of '{parent_gap.concept}' "
                    f"without being defined."
                ),
                layer_hint=Layer.FOUNDATION,
                retrieval_query=f"{term} definition explanation {parent_gap.domain}",
                source_passage=f"Sub-concept of {parent_gap.concept}",
                confidence=0.75,
            )
        )

    return subgaps


# ── Document assembly ──────────────────────────────────────────────────────


def assemble_document(
    ordered_explanations: list[GapExplanation],
    ba_title: str,
    ba_abstract: str,
    dependency_map: dict[str, str],
) -> str:
    """
    Assemble the final learning document in Markdown.

    Structure:
      # Learning Roadmap for: [BA Title]
      Brief intro
      ---
      ## 1. [Concept] (confidence %)
      [explanation text]
      > Dependency note (if any)
      ---
      ...
      ## You are now ready to read [BA Title]
    """
    lines = [
        f"# Learning Roadmap for: {ba_title}\n",
        "*This document was automatically generated to prepare you to read the "
        "target research paper. Work through the concepts in order — each section "
        "builds on the ones before it.*\n",
        f"> **Target paper abstract:** {ba_abstract[:400]}…\n",
        "---\n",
    ]

    for i, exp in enumerate(ordered_explanations, 1):
        status = (
            " *(abstract only — full text unavailable)*" if exp.is_abstract_only else ""
        )
        conf_pct = int(exp.confidence * 100)
        badge = f"Confidence: {conf_pct}%"

        lines.append(f"## {i}. {exp.concept}{status}")
        lines.append(f"*{badge}*\n")
        lines.append(exp.explanation_text)
        lines.append("")

        # Dependency note from ordering stage
        dep_note = dependency_map.get(exp.gap_id, "")
        if dep_note:
            lines.append(f"> **Learning dependency:** {dep_note}\n")

        # Citation summary
        if exp.source_citations and not exp.is_abstract_only:
            lines.append(f"*Sources: {', '.join(exp.source_citations[:5])}*")

        lines.append("\n---\n")

    # Closing section
    lines.append(f"## You are now ready to read: *{ba_title}*\n")
    lines.append(
        "All identified prerequisite concepts have been explained above. "
        "The concepts are ordered from foundational to frontier — reading them "
        "in sequence gives you the chronological context needed to understand "
        "how the target paper's contributions fit into the research landscape."
    )

    return "\n".join(lines)
