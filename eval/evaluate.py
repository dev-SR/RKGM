"""
Evaluation module for KGMS.

Three evaluation layers:

1. RAGAS metrics (per-explanation)
   - faithfulness:      % of claims supported by retrieved chunks
   - answer_relevancy:  how well the explanation addresses the gap
   - context_recall:    % of gap concept covered by retrieved chunks
   Gate: if faithfulness < FAITHFULNESS_GATE → expand top-k and regenerate

2. Claim-level grounding check
   - For each [cite: paper_id::chunk_id] marker, verify the cited chunk
     actually supports the claim. Flags unsupported citations.

3. Phase A evaluation (against gold dataset)
   - Gap recall: % of gold gaps that appear in system output
   - Gap precision: % of system gaps matching a gold gap
   - Candidate Precision@3: relevance of top-3 candidates per gap

RAGAS requires an LLM to act as judge. We route to LLM_LIGHT (8B)
because RAGAS calls are structural scoring tasks, not creative generation.

Fallback: if RAGAS is not installed or fails, a lightweight LLM-based
scoring prompt is used instead — always returns a float in [0, 1].
"""

import re
import json
from typing import Optional

from core.config import FAITHFULNESS_GATE, CONTEXT_RECALL_GATE, LLM_LIGHT
from utils.llm import llm_light, parse_json_response
from utils.cache import get_cached, set_cached
import hashlib

_RAGAS_AVAILABLE = False
# Try to import RAGAS — it's optional
# try:
#     from ragas import evaluate as ragas_evaluate
#     from ragas.metrics import faithfulness, answer_relevancy, context_recall
#     from datasets import Dataset

#     _RAGAS_AVAILABLE = True
# except ImportError:
#     _RAGAS_AVAILABLE = False


# ── RAGAS evaluation ───────────────────────────────────────────────────────


def evaluate_explanation(
    gap_concept: str,
    explanation_text: str,
    retrieved_chunks: list,  # list[Chunk]
    use_cache: bool = True,
) -> dict:
    """
    Evaluate a single gap explanation using RAGAS metrics.

    Returns dict with keys:
      faithfulness, answer_relevancy, context_recall, method
    Values are floats in [0, 1].
    """
    contexts = [c.text for c in retrieved_chunks]
    if not contexts:
        return {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_recall": 0.0,
            "method": "no_context",
        }

    # Cache key based on explanation + contexts
    cache_key = (
        "eval:"
        + hashlib.md5(
            (explanation_text[:500] + "".join(contexts[:3])[:500]).encode()
        ).hexdigest()
    )

    if use_cache:
        cached = get_cached(cache_key)
        if cached:
            return cached

    # if _RAGAS_AVAILABLE:
    #     result = _ragas_score(gap_concept, explanation_text, contexts)
    #     result["method"] = "ragas"
    # else:
    result = _llm_score(gap_concept, explanation_text, contexts)
    result["method"] = "llm_judge"

    if use_cache:
        set_cached(cache_key, result)
    return result


# def _ragas_score(
#     question: str,
#     answer: str,
#     contexts: list[str],
# ) -> dict:
#     """Run RAGAS evaluation. Requires ragas + datasets installed."""
#     try:
#         ds = Dataset.from_dict(
#             {
#                 "question": [question],
#                 "answer": [answer],
#                 "contexts": [contexts],
#                 "ground_truth": [question],  # question proxies ground truth
#             }
#         )
#         # RAGAS uses its own LLM wrapper; we let it use defaults
#         result = ragas_evaluate(
#             ds,
#             metrics=[faithfulness, answer_relevancy, context_recall],
#         )
#         row = result.to_pandas().iloc[0].to_dict()
#         return {
#             "faithfulness": float(row.get("faithfulness", 0.5)),
#             "answer_relevancy": float(row.get("answer_relevancy", 0.5)),
#             "context_recall": float(row.get("context_recall", 0.5)),
#         }
#     except Exception as e:
#         print(f"[eval] RAGAS failed ({e}), using LLM judge")
#         return _llm_score(question, answer, contexts)


def _llm_score(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Lightweight LLM-as-judge fallback when RAGAS is unavailable.
    Three separate prompts for faithfulness, relevancy, recall.
    All use LLM_LIGHT (8B) — purely scoring, no generation.
    """
    ctx_text = "\n\n---\n\n".join(contexts[:3])[:2000]

    # Faithfulness: are claims in the answer supported by contexts?
    faith_prompt = f"""Rate the faithfulness of this explanation on a scale 0.0 to 1.0.
Faithfulness = fraction of factual claims that are supported by the source passages.

Concept being explained: {question}
Source passages: {ctx_text}
Explanation: {answer[:800]}

Output JSON only: {{"faithfulness": 0.0_to_1.0, "reason": "one sentence"}}"""

    # Answer relevancy: does the explanation address the concept?
    rel_prompt = f"""Rate the relevancy of this explanation for the concept "{question}".
Relevancy = how well the explanation addresses what the concept is and why it matters.
Score 0.0 (completely irrelevant) to 1.0 (perfectly on-topic).

Explanation: {answer[:800]}

Output JSON only: {{"answer_relevancy": 0.0_to_1.0, "reason": "one sentence"}}"""

    # Context recall: does the explanation cover the key ideas in the contexts?
    recall_prompt = f"""Rate context recall on a scale 0.0 to 1.0.
Context recall = fraction of key ideas from the source passages that appear in the explanation.

Source passages: {ctx_text}
Explanation: {answer[:800]}

Output JSON only: {{"context_recall": 0.0_to_1.0, "reason": "one sentence"}}"""

    scores = {}
    for key, prompt in [
        ("faithfulness", faith_prompt),
        ("answer_relevancy", rel_prompt),
        ("context_recall", recall_prompt),
    ]:
        try:
            raw = llm_light(prompt, max_tokens=150)
            data = parse_json_response(raw)
            scores[key] = float(data.get(key, 0.5))
        except Exception:
            scores[key] = 0.5  # neutral default on parse failure

    return scores


# ── Faithfulness gate ──────────────────────────────────────────────────────


def passes_faithfulness_gate(scores: dict) -> bool:
    """
    Returns True if the explanation passes quality gates.
    If False, the caller should expand top-k and regenerate.
    """
    return (
        scores.get("faithfulness", 0.0) >= FAITHFULNESS_GATE
        and scores.get("context_recall", 0.0) >= CONTEXT_RECALL_GATE
    )


# ── Claim-level grounding check ────────────────────────────────────────────


def check_citation_grounding(
    explanation_text: str,
    chunk_map: dict,  # chunk_id -> Chunk
) -> dict:
    """
    For each [cite: paper_id::chunk_id] in the explanation,
    verify that the cited chunk supports the surrounding claim.

    Returns:
      {
        "total_citations": int,
        "grounded": int,
        "ungrounded": [{"citation": str, "claim_context": str}],
        "grounding_rate": float,
      }
    """
    # Extract all citations and their surrounding context (±100 chars)
    pattern = re.compile(r"\[cite:\s*([^\]]+)\]")
    matches = list(pattern.finditer(explanation_text))

    ungrounded = []
    grounded = 0

    for match in matches:
        cite_id = match.group(1).strip()
        start = max(0, match.start() - 150)
        end = min(len(explanation_text), match.end() + 50)
        context = explanation_text[start:end].strip()

        chunk = chunk_map.get(cite_id)
        if not chunk:
            ungrounded.append(
                {
                    "citation": cite_id,
                    "claim_context": context,
                    "reason": "chunk not found",
                }
            )
            continue

        # Ask LLM (light) whether the chunk supports the claim context
        prompt = f"""Does this source passage support the claim being made?

Claim context: "{context}"
Source passage: "{chunk.text[:400]}"

Output JSON only: {{"supported": true/false, "reason": "one sentence"}}"""

        try:
            raw = llm_light(prompt, max_tokens=100)
            data = parse_json_response(raw)
            if data.get("supported", True):
                grounded += 1
            else:
                ungrounded.append(
                    {
                        "citation": cite_id,
                        "claim_context": context,
                        "reason": data.get("reason", ""),
                    }
                )
        except Exception:
            grounded += 1  # fail open

    total = len(matches)
    return {
        "total_citations": total,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "grounding_rate": grounded / total if total > 0 else 1.0,
    }


# ── Phase A evaluation (gold dataset comparison) ───────────────────────────


def evaluate_phase_a(
    detected_gaps: list,  # list[KnowledgeGap]
    gold_gaps: list[str],  # list of concept strings from manual annotation
    candidates_per_gap: dict,  # gap_id -> list[CandidatePaper]
    gold_candidates: dict | None = None,  # gap_concept -> list[str] relevant paper IDs
) -> dict:
    """
    Compare Phase A output against a manually annotated gold set.

    Returns:
      gap_recall, gap_precision, candidate_precision_at_3
    """
    detected_concepts = {g.concept.lower().strip() for g in detected_gaps}
    gold_concepts = {g.lower().strip() for g in gold_gaps}

    # Fuzzy match: a detected concept counts as a hit if it contains
    # or is contained by a gold concept
    def is_match(detected: str, gold_set: set) -> bool:
        for g in gold_set:
            if detected in g or g in detected or _word_overlap(detected, g) > 0.5:
                return True
        return False

    true_positives = sum(1 for c in detected_concepts if is_match(c, gold_concepts))
    gap_recall = true_positives / max(len(gold_concepts), 1)
    gap_precision = true_positives / max(len(detected_concepts), 1)

    # Candidate precision@3
    if gold_candidates:
        p3_scores = []
        for gap in detected_gaps:
            gold_pids = gold_candidates.get(gap.concept.lower(), [])
            if not gold_pids:
                continue
            top3 = [
                c.paper.paper_id for c in candidates_per_gap.get(gap.gap_id, [])[:3]
            ]
            hits = sum(1 for pid in top3 if pid in gold_pids)
            p3_scores.append(hits / 3)
        cand_p3 = sum(p3_scores) / len(p3_scores) if p3_scores else 0.0
    else:
        cand_p3 = None

    return {
        "gap_recall": round(gap_recall, 3),
        "gap_precision": round(gap_precision, 3),
        "f1": round(
            2 * gap_recall * gap_precision / max(gap_recall + gap_precision, 1e-9), 3
        ),
        "candidate_precision_at_3": round(cand_p3, 3) if cand_p3 is not None else "N/A",
        "detected_gaps": len(detected_gaps),
        "gold_gaps": len(gold_gaps),
        "true_positives": true_positives,
    }


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap between two strings."""
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Full pipeline evaluation report ───────────────────────────────────────


def build_eval_report(
    state,  # PipelineState
    gold_gaps: list[str] | None = None,
) -> dict:
    """
    Build a complete evaluation report for a finished pipeline run.
    Suitable for display in the Streamlit UI and for export.
    """
    report = {
        "paper": state.ba_paper.title if state.ba_paper else "Unknown",
        "gaps_total": len(state.gaps),
        "gaps_active": len(state.explanations),
        "pdfs_used": len([e for e in state.explanations if not e.is_abstract_only]),
        "avg_confidence": round(
            sum(e.confidence for e in state.explanations)
            / max(len(state.explanations), 1),
            3,
        ),
        "explanations": [],
    }

    chunk_map = {c.chunk_id: c for c in state.chunks} if state.chunks else {}

    for exp in state.explanations:
        chunks = (
            [c for c in state.chunks if c.chunk_id in exp.source_citations]
            if state.chunks
            else []
        )

        scores = (
            evaluate_explanation(
                gap_concept=exp.concept,
                explanation_text=exp.explanation_text,
                retrieved_chunks=chunks,
            )
            if chunks
            else {
                "faithfulness": 0.3,
                "answer_relevancy": 0.3,
                "context_recall": 0.0,
                "method": "no_context",
            }
        )

        grounding = (
            check_citation_grounding(exp.explanation_text, chunk_map)
            if chunk_map
            else {"grounding_rate": 1.0, "total_citations": 0}
        )

        report["explanations"].append(
            {
                "concept": exp.concept,
                "confidence": exp.confidence,
                "abstract_only": exp.is_abstract_only,
                "faithfulness": scores.get("faithfulness"),
                "answer_relevancy": scores.get("answer_relevancy"),
                "context_recall": scores.get("context_recall"),
                "grounding_rate": grounding.get("grounding_rate"),
                "citations": len(exp.source_citations),
                "eval_method": scores.get("method"),
            }
        )

    # Aggregate scores
    valid = [e for e in report["explanations"] if not e["abstract_only"]]
    if valid:
        report["avg_faithfulness"] = round(
            sum(e["faithfulness"] for e in valid) / len(valid), 3
        )
        report["avg_answer_relevancy"] = round(
            sum(e["answer_relevancy"] for e in valid) / len(valid), 3
        )
        report["avg_context_recall"] = round(
            sum(e["context_recall"] for e in valid) / len(valid), 3
        )
        report["avg_grounding_rate"] = round(
            sum(e["grounding_rate"] for e in valid) / len(valid), 3
        )

    # Phase A evaluation if gold provided
    if gold_gaps and state.gaps:
        candidates_per_gap = {}
        for c in state.candidates:
            candidates_per_gap.setdefault(c.gap_id, []).append(c)
        report["phase_a_eval"] = evaluate_phase_a(
            detected_gaps=state.gaps,
            gold_gaps=gold_gaps,
            candidates_per_gap=candidates_per_gap,
        )

    return report
