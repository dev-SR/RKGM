"""
Chronological ordering of knowledge gaps.

Two-stage strategy:
  Stage 1 (deterministic): sort by layer_hint → trendscore of best candidate
    - Foundation → Development → Frontier
    - Within layer: higher trendscore candidate = explains earlier concept
    - Always produces a valid, reproducible ordering

  Stage 2 (LLM refinement): within-layer dependency detection
    - LLM identifies which gaps within the same layer have explicit dependencies
    - LLM produces dependency sentences for display in the final document
    - If LLM fails or returns cycles: Stage 1 ordering stands unchanged

  Cycle resolution: if LLM detects A↔B mutual dependency,
    break tie using trendscore (higher trendscore paper's gap comes first).
    This is deterministic and never blocks the pipeline.
"""

import json

from core.models import KnowledgeGap, Layer
from core.config import LLM_HEAVY
from core.prompts import ORDERING_SYSTEM, ORDERING_USER
from utils.llm import llm_heavy, parse_json_response

LAYER_ORDER = {Layer.FOUNDATION: 0, Layer.DEVELOPMENT: 1, Layer.FRONTIER: 2}


def order_gaps(
    gaps: list[KnowledgeGap],
    candidates: list,  # list[CandidatePaper]
    progress_callback=None,
) -> tuple[list[str], list[dict]]:
    """
    Determine the optimal learning order for a list of gaps.

    Returns:
        ordered_gap_ids  — gap IDs in learning order
        dependencies     — list of {"before", "after", "reason"} dicts
    """
    if not gaps:
        return [], []

    # Build a dict: gap_id → best candidate trendscore
    best_ts: dict[str, float] = {}
    for c in candidates:
        current = best_ts.get(c.gap_id, 0.0)
        best_ts[c.gap_id] = max(current, c.paper.trendscore)

    # ── Stage 1: deterministic sort ────────────────────────────────────────
    def sort_key(gap: KnowledgeGap):
        layer_rank = LAYER_ORDER.get(gap.layer_hint, 1)
        ts = best_ts.get(gap.gap_id, 0.0)
        diff_penalty = {"beginner": 0, "intermediate": 1, "advanced": 2}.get(
            gap.difficulty.value, 1
        )
        # Primary: layer, Secondary: trendscore desc (negate), Tertiary: difficulty
        return (layer_rank, -ts, diff_penalty)

    deterministic_order = sorted(gaps, key=sort_key)
    ordered_ids = [g.gap_id for g in deterministic_order]

    # ── Stage 2: LLM within-layer refinement ───────────────────────────────
    if progress_callback:
        progress_callback("Determining learning order…")

    gaps_json = json.dumps(
        [
            {
                "gap_id": g.gap_id,
                "concept": g.concept,
                "layer": g.layer_hint.value,
                "difficulty": g.difficulty.value,
                "why_needed": g.why_needed,
            }
            for g in deterministic_order
        ],
        indent=2,
        ensure_ascii=False,
    )

    dependencies: list[dict] = []

    try:
        raw = llm_heavy(
            ORDERING_SYSTEM,
            ORDERING_USER.format(gaps_json=gaps_json),
            temperature=0.1,
            max_tokens=1500,
        )
        data = parse_json_response(raw)

        llm_order = data.get("ordered_gap_ids", [])
        llm_deps = data.get("dependencies", [])
        cycles = data.get("cycles", [])

        # Validate LLM order contains all gap IDs
        known_ids = set(ordered_ids)
        valid_llm = [gid for gid in llm_order if gid in known_ids]

        # Only accept LLM order if it contains all gaps
        if len(valid_llm) == len(ordered_ids):
            ordered_ids = valid_llm
        # else: keep Stage 1 order silently

        dependencies = [
            d
            for d in llm_deps
            if d.get("before") in known_ids and d.get("after") in known_ids
        ]

        # Resolve cycles: break ties using trendscore
        if cycles:
            ordered_ids = _resolve_cycles(ordered_ids, cycles, best_ts, gaps)

    except Exception as e:
        print(f"[ordering] LLM ordering failed ({e}), using deterministic order")

    return ordered_ids, dependencies


def _resolve_cycles(
    ordered_ids: list[str],
    cycles: list[list[str]],
    best_ts: dict[str, float],
    gaps: list[KnowledgeGap],
) -> list[str]:
    """
    For each mutual dependency pair [A, B], ensure the one with higher
    trendscore (more foundational paper) comes first.
    Modifies the order list in-place and returns it.
    """
    id_to_pos = {gid: i for i, gid in enumerate(ordered_ids)}

    for pair in cycles:
        if len(pair) != 2:
            continue
        a, b = pair
        if a not in id_to_pos or b not in id_to_pos:
            continue

        ts_a = best_ts.get(a, 0.0)
        ts_b = best_ts.get(b, 0.0)
        pos_a, pos_b = id_to_pos[a], id_to_pos[b]

        # Higher trendscore → should come first
        if ts_a >= ts_b and pos_a > pos_b:
            # Swap
            ordered_ids[pos_a], ordered_ids[pos_b] = (
                ordered_ids[pos_b],
                ordered_ids[pos_a],
            )
            id_to_pos[a], id_to_pos[b] = pos_b, pos_a
        elif ts_b > ts_a and pos_b > pos_a:
            ordered_ids[pos_a], ordered_ids[pos_b] = (
                ordered_ids[pos_b],
                ordered_ids[pos_a],
            )
            id_to_pos[a], id_to_pos[b] = pos_b, pos_a

    return ordered_ids


def build_dependency_map(dependencies: list[dict]) -> dict[str, str]:
    """
    Build a dict: gap_id → dependency note to display in the document.
    E.g. gap_005 → "This concept builds on attention mechanisms (covered above)."
    """
    dep_map: dict[str, str] = {}
    for dep in dependencies:
        after = dep.get("after", "")
        reason = dep.get("reason", "")
        if after and reason:
            dep_map[after] = reason
    return dep_map
