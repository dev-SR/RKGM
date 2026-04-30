"""
Gap detection pipeline.

1. Run GAP_DETECTION prompt at 3 temperatures (self-consistency)
2. Keep gaps appearing in >= SELF_CONSISTENCY_MIN runs
3. Grounding check: each gap must trace back to the BA text
4. Return structured KnowledgeGap objects

Model routing: all calls use LLM_LIGHT (8B) for cost efficiency.
Gap detection is a structured extraction task, not creative generation.
"""

import hashlib
import json
from collections import Counter

from core.models import KnowledgeGap, GapType, Difficulty, Layer
from core.config import (
    SELF_CONSISTENCY_RUNS,
    SELF_CONSISTENCY_MIN,
    CONFIDENCE_THRESHOLD,
)
from core.prompts import (
    GAP_DETECTION_SYSTEM,
    GAP_DETECTION_USER,
    GROUNDING_CHECK_PROMPT,
)
from utils.llm import llm_heavy, llm_light, parse_json_response
from utils.cache import get_cached, set_cached


def detect_gaps(
    paper_text: str,
    abstract: str,
    user_gaps: list[str] | None = None,
    progress_callback=None,
) -> list[KnowledgeGap]:
    """
    Full gap detection pipeline.

    Args:
        paper_text:  Full text of the BA (trimmed to ~6000 words internally)
        abstract:    BA abstract (used for grounding check)
        user_gaps:   Optional list of concepts the user explicitly wants explained
        progress_callback: optional callable(str) for UI status updates

    Returns:
        List of validated KnowledgeGap objects sorted by confidence descending.
    """
    user_gaps = user_gaps or []
    user_gaps_str = ", ".join(user_gaps) if user_gaps else "none"

    # Trim paper text — abstract + intro + related work is most information-dense
    text_trimmed = _extract_key_sections(paper_text)

    cache_key = (
        "gaps:" + hashlib.md5((text_trimmed + user_gaps_str).encode()).hexdigest()
    )

    cached = get_cached(cache_key)
    if cached:
        return [_dict_to_gap(g) for g in cached]

    # ── Self-consistency: 3 runs at different temperatures ─────────────────
    temperatures = [0.1, 0.35, 0.6]
    all_runs: list[list[dict]] = []

    prompt_user = GAP_DETECTION_USER.format(
        paper_text=text_trimmed, user_gaps=user_gaps_str
    )

    for i, temp in enumerate(temperatures):
        if progress_callback:
            progress_callback(f"Gap detection run {i + 1}/{SELF_CONSISTENCY_RUNS}…")
        try:
            raw = llm_heavy(
                GAP_DETECTION_SYSTEM,
                prompt_user,
                temperature=temp,
                max_tokens=4096,  # raised: large papers produce many gaps, 2k cuts mid-array
            )
            run = _parse_gap_json(raw)
            if isinstance(run, list) and run:
                all_runs.append(run)
        except Exception as e:
            print(f"[gaps] Run {i + 1} failed: {e}")

    if not all_runs:
        return []

    # ── Self-consistency filtering ─────────────────────────────────────────
    concept_counts: Counter = Counter()
    concept_data: dict[str, dict] = {}

    for run in all_runs:
        for gap in run:
            concept = gap.get("concept", "").lower().strip()
            if concept:
                concept_counts[concept] += 1
                concept_data[concept] = gap  # last run wins for field values

    stable = [
        concept_data[c]
        for c, count in concept_counts.items()
        if count >= SELF_CONSISTENCY_MIN
    ]

    # Always include user-specified gaps even if they appear in only 1 run
    existing_concepts = {g.get("concept", "").lower() for g in stable}
    for run in all_runs:
        for gap in run:
            c = gap.get("concept", "").lower()
            if c and c not in existing_concepts:
                for ug in user_gaps:
                    if ug.lower() in c or c in ug.lower():
                        stable.append(gap)
                        existing_concepts.add(c)
                        break

    # ── Grounding + build KnowledgeGap objects ─────────────────────────────
    validated: list[KnowledgeGap] = []
    for i, g in enumerate(stable):
        conf = float(g.get("confidence", 0.5))
        grounded = _check_grounding(
            g.get("concept", ""),
            g.get("source_passage", ""),
            abstract,
            progress_callback,
        )

        if not grounded:
            if conf < CONFIDENCE_THRESHOLD:
                continue  # discard hallucinated low-confidence gaps
            conf = round(conf * 0.75, 3)  # penalise but keep flagged

        try:
            gap_obj = KnowledgeGap(
                gap_id=f"gap_{i:03d}",
                concept=g.get("concept", ""),
                gap_type=GapType(g.get("gap_type", "methodology")),
                difficulty=Difficulty(g.get("difficulty", "intermediate")),
                domain=g.get("domain", ""),
                why_needed=g.get("why_needed", ""),
                layer_hint=Layer(g.get("layer_hint", "development")),
                retrieval_query=g.get("retrieval_query", g.get("concept", "")),
                source_passage=g.get("source_passage", ""),
                confidence=conf,
            )
            validated.append(gap_obj)
        except (ValueError, KeyError) as e:
            print(f"[gaps] Skipping malformed gap '{g.get('concept')}': {e}")

    # Sort by confidence descending
    validated.sort(key=lambda x: x.confidence, reverse=True)

    # ── Cap historical gaps ────────────────────────────────────────────────
    # Historical gaps = author citation mentions. A paper with 40 citations
    # produces 40 historical gaps, drowning out the real concept gaps.
    # Keep only the top MAX_HISTORICAL_GAPS by confidence; prioritise all
    # other gap types first.
    MAX_HISTORICAL_GAPS = 5
    non_historical = [g for g in validated if g.gap_type != GapType.HISTORICAL]
    historical = [g for g in validated if g.gap_type == GapType.HISTORICAL]

    if len(historical) > MAX_HISTORICAL_GAPS:
        kept_historical = historical[:MAX_HISTORICAL_GAPS]
        dropped = len(historical) - MAX_HISTORICAL_GAPS
        print(
            f"[gaps] Capped historical gaps: kept {MAX_HISTORICAL_GAPS}, "
            f"dropped {dropped} low-confidence citation gaps"
        )
        validated = non_historical + kept_historical
        # Re-sort after merge
        validated.sort(key=lambda x: (x.gap_type == GapType.HISTORICAL, -x.confidence))

    set_cached(cache_key, [_gap_to_dict(g) for g in validated])
    return validated


def _parse_gap_json(raw: str) -> list[dict]:
    """
    Parse LLM gap JSON with partial-response recovery.

    The LLM sometimes truncates mid-array when max_tokens is hit.
    Strategy:
      1. Try standard parse first.
      2. If that fails, find the last complete {...} object before the
         truncation point and parse everything up to it.
      3. Return whatever valid gap objects were recovered.
    """
    import json
    import re as _re

    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Standard parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Partial recovery: find all complete {...} objects
    # Walk from end, find last position where a complete object closes
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict) and "concept" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if objects:
        recovered = len(objects)
        print(
            f"[gaps] Partial JSON recovery: salvaged {recovered} gap(s) "
            f"from truncated response"
        )
        return objects

    raise ValueError(f"JSON parse failed entirely.\nRaw[:300]: {raw[:300]}")


def _check_grounding(
    concept: str,
    source_passage: str,
    abstract: str,
    progress_callback=None,
) -> bool:
    """
    Ask LLM (light model) whether the gap traces back to the paper.
    Fail-open: if the call fails, assume grounded to avoid false discards.
    """
    if not source_passage or not concept:
        return False
    prompt = GROUNDING_CHECK_PROMPT.format(
        concept=concept,
        source_passage=source_passage[:300],
        abstract=abstract[:800],
    )
    try:
        raw = llm_light(prompt, max_tokens=200)
        data = parse_json_response(raw)
        return bool(data.get("grounded", True))
    except Exception:
        return True  # fail open


def _extract_key_sections(text: str, max_words: int = 5000) -> str:
    """
    Prefer abstract + introduction + related work sections.
    Falls back to first max_words words if sections not identified.
    """
    lower = text.lower()
    sections = []

    # Try to extract named sections
    markers = [
        ("abstract", "introduction"),
        ("introduction", "related"),
        ("related work", "method"),
        ("related work", "background"),
        ("background", "method"),
    ]
    for start_kw, end_kw in markers:
        si = lower.find(start_kw)
        ei = lower.find(end_kw, si + len(start_kw) + 10) if si >= 0 else -1
        if si >= 0 and ei > si:
            sections.append(text[si:ei])

    combined = " ".join(sections) if sections else text
    words = combined.split()
    return " ".join(words[:max_words])


def _gap_to_dict(g: KnowledgeGap) -> dict:
    return {
        "gap_id": g.gap_id,
        "concept": g.concept,
        "gap_type": g.gap_type.value,
        "difficulty": g.difficulty.value,
        "domain": g.domain,
        "why_needed": g.why_needed,
        "layer_hint": g.layer_hint.value,
        "retrieval_query": g.retrieval_query,
        "source_passage": g.source_passage,
        "confidence": g.confidence,
    }


def _dict_to_gap(d: dict) -> KnowledgeGap:
    return KnowledgeGap(
        gap_id=d["gap_id"],
        concept=d["concept"],
        gap_type=GapType(d["gap_type"]),
        difficulty=Difficulty(d["difficulty"]),
        domain=d["domain"],
        why_needed=d["why_needed"],
        layer_hint=Layer(d["layer_hint"]),
        retrieval_query=d["retrieval_query"],
        source_passage=d["source_passage"],
        confidence=d["confidence"],
    )
