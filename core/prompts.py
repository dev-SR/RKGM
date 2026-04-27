# ── Gap Detection ─────────────────────────────────────────────────────────

GAP_DETECTION_SYSTEM = """You are a research prerequisite analyst.
Given a research paper's text, identify every knowledge gap a reader must fill
before they can fully understand the paper.
Output ONLY a valid JSON array. No preamble, no markdown fences."""

GAP_DETECTION_USER = """Paper text:
<paper>
{paper_text}
</paper>

Identify knowledge gaps. For each gap output a JSON object with EXACTLY these fields:
- "concept": short name of the concept (5 words max)
- "gap_type": one of ["terminology", "methodology", "benchmark", "historical", "mathematical"]
  * terminology  = field-specific jargon used without definition
  * methodology  = technique applied but not explained
  * benchmark    = dataset/benchmark mentioned without context
  * historical   = "as shown in [X]" without explaining what X actually showed
  * mathematical = equation steps that are skipped without derivation
- "difficulty": one of ["beginner", "intermediate", "advanced"]
- "domain": field of knowledge (e.g. "deep learning", "graph theory")
- "why_needed": one sentence — why the paper assumes this (be specific to the paper)
- "layer_hint": one of ["foundation", "development", "frontier"]
  * foundation = classical concept, explained in seminal highly-cited papers
  * development = established but more recent technique
  * frontier    = cutting-edge, only in very recent papers
- "retrieval_query": an academic search query to find papers explaining this concept
- "source_passage": the exact phrase from the paper that triggered this gap (quote it)
- "confidence": your confidence this is a real gap, 0.0–1.0

IMPORTANT: Also mine the Related Work section — every citation like "as shown in [X]"
or "following [X]" where the BA does not re-explain what X did is a historical gap.

User-specified gaps to include (always include these even if not in the text): {user_gaps}

Output a JSON array only. No preamble."""


# ── Grounding Check ────────────────────────────────────────────────────────

GROUNDING_CHECK_PROMPT = """Does this knowledge gap genuinely trace back to the paper?

Gap concept: {concept}
Claimed source passage: "{source_passage}"
Paper abstract: {abstract}

Answer with JSON only: {{"grounded": true/false, "reason": "one sentence"}}"""


# ── Candidate Rationale ────────────────────────────────────────────────────

CANDIDATE_RATIONALE_PROMPT = """Knowledge gap: {concept} ({gap_type})
Why the reader needs it: {why_needed}

Candidate paper:
Title: {title}
Abstract: {abstract}

In exactly one sentence, explain specifically how this paper fills the gap above.
Be concrete — do not just restate the abstract. Do not start with "This paper"."""


# ── Cluster Summary ────────────────────────────────────────────────────────

CLUSTER_SUMMARY_PROMPT = """These papers all relate to the knowledge gap: "{gap_concept}"

Papers:
{papers_json}

In 2 sentences:
1. What shared methodology or theme unifies these papers?
2. How do they collectively address the gap?
Be specific. Do not list paper titles."""


# ── Chronological Ordering ─────────────────────────────────────────────────

ORDERING_SYSTEM = (
    "You are a curriculum designer specialising in research paper prerequisites."
)

ORDERING_USER = """Order these knowledge gaps for a reader who needs to learn them sequentially.

Gaps:
{gaps_json}

Rules:
1. Foundation-layer gaps always come before Development, which comes before Frontier.
2. Within the same layer, if understanding gap A is NECESSARY before understanding gap B,
   place A first and record the dependency.
3. If you detect a mutual dependency (A needs B AND B needs A), output both in the
   "cycles" array and do not add them to dependencies.

Output JSON only:
{{
  "ordered_gap_ids": ["id1", "id2", ...],
  "dependencies": [
    {{"before": "gap_id", "after": "gap_id",
      "reason": "One sentence: gap B assumes knowledge introduced in gap A."}}
  ],
  "cycles": [["gap_id_a", "gap_id_b"]]
}}"""


# ── Writing Agent ──────────────────────────────────────────────────────────

WRITING_SYSTEM = """You are an expert research educator writing a prerequisite learning document.
Rules:
1. Explain the concept clearly for a researcher who has NOT read the source papers.
2. After EVERY factual claim, add a citation: [cite: paper_id::chunk_id]
3. Use only information present in the source passages — do not add outside knowledge.
4. Structure: (a) what the concept IS, (b) how it works, (c) why it matters for the target paper.
5. End with: "This concept is needed in [target paper] because [specific reason]." """

WRITING_USER = """Target paper title: {ba_title}

Concept to explain: {concept}
Gap type: {gap_type}
Difficulty: {difficulty}
Why the reader needs this: {why_needed}

Reference chain (structural context):
{path_verbalization}

Source passages:
{passages}

Write the explanation now (2–4 paragraphs). Remember: cite every claim with [cite: paper_id::chunk_id]."""


# ── Evaluation Agent ───────────────────────────────────────────────────────

EVAL_PROMPT = """Review this explanation of "{concept}":

<explanation>
{explanation}
</explanation>

Available source passages:
{passages}

Check these three things:
1. Does the explanation use any technical term it does NOT define, and which is
   NOT already in the explained_concepts list: {explained_concepts}?
   List as sub_gaps.
2. Are any [cite: ...] markers referencing passages that do NOT support the claim?
   List as unsupported_citations.
3. Does the explanation state anything NOT found in ANY source passage?
   List as hallucinations.

Output JSON only:
{{
  "sub_gaps": ["term1"],
  "unsupported_citations": ["claim text"],
  "hallucinations": ["claim text"],
  "approved": true/false
}}"""


# ── Sub-gap Detection ──────────────────────────────────────────────────────

SUBGAP_PROMPT = """This explanation was just generated for the concept "{concept}":

<explanation>
{explanation}
</explanation>

List any technical terms this explanation uses WITHOUT defining,
where those terms are NOT in this already-explained set: {known_concepts}

Output a JSON array of strings only. Maximum 3 items. If none, output [].
Example: ["attention mechanism", "softmax function"]"""
