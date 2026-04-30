# ── Gap Detection ─────────────────────────────────────────────────────────

GAP_DETECTION_SYSTEM = """You are a research prerequisite analyst.
Given a research paper's text, identify every knowledge gap a reader must fill
before they can fully understand the paper.
Output ONLY a valid JSON array. No preamble, no markdown fences."""

GAP_DETECTION_USER = """Paper text:
<paper>
{paper_text}
</paper>

Identify knowledge gaps a reader needs to understand before reading this paper.
For each gap output a JSON object with EXACTLY these fields:
- "concept": short name of the concept (5 words max)
- "gap_type": one of ["terminology", "methodology", "benchmark", "historical", "mathematical"]
  * terminology  = field-specific jargon used without definition
  * methodology  = technique applied but not explained
  * benchmark    = dataset/benchmark mentioned without context
  * historical   = a prior work the paper BUILDS ON that the reader must understand
  * mathematical = equation steps that are skipped without derivation
- "difficulty": one of ["beginner", "intermediate", "advanced"]
- "domain": field of knowledge (e.g. "deep learning", "graph theory")
- "why_needed": one sentence — why the paper assumes this (be specific, not generic)
- "layer_hint": one of ["foundation", "development", "frontier"]
- "retrieval_query": an academic search query to find papers explaining this concept
- "source_passage": the exact phrase from the paper that triggered this gap (quote it)
- "confidence": your confidence this is a real gap, 0.0–1.0

CRITICAL RULES for historical gaps:
- ONLY include a historical gap if the paper RELIES ON that work's specific technique
  or finding — not just mentions it in passing
- Include AT MOST 5 historical gaps — pick only the most important cited works
- Do NOT include every citation in Related Work — only foundational dependencies
- Prefer terminology, methodology, and mathematical gaps over historical ones

User-specified gaps to always include: {user_gaps}

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

Already-explained concepts (do NOT include these): {known_concepts}

List ONLY technical terms this explanation uses WITHOUT defining, that a researcher
unfamiliar with the field would not understand without a separate explanation.

STRICT RULES — do NOT include:
- Author names or citations like "Brown et al.", "Smith et al.", any "et al." pattern
- Generic terms: "parameter", "model", "task", "method", "approach", "system", "data"
- Terms that are obvious from context or self-explanatory
- Anything already in the known concepts list above

Output a JSON array of strings only. Maximum 2 items. If none (or all filtered), output [].
Example: ["attention mechanism", "softmax function"]"""


# ── Document Preamble ──────────────────────────────────────────────────────

PREAMBLE_PROMPT = """You are writing the introduction to a personalised learning document.

Target paper: "{ba_title}"
Abstract: {ba_abstract}

The reader needs to understand these prerequisite concepts before reading the paper:
{gap_list}

Write a concise, engaging 2-paragraph introduction (plain text, no bullet points, no markdown):
Paragraph 1: What the target paper is about and why reading it requires background knowledge.
Paragraph 2: What this document covers and how the concepts connect to each other.

Be specific — reference actual concept names from the list above.
Write for a researcher, not a student. Keep it under 150 words total."""


# ── Historical gap section title ───────────────────────────────────────────

HISTORICAL_TITLE_PROMPT = """A research paper references this prior work without explaining it:

Citation: "{concept}"
Why it matters: {why_needed}
Source passage: "{source_passage}"

Write a short section title (6 words max) that describes WHAT THIS WORK CONTRIBUTED,
not who wrote it. Focus on the key technique, finding, or concept it introduced.

Examples:
  "Brown et al." → "Large-Scale Language Model Pre-training"
  "Vaswani et al." → "Transformer Self-Attention Architecture"
  "Kipf and Welling" → "Graph Convolutional Network Fundamentals"

Output only the title string, nothing else."""
