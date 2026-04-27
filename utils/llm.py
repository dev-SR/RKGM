"""
LLM client with model routing.
Heavy model (70B) for generation and ordering.
Light model (8B) for all structural/cheap tasks.
"""

import time
import os
from groq import Groq
from core.config import LLM_HEAVY, LLM_LIGHT

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def _call(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    retries: int = 3,
) -> str:
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() and attempt < retries - 1:
                wait = 2 ** (attempt + 1)  # exponential backoff: 2s, 4s
                time.sleep(wait)
                continue
            raise
    return ""


def llm_light(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Cheap structural tasks: JSON extraction, grounding checks, eval agent."""
    return _call(
        LLM_LIGHT,
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def llm_heavy(
    system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048
) -> str:
    """Expensive reasoning: gap detection, writing agent, ordering."""
    return _call(
        LLM_HEAVY,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_json_response(raw: str) -> any:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
    import json

    cleaned = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}\nRaw: {raw[:300]}")
