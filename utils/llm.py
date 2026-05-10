"""
LLM client with model routing.
Heavy model (70B / GPT-4o) for generation and ordering.
Light model (8B / GPT-4o-mini) for all structural/cheap tasks.

Provider is resolved via core.config.ACTIVE_PROVIDER:
  - "groq"   → uses GROQ_API_KEY  (default / free tier)
  - "openai" → uses OPENAI_API_KEY
  - "auto"   → prefers Groq, falls back to OpenAI
"""

import re
import time
import os
from core.config import LLM_HEAVY, LLM_LIGHT, ACTIVE_PROVIDER


# ── Custom exceptions ──────────────────────────────────────────────────────


class RateLimitError(Exception):
    """Raised when the LLM API daily / quota limit is exhausted.

    Attributes:
        wait_time: Human-readable wait string parsed from the API response
                   (e.g. '1h1m6s'), or None if it could not be parsed.
        model:     The model that hit the limit.
        provider:  'groq' or 'openai'.
    """

    def __init__(
        self,
        message: str,
        wait_time: str | None = None,
        model: str = "",
        provider: str = "",
    ):
        super().__init__(message)
        self.wait_time = wait_time
        self.model = model
        self.provider = provider


def _parse_wait_time(err_str: str) -> str | None:
    """Extract the 'Please try again in Xh Ym Zs' wait duration from an error string."""
    match = re.search(r"Please try again in ([\dhms .]+)\.?", err_str)
    if match:
        return match.group(1).strip().rstrip(".")
    return None


# ── Groq client ────────────────────────────────────────────────────────────

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── OpenAI client ──────────────────────────────────────────────────────────

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Get a key at https://platform.openai.com/api-keys"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── Unified call ───────────────────────────────────────────────────────────


def _call(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    retries: int = 3,
) -> str:
    provider = ACTIVE_PROVIDER  # resolved once at import time

    for attempt in range(retries):
        try:
            if provider == "openai":
                client = _get_openai_client()
            else:
                client = _get_groq_client()

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "rate_limit" in err.lower()
                or "429" in err
                or "quota" in err.lower()
                or "insufficient_quota" in err.lower()
            )

            if is_rate_limit:
                # Daily / quota limit — no point retrying, raise immediately
                is_daily = (
                    "tokens per day" in err.lower()
                    or "tpd" in err.lower()
                    or "insufficient_quota" in err.lower()
                    or "billing" in err.lower()
                )
                if is_daily:
                    wait_time = _parse_wait_time(err)
                    provider_label = provider.capitalize()
                    raise RateLimitError(
                        f"{provider_label} quota limit reached for model '{model}'. "
                        + (
                            f"Please try again in {wait_time}."
                            if wait_time
                            else "Please check your billing / quota."
                        ),
                        wait_time=wait_time,
                        model=model,
                        provider=provider,
                    ) from e

                # Short-term rate limit (TPS/TPM) — exponential backoff
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    time.sleep(wait)
                    continue

            raise
    return ""


# ── Public helpers ─────────────────────────────────────────────────────────


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
