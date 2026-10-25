"""Thin Groq chat client (OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime

import httpx

from papertrail.config import get_settings

URL = "https://api.groq.com/openai/v1/chat/completions"

# Guardrails state (per-process, resets daily)
_DAILY_CALLS: int = 0
_DAILY_SPEND_USD: float = 0.0
_LAST_RESET: date = date.min


def _ensure_reset() -> None:
    """Reset daily counters if a new day has started."""
    global _DAILY_CALLS, _DAILY_SPEND_USD, _LAST_RESET
    today = date.today()
    if today != _LAST_RESET:
        _DAILY_CALLS = 0
        _DAILY_SPEND_USD = 0.0
        _LAST_RESET = today


class GuardrailExceeded(RuntimeError):
    """Raised when a local usage cap is hit."""


def _check_guardrails(cost_usd: float) -> None:
    """Guardrails: daily call cap and spend limit."""
    _ensure_reset()
    settings = get_settings()
    if _DAILY_CALLS >= settings.daily_llm_call_cap:
        raise GuardrailExceeded(
            f"Daily LLM call cap reached ({settings.daily_llm_call_cap} calls)"
        )
    if _DAILY_SPEND_USD + cost_usd > settings.spend_limit_usd:
        raise GuardrailExceeded(
            f"Spend limit exceeded: ${_DAILY_SPEND_USD + cost_usd:.4f} > ${settings.spend_limit_usd:.2f}"
        )


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> str:
    settings = get_settings()
    model = model or settings.llm_model
    if not settings.groq_api_key or not model:
        raise RuntimeError("set GROQ_API_KEY and LLM_MODEL in .env")
    # Estimate cost per call (rough approximation)
    cost_per_call = settings.cost_per_llm_call_usd or 0.001
    _check_guardrails(cost_per_call)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": messages,
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"  # keeps calls fast and cheap
    for attempt in range(6):
        resp = httpx.post(
            URL, headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json=payload, timeout=90,
        )
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("retry-after", 5 * (attempt + 1))))
            continue
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        # Track the call and estimated cost
        global _DAILY_CALLS, _DAILY_SPEND_USD
        _DAILY_CALLS += 1
        _DAILY_SPEND_USD += cost_per_call
        return content
    raise RuntimeError("rate-limited too many times")
