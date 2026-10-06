"""Thin Groq chat client (OpenAI-compatible endpoint)."""

from __future__ import annotations

import time

import httpx

from papertrail.config import get_settings

URL = "https://api.groq.com/openai/v1/chat/completions"


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
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()
    raise RuntimeError("rate-limited too many times")
