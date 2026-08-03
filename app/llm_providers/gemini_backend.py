"""
Gemini backend.

Two real shape differences from the OpenAI-style providers, handled here
so nothing above this layer needs to know about them: the system prompt is
a separate config parameter, not a message in the list, and Gemini uses
"model" as the assistant role name instead of "assistant".
"""

from __future__ import annotations

import os

from app.llm_providers.base import (
    LLMBackend,
    ProviderRateLimitError,
    ProviderTransientError,
    ProviderTruncationError,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


class GeminiBackend(LLMBackend):
    def __init__(self):
        from google import genai
        from google.genai import errors as genai_errors

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self._errors = genai_errors
        self._client = genai.Client(api_key=api_key)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                # Gemini folds multiple system messages together if there's
                # more than one; in practice llm_client.py only ever sends
                # one, but concatenating is a safe default either way.
                system_instruction = (
                    msg["content"]
                    if system_instruction is None
                    else f"{system_instruction}\n\n{msg['content']}"
                )
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        try:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                raise ProviderTruncationError(
                    f"Gemini output was truncated before JSON completed (max_tokens={max_tokens})"
                )
            return (response.text or "").strip()
        except self._errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise ProviderRateLimitError(str(exc), retry_after=_retry_delay(exc)) from exc
            raise
        except self._errors.ServerError as exc:
            raise ProviderTransientError(str(exc)) from exc


def _retry_delay(exc) -> float | None:
    """
    Gemini's 429 body includes a RetryInfo entry with a retryDelay like
    "13s" under error.details, when it's provided at all -- fall back to
    None (the shared backoff loop in llm_client.py has its own default)
    rather than guessing a number that isn't in the response.
    """
    details = getattr(exc, "details", None) or {}
    error = details.get("error", {}) if isinstance(details, dict) else {}
    for detail in error.get("details", []) if isinstance(error, dict) else []:
        if isinstance(detail, dict) and "retryDelay" in detail:
            raw = str(detail["retryDelay"]).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                continue
    return None
