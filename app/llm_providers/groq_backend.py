"""Groq backend. OpenAI-compatible message shape, native JSON mode."""

from __future__ import annotations

import os
import re

from app.llm_providers.base import LLMBackend, ProviderRateLimitError, ProviderTransientError

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqBackend(LLMBackend):
    def __init__(self):
        import groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._groq = groq
        self._client = groq.Groq(
    api_key=api_key,
    timeout=60.0,
)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except self._groq.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), retry_after=_seconds_until_retry(exc)) from exc
        except self._groq.BadRequestError as exc:
            # Groq sometimes returns a 400 with the model's malformed JSON
            # attached in error.failed_generation rather than a clean
            # completion. Salvaging it and returning it as normal output
            # lets the shared JSON-parse/schema-validate/retry loop in
            # llm_client.py handle it the same way it handles any other
            # malformed response, rather than needing Groq-specific
            # recovery logic outside this file.
            salvaged = _salvage_failed_generation(exc)
            if salvaged is not None:
                return salvaged
            raise
        except (self._groq.APIConnectionError, self._groq.APITimeoutError, self._groq.InternalServerError) as exc:
            raise ProviderTransientError(str(exc)) from exc


def _seconds_until_retry(exc) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value:
            try:
                return float(header_value)
            except ValueError:
                pass
    message = str(exc)
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", message)
    if match:
        return float(match.group(1))
    return None


def _salvage_failed_generation(exc) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            failed_generation = error.get("failed_generation")
            if isinstance(failed_generation, str) and failed_generation.strip():
                return failed_generation

    message = str(exc)
    match = re.search(r"'failed_generation':\s*'(?P<json>.*)'\s*}\s*$", message, flags=re.S)
    if match:
        return match.group("json")
    return None
