"""OpenAI backend. Standard messages list (system included), native JSON mode."""

from __future__ import annotations

import os

from app.llm_providers.base import (
    LLMBackend,
    ProviderRateLimitError,
    ProviderTransientError,
    ProviderTruncationError,
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


class OpenAIBackend(LLMBackend):
    def __init__(self):
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ProviderTruncationError(
                    f"OpenAI output was truncated before JSON completed (max_tokens={max_tokens})"
                )
            return choice.message.content.strip()
        except self._openai.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), retry_after=_retry_after(exc)) from exc
        except (self._openai.APIConnectionError, self._openai.APITimeoutError, self._openai.InternalServerError) as exc:
            raise ProviderTransientError(str(exc)) from exc


def _retry_after(exc) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value:
            try:
                return float(header_value)
            except ValueError:
                pass
    return None
