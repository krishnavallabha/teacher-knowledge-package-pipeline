

from __future__ import annotations

import os

from app.llm_providers.base import (
    LLMBackend,
    ProviderRateLimitError,
    ProviderTransientError,
    ProviderTruncationError,
)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


class AnthropicBackend(LLMBackend):
    def __init__(self):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=60.0,)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
# having error handling here is important because Claude can return a 429 or 500 if the request is too long or the server is busy, and we want to surface that to the user rather than crashing the whole app.
        try:
            print("Sending request to Claude...")
            print("=" * 80)
            print("Calling Claude")
            print(f"Model: {MODEL}")
            print(f"Messages: {len(conversation)}")
            print(f"Max tokens: {max_tokens}")
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system="\n\n".join(system_parts) if system_parts else None,
                messages=conversation,

            )
            print("=" * 80)
            print("STOP REASON:", response.stop_reason)
            print("USAGE:", response.usage)
            print("Claude returned successfully.")
            if response.stop_reason == "max_tokens":
                raise ProviderTruncationError(
                    f"Claude output was truncated before JSON completed (max_tokens={max_tokens})"
                )
            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            ).strip()

            print("=" * 80)
            print("LAST 500 CHARACTERS")
            print(text[-500:])
            return text
        except self._anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), retry_after=_retry_after(exc)) from exc
        except (self._anthropic.APIConnectionError, self._anthropic.APITimeoutError, self._anthropic.InternalServerError) as exc:
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
