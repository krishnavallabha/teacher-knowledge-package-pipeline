"""
Provider-agnostic backend interface.


Every backend implements ONE method: complete(messages, max_tokens) -> str,
returning raw text output, with normalized rate-limit and transient-error
exceptions. All provider-specific details -- message format quirks (system
prompt as a separate parameter vs. a message in the list, "assistant" vs
"model" as the role name), JSON-mode flag names, and each SDK's own
exception classes -- are absorbed inside the backend that owns them. The
shared retry loop, schema-validation loop, and backoff logic in
app/llm_client.py are written ONCE against this interface and never import
a provider SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderRateLimitError(Exception):
    """Normalized 429 / rate-limit signal from any provider."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTransientError(Exception):
    """Normalized connection/timeout/5xx signal -- worth a retry, not a hard failure."""


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """
        messages is a list of {"role": "system"|"user"|"assistant", "content": str}
        in that generic shape regardless of provider. Returns raw text content.
        Must request JSON-only output from the underlying API where the
        provider supports a native flag for it (this is a second layer of
        defense on top of the prompt instruction the caller already
        includes, not a replacement for it).

        Raises ProviderRateLimitError or ProviderTransientError on the
        corresponding failure classes; any other exception is treated by
        the caller as unrecoverable for that attempt.
        """
        raise NotImplementedError
