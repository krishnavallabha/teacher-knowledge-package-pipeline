"""
Auto-detects which provider backend to use.

Priority when LLM_PROVIDER isn't set explicitly: check each provider's API
key env var in turn and use the first one found. 

If more than one key happens to be set, LLM_PROVIDER lets you pick
explicitly rather than silently picking by priority order.
"""

from __future__ import annotations

import os

from app.llm_providers.base import LLMBackend

_ENV_KEY_BY_PROVIDER = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Checked in this order when LLM_PROVIDER is not set explicitly.
_AUTO_DETECT_PRIORITY = ["groq", "gemini", "anthropic", "openai"]

_backend: LLMBackend | None = None


def get_provider() -> LLMBackend:
    global _backend
    if _backend is not None:
        return _backend

    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in _ENV_KEY_BY_PROVIDER:
            raise RuntimeError(
                f"LLM_PROVIDER={explicit!r} is not recognized. "
                f"Valid values: {', '.join(_ENV_KEY_BY_PROVIDER)}"
            )
        _backend = _build(explicit)
        return _backend

    for provider_name in _AUTO_DETECT_PRIORITY:
        if os.environ.get(_ENV_KEY_BY_PROVIDER[provider_name]):
            _backend = _build(provider_name)
            return _backend
    # Gemini also accepts GOOGLE_API_KEY as an alternate env var name.
    if os.environ.get("GOOGLE_API_KEY"):
        _backend = _build("gemini")
        return _backend

    raise RuntimeError(
        "No LLM provider API key found. Set one of: "
        + ", ".join(_ENV_KEY_BY_PROVIDER.values())
        + " (or GOOGLE_API_KEY for Gemini), or set LLM_PROVIDER explicitly."
    )


def _build(provider_name: str) -> LLMBackend:
    if provider_name == "groq":
        from app.llm_providers.groq_backend import GroqBackend
        return GroqBackend()
    if provider_name == "gemini":
        from app.llm_providers.gemini_backend import GeminiBackend
        return GeminiBackend()
    if provider_name == "anthropic":
        from app.llm_providers.anthropic_backend import AnthropicBackend
        return AnthropicBackend()
    if provider_name == "openai":
        from app.llm_providers.openai_backend import OpenAIBackend
        return OpenAIBackend()
    raise RuntimeError(f"Unknown provider: {provider_name}")
