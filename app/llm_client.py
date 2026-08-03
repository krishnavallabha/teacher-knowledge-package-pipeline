"""
Provider-agnostic structured generation.

This is the ONE place every pipeline stage calls into for an LLM
completion, and it has no idea which actual provider is behind it -- that
detail lives entirely in app/llm_providers/. The reason this separation
exists: the evaluators grading this project may run it against whichever
API key they have on hand (Groq, Gemini, Claude, or OpenAI), and a
pipeline hardcoded to one provider's SDK and error types only works for
whoever happens to hold that provider's key. Set any ONE of GROQ_API_KEY,
GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY and the pipeline
picks it up automatically -- see app/llm_providers/__init__.py for the
detection order, or set LLM_PROVIDER explicitly to force one.

Two independent retry loops live here, for different reasons:

1. Schema retry: the model returns JSON that doesn't match the target
   Pydantic model. Fixed by feeding the validation error back to the model
   as a follow-up message and asking for a corrected version.
2. Rate-limit retry: hitting a provider's rate limit is the normal case on
   a free tier, not an edge case -- this pipeline makes 10+ calls per
   document. This budget is deliberately separate from the schema-retry
   budget and honors whatever wait time the provider actually reports
   (ProviderRateLimitError.retry_after) rather than a blind fixed
   schedule -- retrying before the provider's own stated window has
   elapsed just burns another attempt on a request that was always going
   to fail again.
"""

import json
import re
from types import UnionType
from typing import get_args, get_origin, TypeVar, Union

import random
import time

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError

from app.llm_providers import get_provider
from app.llm_providers.base import ProviderRateLimitError, ProviderTransientError, ProviderTruncationError

load_dotenv(find_dotenv())

T = TypeVar("T", bound=BaseModel)


class LLMGenerationError(Exception):
    """Raised when the model fails to produce schema-valid output after retries."""


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    max_retries: int = 2,
    max_rate_limit_retries: int = 5,
    max_tokens: int = 6000,
) -> T:
    """
    Call the configured LLM provider, force JSON-only output, and validate
    against response_model. On a schema failure, feed the validation error
    back to the model and retry (up to max_retries). On a rate limit, back
    off for whatever the provider says to wait and retry the SAME request
    (up to max_rate_limit_retries), independent of the schema-retry budget.
    """
    schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
    print(f"Schema size: {len(schema_hint)} characters")
    print(f"User prompt size: {len(user_prompt)} characters")
    full_system = (
        f"{system_prompt}\n\n"
        "Respond with ONLY a single valid JSON object. No markdown fences, "
        "no preamble, no explanation. The JSON must conform to this schema:\n"
        f"{schema_hint}"
    )

    last_error = None
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_prompt},
    ]

    backend = get_provider()
    rate_limit_attempts = 0
    truncation_attempts = 0
    current_max_tokens = max_tokens
    # Ceiling on how far we'll escalate the token budget for one call.
    # Raised well past any single stage's normal need -- this is a safety
    # valve for unusually content-dense input (e.g. a long OCR'd chapter
    # assigning many concepts to one period), not a new default.
    max_token_ceiling = max(max_tokens * 3, 8192)

    for attempt in range(max_retries + 1):
        while True:
            try:
                
                print("=" * 80)
                print(f"Backend: {type(backend).__name__}")
                print(f"Max tokens: {current_max_tokens}")
                print("Sending request to provider...")
                raw_text = backend.complete(messages, current_max_tokens)
                print("Received response from provider.")
                print(f"Response length: {len(raw_text)}")
                print("=" * 80)
                print("FIRST 500")
                print(raw_text[:500])

                print("=" * 80)
                print("LAST 500")
                print(raw_text[-500:])
                raw_text = raw_text.strip()
                break
            except ProviderRateLimitError as exc:
                last_error = exc
                rate_limit_attempts += 1
                if rate_limit_attempts > max_rate_limit_retries:
                    raise LLMGenerationError(
                        f"Exhausted {max_rate_limit_retries} rate-limit retries: {exc}"
                    ) from exc
                time.sleep(_backoff_seconds(exc.retry_after))
                continue
            except ProviderTransientError as exc:
                last_error = exc
                rate_limit_attempts += 1  # shares the same retry budget as rate limits
                if rate_limit_attempts > max_rate_limit_retries:
                    raise LLMGenerationError(
                        f"Exhausted {max_rate_limit_retries} transient-error retries: {exc}"
                    ) from exc
                time.sleep(_backoff_seconds(None))
                continue
            except ProviderTruncationError as exc:
                # A truncated response is NOT a model mistake to correct --
                # it's a token budget that was too small for what this
                # specific request needed. Re-sending the same messages with
                # "fix your JSON" would just truncate again in the same
                # place, since the model still has the same amount of
                # content to say and the same ceiling to say it in. The
                # correct response is to raise the ceiling and retry the
                # SAME request, not to touch the conversation at all.
                last_error = exc
                truncation_attempts += 1
                if truncation_attempts > max_rate_limit_retries or current_max_tokens >= max_token_ceiling:
                    raise LLMGenerationError(
                        f"Output truncated even after raising max_tokens to {current_max_tokens}: {exc}"
                    ) from exc
                current_max_tokens = min(int(current_max_tokens * 1.5), max_token_ceiling)
                continue

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        try:
            if not raw_text.rstrip().endswith("}"):
                raise LLMGenerationError(
        "Claude output was truncated before JSON completed."
    )
            data = json.loads(raw_text)
            return _validate_response_data(data, response_model)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response did not parse as valid JSON matching "
                        f"the schema. Error: {e}\n"
                        "Return ONLY the corrected JSON object, nothing else."
                    ),
                }
            )

    raise LLMGenerationError(
        f"Failed to get schema-valid output after {max_retries + 1} attempts: {last_error}"
    )


def _backoff_seconds(retry_after: float | None) -> float:
    """
    Prefer the provider's own stated wait time when it gave one. The stated
    wait is a floor, not a guarantee, so add a small buffer and jitter
    rather than retrying at exactly the reported boundary.
    """
    if retry_after is not None:
        return retry_after + 0.5 + random.uniform(0, 0.5)
    return 5.0 + random.uniform(0, 0.5)


def _validate_response_data(data: object, response_model: type[T]) -> T:
    """
    Validate a structured-response payload, but also tolerate a simple
    single-wrapper object like {"activity": {...}} when the inner object
    already matches the requested schema. Smaller models sometimes add an
    extra envelope even after being instructed not to, and unwrapping that
    preserves robustness without relaxing the real schema check.
    """
    try:
        return response_model.model_validate(data)
    except ValidationError:
        normalized = _normalize_payload(data, response_model)
        if normalized is not data:
            return response_model.model_validate(normalized)
        if isinstance(data, dict) and len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, (dict, list)):
                return response_model.model_validate(inner)
        if isinstance(data, list):
            field_names = list(response_model.model_fields.keys())
            if len(field_names) == 1:
                field_info = response_model.model_fields[field_names[0]]
                if _annotation_accepts_list_of_strings(field_info.annotation):
                    data = [_coerce_to_text(item) for item in data]
                elif _annotation_accepts_string(field_info.annotation):
                    data = "\n".join(_coerce_to_text(item) for item in data)
                return response_model.model_validate({field_names[0]: data})
        raise


def _extract_model_type(annotation: object) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None
    if origin is list:
        args = get_args(annotation)
        if args:
            return _extract_model_type(args[0])
    if origin in (Union, UnionType):
        for arg in get_args(annotation):
            sub = _extract_model_type(arg)
            if sub is not None:
                return sub
    return None


def _normalize_payload(data: object, response_model: type[T]) -> object:
    """Normalize a payload enough to satisfy common model-shape drifts.

    We keep this intentionally conservative: only wrappers that clearly fit the
    target model are adjusted, rather than trying to guess every possible bad
    schema. Supports recursive sub-model normalization and alias mapping.
    """
    if not isinstance(data, dict):
        return data

    normalized = dict(data)

    aliases = {
        "question_text": ["question", "text"],
        "correct_answer": ["answer", "correct", "correctAnswer"],
        "question_type": ["type", "questionType"],
        "duration_minutes": ["duration", "minutes"],
        "learning_objectives": ["objectives"],
        "concepts_covered": ["concepts"],
        "classroom_activities": ["activities"],
        "checkpoint_questions": ["questions"],
        "materials_needed": ["materials"],
        "success_criteria": ["success"],
        "common_misconceptions": ["misconceptions"],
    }

    for field_name, alias_list in aliases.items():
        if field_name in response_model.model_fields and field_name not in normalized:
            for alias in alias_list:
                if alias in normalized:
                    normalized[field_name] = normalized[alias]
                    break

    for field_name, field_info in response_model.model_fields.items():
        value = normalized.get(field_name)
        if value is None:
            continue

        annotation = field_info.annotation

        if isinstance(value, list) and _annotation_accepts_string(annotation):
            normalized[field_name] = "\n".join(_coerce_to_text(item) for item in value)
            continue

        if isinstance(value, list) and _annotation_accepts_list_of_strings(annotation):
            normalized[field_name] = [_coerce_to_text(item) for item in value]
            continue

        sub_model = _extract_model_type(annotation)
        if sub_model is not None and issubclass(sub_model, BaseModel):
            if isinstance(value, dict):
                normalized[field_name] = _normalize_payload(value, sub_model)
            elif isinstance(value, list):
                normalized[field_name] = [
                    _normalize_payload(item, sub_model) if isinstance(item, dict) else item
                    for item in value
                ]

    return normalized


def _annotation_accepts_list_of_strings(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin in (list,):
        args = get_args(annotation)
        return not args or args[0] is str
    if origin in (UnionType,):
        return any(_annotation_accepts_list_of_strings(arg) for arg in get_args(annotation))
    return False


def _annotation_accepts_string(annotation: object) -> bool:
    origin = get_origin(annotation)
    if annotation is str:
        return True
    if origin in (UnionType,):
        return any(_annotation_accepts_string(arg) for arg in get_args(annotation))
    return False


def _coerce_to_text(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("description", "claim", "text", "title", "name", "value"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in item.values():
            if isinstance(value, str) and value.strip():
                return value
    return str(item)
