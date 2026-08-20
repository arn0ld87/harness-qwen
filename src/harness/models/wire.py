"""Translation between harness types and llama-server's JSON wire format.

Pure functions, no I/O: every mapping decision — how a JSON Schema is wrapped,
how an error body becomes an exception class, how a cache hit is reconstructed
from the timings object — is testable without a transport. :mod:`llamacpp`
owns the HTTP, this module owns the meaning.
"""

from __future__ import annotations

import json
from typing import Any, Final

from harness.core import (
    Chunk,
    ContextOverflow,
    GenerationRequest,
    Message,
    ModelResponse,
    ProviderError,
    ProviderUnavailable,
    Timings,
    ToolCall,
    Usage,
)

# llama-server answers 503 with this message for roughly 40 s after start.
LOADING_MARKERS: Final = ("loading model", "model is loading")

# Substrings that identify a context-window failure. The status code alone is
# not enough: builds differ between 400, 500 and 413 for the same condition.
OVERFLOW_MARKERS: Final = (
    "exceed_context_size_error",
    "exceeds the available context size",
    "exceed the available context",
    "context size",
    "context window",
    "context shift is disabled",
    "prompt is too long",
    "n_ctx",
)

RAW_ARGUMENTS_KEY: Final = "_raw_arguments"
"""Where unparsable tool arguments are parked so the protocol layer can
report a parse error instead of the provider inventing an empty call."""


def as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _message_payload(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        out["name"] = message.name
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    return out


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a bare JSON Schema in the ``json_schema`` response format.

    A caller may pass either the schema itself or an already-wrapped object;
    double-wrapping produces a grammar that matches nothing.
    """
    if "schema" in schema and "type" not in schema:
        wrapper = dict(schema)
    else:
        wrapper = {"name": "response", "schema": schema, "strict": True}
    return {"type": "json_schema", "json_schema": wrapper}


def build_payload(request: GenerationRequest, *, model: str | None,
                  stream: bool) -> dict[str, Any]:
    """Translate a :class:`GenerationRequest` into a chat-completions body.

    Unset sampling parameters are omitted rather than sent as ``null`` so the
    server's own defaults — and the sampler defaults embedded in the GGUF —
    stay in force.
    """
    payload: dict[str, Any] = {
        "messages": [_message_payload(m) for m in request.messages],
        "stream": stream,
        # Explicit because the whole design rests on it; the default has
        # changed across llama.cpp builds and must not be assumed.
        "cache_prompt": True,
    }
    if model is not None:
        payload["model"] = model
    for field in ("max_tokens", "temperature", "top_p", "top_k", "min_p", "seed"):
        value = getattr(request, field)
        if value is not None:
            payload[field] = value
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.json_schema is not None:
        payload["response_format"] = _response_format(request.json_schema)
    if request.tools:
        payload["tools"] = request.tools
    return payload


def error_text(status: int, body: str) -> str:
    """Flatten an error body to searchable text, JSON or not."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body.strip()[:500]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            return f"{error.get('type', '')} {error.get('message', '')}".strip()
        if isinstance(error, str):
            return error
    return body.strip()[:500]


def classify_error(status: int, body: str) -> ProviderError:
    """Map an HTTP failure onto the exception the loop knows how to handle."""
    text = error_text(status, body)
    lowered = text.lower()
    if status == 413 or any(marker in lowered for marker in OVERFLOW_MARKERS):
        return ContextOverflow(f"context overflow ({status}): {text}")
    if status == 503 and any(marker in lowered for marker in LOADING_MARKERS):
        return ProviderUnavailable(f"model is still loading: {text}")
    if status >= 500 or status == 503:
        return ProviderUnavailable(f"llama-server unavailable ({status}): {text}")
    return ProviderError(f"llama-server rejected the request ({status}): {text}")


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if not isinstance(raw, list):
        return calls
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments or "{}")
            except (json.JSONDecodeError, ValueError):
                decoded = {RAW_ARGUMENTS_KEY: arguments}
        elif isinstance(arguments, dict):
            decoded = arguments
        else:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {RAW_ARGUMENTS_KEY: arguments}
        calls.append(ToolCall(
            id=str(item.get("id") or f"call_{index}"),
            name=str(function.get("name") or ""),
            arguments=decoded,
        ))
    return calls


def parse_usage(payload: dict[str, Any]) -> Usage:
    """Reconstruct token accounting, including how much came from cache.

    ``timings.prompt_n`` counts only the tokens the server actually processed,
    while ``usage.prompt_tokens`` counts the whole prompt. Their difference is
    the cache hit, which is what a prefix regression shows up in. Newer builds
    report it directly as ``timings.cache_n``; older ones do not, so it is
    derived when absent.
    """
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}

    prompt_tokens = as_int(usage.get("prompt_tokens"))
    completion_tokens = as_int(usage.get("completion_tokens"))
    processed = as_int(timings.get("prompt_n"))

    cached = as_int(timings.get("cache_n"))
    if cached is None:
        cached = as_int(timings.get("prompt_cache_n"))
    if cached is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = as_int(details.get("cached_tokens"))
    if cached is None and prompt_tokens is not None and processed is not None:
        cached = max(prompt_tokens - processed, 0)

    if prompt_tokens is None and processed is not None:
        prompt_tokens = processed + (cached or 0)
    if completion_tokens is None:
        completion_tokens = as_int(timings.get("predicted_n"))

    return Usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        cached_tokens=cached or 0,
    )


def parse_timings(payload: dict[str, Any]) -> Timings:
    raw = payload.get("timings")
    if not isinstance(raw, dict):
        return Timings()
    return Timings(
        prompt_ms=as_float(raw.get("prompt_ms")),
        predicted_ms=as_float(raw.get("predicted_ms")),
        prompt_per_second=as_float(raw.get("prompt_per_second")),
        predicted_per_second=as_float(raw.get("predicted_per_second")),
    )


def parse_response(payload: dict[str, Any]) -> ModelResponse:
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return ModelResponse(
        content=content if isinstance(content, str) else "",
        tool_calls=parse_tool_calls(message.get("tool_calls")),
        finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        usage=parse_usage(payload),
        timings=parse_timings(payload),
        reasoning=reasoning if isinstance(reasoning, str) else None,
    )


def chunks_from_event(data: dict[str, Any]) -> list[Chunk]:
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        return []
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    finish_reason = choice.get("finish_reason")
    out: list[Chunk] = []

    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        out.extend(Chunk(tool_call_delta=tc) for tc in tool_calls if isinstance(tc, dict))

    text = delta.get("content")
    if isinstance(text, str) and text:
        out.append(Chunk(delta=text))

    if finish_reason:
        out.append(Chunk(finish_reason=str(finish_reason)))
    elif not out and isinstance(text, str):
        out.append(Chunk())
    return out
