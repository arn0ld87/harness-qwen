"""Scripted provider for tests.

Replays a fixed script instead of calling a model, and records what it was
asked. Two of its features exist for one reason each:

* ``fail_on_call`` makes retry and recovery paths reachable without a flaky
  server.
* ``prefix_hashes`` records the hash of the stable prompt prefix on every call,
  so a test can assert the prefix stayed byte-identical for a whole run. That
  property is the one the entire cache design rests on, and an assertion is the
  only thing that keeps it true.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

from harness.core import (
    Chunk,
    GenerationRequest,
    HealthStatus,
    Message,
    ModelResponse,
    ProviderError,
    RuntimeModelInfo,
    Usage,
)

_CHARS_PER_TOKEN = 4
"""Crude but deterministic: the point is a reproducible number for cache
arithmetic in tests, not an accurate tokenisation."""

_DEFAULT_CHUNK_CHARS = 16


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN) if text else 0


def _as_response(item: ModelResponse | str) -> ModelResponse:
    if isinstance(item, ModelResponse):
        return item
    return ModelResponse(content=item, finish_reason="stop")


def prefix_messages(messages: Sequence[Message]) -> list[Message]:
    """The leading run of system messages.

    The stable prefix is system instructions, tool schemas, repository map and
    task statement — everything the assembler puts before the first turn of
    conversation. In message terms that is exactly the leading system block.
    """
    out: list[Message] = []
    for message in messages:
        if message.role != "system":
            break
        out.append(message)
    return out


def prefix_hash(request: GenerationRequest, *, include_tools: bool = True) -> str:
    """Stable hash over everything that lands in the cached prompt prefix.

    Tool definitions are included because the chat template renders them into
    the prompt: registering a tool mid-run is a prefix change even though no
    message moved. ``json_schema`` is excluded — it constrains sampling and is
    never injected into the prompt.
    """
    payload: list[object] = [
        [m.role, m.name, m.tool_call_id, m.content] for m in prefix_messages(request.messages)
    ]
    if include_tools:
        payload.append(request.tools or [])
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FakeProvider:
    """Replays ``responses`` in order, recording every request it received."""

    def __init__(self, responses: Iterable[ModelResponse | str] | None = None, *,
                 errors: Mapping[int, BaseException] | None = None,
                 health_status: HealthStatus | None = None,
                 info: RuntimeModelInfo | None = None,
                 repeat_last: bool = False,
                 include_tools_in_prefix: bool = True,
                 synthesize_usage: bool = True,
                 chunk_chars: int = _DEFAULT_CHUNK_CHARS) -> None:
        self.responses: list[ModelResponse] = [_as_response(r) for r in (responses or [])]
        self.errors: dict[int, BaseException] = dict(errors or {})
        """Call number (1-based) -> exception to raise instead of answering."""
        self.health_status = health_status or HealthStatus(reachable=True)
        self.info = info or RuntimeModelInfo(model_id="fake", n_ctx=65536, total_slots=1)
        self.repeat_last = repeat_last
        self.include_tools_in_prefix = include_tools_in_prefix
        self.synthesize_usage = synthesize_usage
        self.chunk_chars = max(1, chunk_chars)

        self.requests: list[GenerationRequest] = []
        self.prefix_hashes: list[str] = []
        self.calls = 0
        """Counts every attempt, shared by ``generate`` and ``stream``, and
        incremented for failed calls too — a retry is a call."""

    def extend(self, *responses: ModelResponse | str) -> None:
        """Append to the script; lets a test stage a second phase mid-run."""
        self.responses.extend(_as_response(r) for r in responses)

    def fail_on_call(self, call_number: int, error: BaseException) -> None:
        """Raise ``error`` on the given 1-based call instead of answering."""
        self.errors[call_number] = error

    @property
    def prefix_stable(self) -> bool:
        """True when every call so far saw a byte-identical prefix."""
        return len(set(self.prefix_hashes)) <= 1

    @property
    def last_request(self) -> GenerationRequest:
        if not self.requests:
            raise AssertionError("FakeProvider received no requests")
        return self.requests[-1]

    def _record(self, request: GenerationRequest) -> str:
        self.calls += 1
        self.requests.append(request)
        digest = prefix_hash(request, include_tools=self.include_tools_in_prefix)
        self.prefix_hashes.append(digest)
        return digest

    def _next_response(self, request: GenerationRequest, digest: str) -> ModelResponse:
        """Consume one scripted response.

        A failing call raises before consuming, so the script position still
        lines up with the retry that follows it.
        """
        error = self.errors.get(self.calls)
        if error is not None:
            raise error

        index = self.calls - 1
        if index < len(self.responses):
            response = self.responses[index]
        elif self.repeat_last and self.responses:
            response = self.responses[-1]
        else:
            raise ProviderError(
                f"FakeProvider script exhausted: call {self.calls} of "
                f"{len(self.responses)} scripted responses"
            )
        return self._with_usage(response, request, digest)

    def _with_usage(self, response: ModelResponse, request: GenerationRequest,
                    digest: str) -> ModelResponse:
        """Fill in plausible token accounting when the script left it empty.

        ``cached_tokens`` is reported only when this call's prefix matched the
        previous one, which is what makes a prefix regression visible in a
        test the same way it is visible against the real server.
        """
        if not self.synthesize_usage or response.usage != Usage():
            return response
        prompt_tokens = sum(_estimate_tokens(m.content) for m in request.messages)
        repeated = len(self.prefix_hashes) > 1 and self.prefix_hashes[-2] == digest
        if request.prefix_token_estimate is not None:
            prefix_tokens = request.prefix_token_estimate
        else:
            prefix_tokens = sum(
                _estimate_tokens(m.content) for m in prefix_messages(request.messages)
            )
        return response.model_copy(update={"usage": Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=_estimate_tokens(response.content),
            cached_tokens=min(prefix_tokens, prompt_tokens) if repeated else 0,
        )})

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        digest = self._record(request)
        return self._next_response(request, digest)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk]:
        """Emit the scripted response as deltas.

        Content is split on a fixed character width so the concatenation of
        deltas is byte-identical to ``ModelResponse.content``.
        """
        digest = self._record(request)
        response = self._next_response(request, digest)
        for call in response.tool_calls:
            yield Chunk(tool_call_delta={
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            })
        text = response.content
        for start in range(0, len(text), self.chunk_chars):
            yield Chunk(delta=text[start:start + self.chunk_chars])
        yield Chunk(finish_reason=response.finish_reason or "stop")

    async def health(self) -> HealthStatus:
        return self.health_status

    async def model_info(self) -> RuntimeModelInfo:
        return self.info
