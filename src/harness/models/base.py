"""The interface every model backend implements.

One structurally typed Protocol, so the agent loop is written once against
``FakeProvider`` and run against ``LlamaCppProvider`` without a conditional
anywhere. Nothing here imports a concrete provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from harness.core import (
    Chunk,
    GenerationRequest,
    HealthStatus,
    ModelResponse,
    RuntimeModelInfo,
)


@runtime_checkable
class ModelProvider(Protocol):
    """A source of model completions.

    Implementations raise :class:`harness.core.ProviderError` subclasses on
    failure. ``health`` is the one exception: it reports unreachability as a
    value, because a probe that raises cannot be polled in a loop.
    """

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        """Run one request to completion."""
        ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk]:
        """Run one request, yielding deltas as they arrive.

        Declared non-``async`` so implementations can be plain async
        generators: callers write ``async for chunk in provider.stream(req)``
        rather than awaiting a coroutine that returns an iterator.
        """
        ...

    async def health(self) -> HealthStatus:
        """Probe the endpoint. Never raises for an unreachable server."""
        ...

    async def model_info(self) -> RuntimeModelInfo:
        """What the serving runtime reports about the loaded model."""
        ...
