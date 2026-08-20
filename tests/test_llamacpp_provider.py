import json

import httpx
import pytest

from harness.core import ContextOverflow, GenerationRequest, Message
from harness.models import LlamaCppProvider


@pytest.mark.asyncio
async def test_generate_maps_response_usage_and_cache_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["cache_prompt"] is True
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "timings": {"prompt_n": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LlamaCppProvider(client=client)
        response = await provider.generate(
            GenerationRequest(messages=[Message(role="user", content="test")])
        )

    assert response.content == "done"
    assert response.usage.cached_tokens == 7


@pytest.mark.asyncio
async def test_health_recognises_loading_model() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(503, json={"error": "Loading model"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        status = await LlamaCppProvider(client=client).health()

    assert status.reachable is True
    assert status.loading is True


@pytest.mark.asyncio
async def test_generate_maps_context_overflow() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            400,
            json={"error": {"message": "prompt exceeds the available context size"}},
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = LlamaCppProvider(client=client)
        with pytest.raises(ContextOverflow):
            await provider.generate(
                GenerationRequest(messages=[Message(role="user", content="test")])
            )
