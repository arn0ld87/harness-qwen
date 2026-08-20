"""HTTP client for a running ``llama-server``.

Owns the connection, the timeouts and the error surface; the JSON mapping
lives in :mod:`harness.models.wire`. An injected client is never closed here —
its owner keeps that responsibility, which is what makes
``httpx.MockTransport`` usable in tests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Final

import httpx

from harness.core import (
    Chunk,
    GenerationRequest,
    HealthStatus,
    ModelResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RuntimeModelInfo,
    Timings,
    Usage,
)
from harness.models.wire import (
    LOADING_MARKERS,
    as_int,
    build_payload,
    chunks_from_event,
    classify_error,
    error_text,
    parse_response,
    parse_timings,
    parse_usage,
)

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 18080
DEFAULT_BASE_URL: Final = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
"""Derived, not typed out twice: the config layer needs the host and the port
separately to launch a server, and a second literal would drift from this one
the first time either is changed."""


class LlamaCppProvider:
    """Talks to one ``llama-server`` instance over HTTP.

    An injected client is never closed by this class — its owner keeps that
    responsibility, which is what makes ``httpx.MockTransport`` usable in
    tests.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, model: str | None = None,
                 api_key: str | None = None, client: httpx.AsyncClient | None = None,
                 connect_timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._connect_timeout = connect_timeout
        self.last_stream_usage: Usage | None = None
        """Token accounting from the final streamed event, when the server
        sent one. Streaming would otherwise discard the cache counters."""
        self.last_stream_timings: Timings | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        return self._client

    def _timeout(self, total: float) -> httpx.Timeout:
        return httpx.Timeout(total, connect=min(self._connect_timeout, total))

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LlamaCppProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        payload = build_payload(request, model=self.model, stream=False)
        try:
            response = await self._http().post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(request.timeout_s),
            )
        except httpx.ConnectTimeout as exc:
            raise ProviderUnavailable(f"connect timeout to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"request exceeded {request.timeout_s}s: {exc}") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderUnavailable(f"cannot reach {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise classify_error(response.status_code, response.text)
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"malformed response body: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderError("malformed response body: expected a JSON object")
        return parse_response(data)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk]:
        """Yield deltas from the SSE stream.

        Blank keepalive lines, comment lines and the terminal ``[DONE]``
        sentinel are not events and are skipped; a single unparsable event is
        skipped rather than killing a run that is otherwise fine.
        """
        payload = build_payload(request, model=self.model, stream=True)
        self.last_stream_usage = None
        self.last_stream_timings = None
        try:
            async with self._http().stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(request.timeout_s),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise classify_error(response.status_code, body)
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith(":"):
                        continue
                    if not stripped.startswith("data:"):
                        continue
                    data = stripped[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    if "timings" in event or "usage" in event:
                        self.last_stream_usage = parse_usage(event)
                        self.last_stream_timings = parse_timings(event)
                    for chunk in chunks_from_event(event):
                        yield chunk
        except httpx.ConnectTimeout as exc:
            raise ProviderUnavailable(f"connect timeout to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"stream exceeded {request.timeout_s}s: {exc}") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderUnavailable(f"cannot reach {self.base_url}: {exc}") from exc

    async def health(self) -> HealthStatus:
        try:
            response = await self._http().get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self._timeout(self._connect_timeout),
            )
        except (httpx.HTTPError, OSError) as exc:
            return HealthStatus(reachable=False, detail=str(exc) or type(exc).__name__)

        if response.status_code == 200:
            return HealthStatus(reachable=True, detail=error_text(200, response.text) or None)
        detail = error_text(response.status_code, response.text)
        if response.status_code == 503:
            loading = any(marker in detail.lower() for marker in LOADING_MARKERS)
            return HealthStatus(reachable=True, loading=loading, detail=detail)
        return HealthStatus(reachable=True, detail=f"HTTP {response.status_code}: {detail}")

    async def model_info(self) -> RuntimeModelInfo:
        info = RuntimeModelInfo()
        try:
            props = await self._http().get(
                f"{self.base_url}/props",
                headers=self._headers(),
                timeout=self._timeout(self._connect_timeout),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderUnavailable(f"cannot reach {self.base_url}: {exc}") from exc

        if props.status_code != 200:
            raise classify_error(props.status_code, props.text)
        try:
            data = props.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"malformed /props body: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderError("malformed /props body: expected a JSON object")

        info.model_path = data.get("model_path")
        info.build_info = data.get("build_info")
        info.total_slots = as_int(data.get("total_slots"))
        caps = data.get("chat_template_caps")
        if isinstance(caps, dict):
            info.chat_template_caps = {k: v for k, v in caps.items() if isinstance(v, bool)}
        generation = data.get("default_generation_settings")
        if isinstance(generation, dict):
            info.n_ctx = as_int(generation.get("n_ctx"))
            model = generation.get("model")
            if isinstance(model, str):
                info.model_id = model
        if info.model_id is None:
            info.model_id = self.model or await self._first_model_id()
        return info

    async def _first_model_id(self) -> str | None:
        """Fall back to ``/v1/models``; older builds omit the id from /props."""
        try:
            response = await self._http().get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=self._timeout(self._connect_timeout),
            )
            data = response.json() if response.status_code == 200 else {}
        except (httpx.HTTPError, OSError, json.JSONDecodeError, ValueError):
            return None
        entries = data.get("data") if isinstance(data, dict) else None
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            model_id = entries[0].get("id")
            return model_id if isinstance(model_id, str) else None
        return None
