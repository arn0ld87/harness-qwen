import asyncio

import pytest

from harness.core import Risk, ToolResult, ToolSpec
from harness.tools import ToolRegistry


def _spec(*, timeout_s: float = 1.0) -> ToolSpec:
    return ToolSpec(
        name="example",
        description="Example tool",
        risk=Risk.ALLOW,
        timeout_s=timeout_s,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": 1}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )


@pytest.mark.asyncio
async def test_registry_validates_and_invokes_async_tool() -> None:
    async def tool(value: int) -> ToolResult:
        return ToolResult(tool="ignored", ok=True, content=str(value))

    registry = ToolRegistry()
    registry.register(_spec(), tool)

    invalid = await registry.invoke("example", {"value": 0, "extra": True})
    valid = await registry.invoke("example", {"value": 2})

    assert invalid.error_kind == "invalid_arguments"
    assert "unexpected argument" in invalid.content
    assert valid.ok is True
    assert valid.tool == "example"


@pytest.mark.asyncio
async def test_registry_reports_unknown_tool_and_timeout() -> None:
    async def slow(value: int) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(tool="example", ok=True, content=str(value))

    registry = ToolRegistry()
    registry.register(_spec(timeout_s=0.001), slow)

    missing = await registry.invoke("missing", {})
    timed_out = await registry.invoke("example", {"value": 1})

    assert missing.error_kind == "not_found"
    assert timed_out.error_kind == "timeout"


@pytest.mark.asyncio
async def test_registry_never_executes_denied_tool() -> None:
    called = False

    async def denied(value: int) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(tool="example", ok=True, content=str(value))

    registry = ToolRegistry()
    registry.register(_spec().model_copy(update={"risk": Risk.DENY}), denied)

    result = await registry.invoke("example", {"value": 1})

    assert result.error_kind == "denied"
    assert called is False
