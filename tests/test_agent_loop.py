import json
from pathlib import Path

import pytest

from harness.agent import AgentLoop
from harness.context import PromptAssembler, TokenBudget
from harness.core import Budget, Risk, StopReason, ToolResult, ToolSpec
from harness.memory import MemoryStore
from harness.models import FakeProvider
from harness.protocol.constrained import ConstrainedJsonCodec
from harness.telemetry import RunJournal
from harness.tools import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "model_responses" / "agent_loop.json"


def _responses(name: str) -> list[str]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))[name]
    return [item if isinstance(item, str) else json.dumps(item) for item in raw]


def _tools(workspace: Path) -> ToolRegistry:
    async def read_file(path: str) -> ToolResult:
        return ToolResult(
            tool="read_file",
            ok=True,
            content=(workspace / path).read_text(encoding="utf-8"),
        )

    async def write_file(path: str, content: str) -> ToolResult:
        (workspace / path).write_text(content, encoding="utf-8")
        return ToolResult(tool="write_file", ok=True, content=f"written {path}")

    async def run_command(command: str) -> ToolResult:
        return ToolResult(
            tool="run_command",
            ok=True,
            exit_code=0,
            content=f"$ {command}\nexit code: 0\n1 passed",
        )

    registry = ToolRegistry()
    specs = [
        ToolSpec(
            name="read_file",
            description="Read a file",
            risk=Risk.ALLOW,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="write_file",
            description="Write a file",
            risk=Risk.ALLOW,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="run_command",
            description="Run a command",
            risk=Risk.ALLOW,
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
    ]
    functions = [read_file, write_file, run_command]
    for spec, function in zip(specs, functions, strict=True):
        registry.register(spec, function)
    return registry


def _loop(
    tmp_path: Path,
    scenario: str,
    *,
    budget: Budget | None = None,
) -> tuple[AgentLoop, FakeProvider]:
    tools = _tools(tmp_path)
    provider = FakeProvider(_responses(scenario))
    loop = AgentLoop(
        run_id=f"run-{scenario}",
        goal="update main.py",
        workspace=tmp_path,
        provider=provider,
        codec=ConstrainedJsonCodec(tools.specs()),
        tools=tools,
        assembler=PromptAssembler(
            system="system",
            task="update main.py",
            tools=tools.specs(),
        ),
        token_budget=TokenBudget(),
        memory=MemoryStore(tmp_path / f"{scenario}.sqlite"),
        journal=RunJournal(tmp_path / f"journal-{scenario}", f"run-{scenario}"),
        budget=budget or Budget(max_steps=10),
    )
    return loop, provider


def _close(loop: AgentLoop) -> None:
    loop.memory.close()
    loop.journal.close()


@pytest.mark.asyncio
async def test_successful_coding_run_is_verified(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    loop, _ = _loop(tmp_path, "coding_run")

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.ANSWERED
    assert result.verified is True
    assert result.tool_calls == 3
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_false_success_claim_is_unverified(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    loop, _ = _loop(tmp_path, "false_success")

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.ANSWERED
    assert result.verified is False


@pytest.mark.asyncio
async def test_parse_error_retries_with_changed_input(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    loop, provider = _loop(tmp_path, "parse_retry")

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.ANSWERED
    assert result.retries == 1
    assert len(provider.requests) == 2
    assert provider.requests[0].messages != provider.requests[1].messages
    assert "could not be parsed" in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_repeated_tool_call_stops_for_no_progress(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    loop, _ = _loop(tmp_path, "no_progress")

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.NO_PROGRESS
    assert result.tool_calls == 2


@pytest.mark.asyncio
async def test_max_steps_stops_cleanly(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    loop, provider = _loop(tmp_path, "budget", budget=Budget(max_steps=1))

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert result.steps_taken == 1
    assert provider.calls == 1
