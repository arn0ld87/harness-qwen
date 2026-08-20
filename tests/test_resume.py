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


def _tools() -> ToolRegistry:
    async def run_test_command(command: str) -> ToolResult:
        return ToolResult(
            tool="run_command",
            ok=True,
            exit_code=0,
            content=f"$ {command}\nexit code: 0\n1 passed",
        )

    registry = ToolRegistry()
    registry.register(
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
        run_test_command,
    )
    return registry


def _loop(
    tmp_path: Path,
    *,
    provider: FakeProvider,
    memory: MemoryStore,
    journal_name: str,
) -> AgentLoop:
    tools = _tools()
    return AgentLoop(
        run_id="resume-run",
        goal="make tests pass",
        workspace=tmp_path,
        provider=provider,
        codec=ConstrainedJsonCodec(tools.specs()),
        tools=tools,
        assembler=PromptAssembler(
            system="system",
            task="make tests pass",
            tools=tools.specs(),
        ),
        token_budget=TokenBudget(),
        memory=memory,
        journal=RunJournal(tmp_path / journal_name, "resume-run"),
        budget=Budget(max_steps=10, max_tool_calls=2, max_retries=2),
    )


@pytest.mark.asyncio
async def test_resume_restores_evidence_budgets_context_and_usage(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    first_memory = MemoryStore(database)
    first_provider = FakeProvider(
        [
            "not json",
            (
                '{"action":"tool","tool":"run_command",'
                '"arguments":{"command":"pytest"},"reason":"verify"}'
            ),
            "unused",
        ],
        errors={3: RuntimeError("simulated process death")},
    )
    first_loop = _loop(
        tmp_path,
        provider=first_provider,
        memory=first_memory,
        journal_name="first-run",
    )

    with pytest.raises(RuntimeError, match="simulated process death"):
        await first_loop.run()

    tool_step = next(step for step in first_memory.get_steps("resume-run") if step.tool)
    stored = first_memory.load_runtime_state("resume-run")
    assert stored is not None
    assert stored.tool_calls == 1
    assert stored.retries_used == 1
    assert stored.executed_steps[0].id == tool_step.id
    assert any(message.role == "tool" for message in stored.append_history)
    assert stored.prompt_tokens > 0
    first_loop.journal.close()
    first_memory.close()

    second_memory = MemoryStore(database)
    answer = (
        '{"action":"answer","content":"All tests pass",'
        f'"evidence":[{{"kind":"test","step_id":{tool_step.id}}}]}}'
    )
    second_loop = _loop(
        tmp_path,
        provider=FakeProvider([answer]),
        memory=second_memory,
        journal_name="second-run",
    )

    result = await second_loop.run()

    assert result.stop_reason is StopReason.ANSWERED
    assert result.verified is True
    assert result.tool_calls == 1
    assert result.retries == 1
    assert result.total_prompt_tokens > 0
    assert any(message.role == "tool" for message in second_loop.assembler.append_messages)
    second_loop.journal.close()
    second_memory.close()
