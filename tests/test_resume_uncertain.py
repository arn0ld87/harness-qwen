"""Resume after a crash around a tool with side effects (issue #8).

Three moments matter, and only one of them is safe to replay blindly:

1. killed *before* the tool ran — nothing happened, but the database cannot
   tell that apart from (2);
2. killed *after* the side effect, *before* the checkpoint — the effect is on
   disk and the run does not know it;
3. killed *after* the checkpoint — the step is DONE and resume is boring.

Cases 1 and 2 are indistinguishable in the step history, which is precisely
why a mutating step found RUNNING resumes as UNCERTAIN instead of FAILED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.agent import AgentLoop
from harness.context import PromptAssembler, TokenBudget
from harness.core import (
    Budget,
    Risk,
    SideEffect,
    StepStatus,
    StopReason,
    ToolResult,
    ToolSpec,
)
from harness.memory import MemoryStore
from harness.models import FakeProvider
from harness.protocol.constrained import ConstrainedJsonCodec
from harness.telemetry import RunJournal
from harness.tools import ToolRegistry

RUN_ID = "uncertain-run"

WRITE_CALL = (
    '{"action":"tool","tool":"append_line",'
    '"arguments":{"text":"one"},"reason":"record it"}'
)
READ_CALL = (
    '{"action":"tool","tool":"read_ledger","arguments":{},"reason":"look"}'
)
ANSWER = '{"action":"answer","content":"done","evidence":[]}'


def _tools(ledger: Path) -> ToolRegistry:
    """A mutating tool that appends to a file, plus a read-only one."""

    async def append_line(text: str) -> ToolResult:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(f"{text}\n")
        return ToolResult(tool="append_line", ok=True, exit_code=0, content="written")

    async def read_ledger() -> ToolResult:
        content = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
        return ToolResult(tool="read_ledger", ok=True, exit_code=0, content=content)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="append_line",
            description="Append a line to the ledger",
            risk=Risk.ALLOW,
            side_effect=SideEffect.MUTATING,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        append_line,
    )
    registry.register(
        ToolSpec(
            name="read_ledger",
            description="Read the ledger",
            risk=Risk.ALLOW,
            side_effect=SideEffect.NONE,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        read_ledger,
    )
    return registry


def _loop(
    tmp_path: Path,
    *,
    provider: FakeProvider,
    memory: MemoryStore,
    journal_name: str,
    tools: ToolRegistry,
) -> AgentLoop:
    return AgentLoop(
        run_id=RUN_ID,
        goal="keep a ledger",
        workspace=tmp_path,
        provider=provider,
        codec=ConstrainedJsonCodec(tools.specs()),
        tools=tools,
        assembler=PromptAssembler(
            system="system", task="keep a ledger", tools=tools.specs()
        ),
        token_budget=TokenBudget(),
        memory=memory,
        journal=RunJournal(tmp_path / journal_name, RUN_ID),
        budget=Budget(max_steps=10, max_tool_calls=6, max_retries=2),
    )


def _steps_for(memory: MemoryStore, tool: str) -> list[Any]:
    return [step for step in memory.get_steps(RUN_ID) if step.tool == tool]


async def _dead_invoke(name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
    """Stand-in for a process killed between ``append_step`` and execution."""
    raise RuntimeError("simulated process death")


@pytest.mark.asyncio
async def test_crash_before_side_effect_resumes_as_uncertain(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    tools = _tools(ledger)
    tools.invoke = _dead_invoke  # type: ignore[method-assign]
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL]),
        memory=memory,
        journal_name="crash-1",
        tools=tools,
    )

    with pytest.raises(RuntimeError, match="simulated process death"):
        await loop.run()

    assert not ledger.exists()
    assert _steps_for(memory, "append_line")[0].status is StepStatus.RUNNING
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([ANSWER]),
        memory=second,
        journal_name="resume-1",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    step = _steps_for(second, "append_line")[0]
    assert step.status is StepStatus.UNCERTAIN
    assert result.uncertain_steps and result.uncertain_steps[0].tool == "append_line"
    assert result.uncertain_steps[0].step_id == step.id
    resumed.journal.close()
    second.close()


@pytest.mark.asyncio
async def test_crash_after_side_effect_does_not_repeat_it(tmp_path: Path) -> None:
    """The effect landed; resume must not silently apply it a second time."""
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    tools = _tools(ledger)
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL]),
        memory=memory,
        journal_name="crash-2",
        tools=tools,
    )

    def dead_checkpoint(**kwargs: Any) -> None:
        raise RuntimeError("simulated process death")

    memory.save_tool_checkpoint = dead_checkpoint  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated process death"):
        await loop.run()

    assert ledger.read_text(encoding="utf-8") == "one\n"
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        # The model tries the very same call again: it must be refused once.
        provider=FakeProvider([WRITE_CALL, ANSWER]),
        memory=second,
        journal_name="resume-2",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    assert ledger.read_text(encoding="utf-8") == "one\n"
    assert result.stop_reason is StopReason.ANSWERED
    assert _steps_for(second, "append_line")[0].status is StepStatus.UNCERTAIN
    refused = _steps_for(second, "append_line")[1]
    assert refused.status is StepStatus.SKIPPED
    assert refused.error_kind == "uncertain_side_effect"
    resumed.journal.close()
    second.close()


@pytest.mark.asyncio
async def test_refused_call_runs_when_the_model_insists(tmp_path: Path) -> None:
    """The guard informs once; a second deliberate attempt is the model's call."""
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    tools = _tools(ledger)
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL]),
        memory=memory,
        journal_name="crash-3",
        tools=tools,
    )

    def dead_checkpoint(**kwargs: Any) -> None:
        raise RuntimeError("simulated process death")

    memory.save_tool_checkpoint = dead_checkpoint  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await loop.run()
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL, WRITE_CALL, ANSWER]),
        memory=second,
        journal_name="resume-3",
        tools=_tools(ledger),
    )
    await resumed.run()

    assert ledger.read_text(encoding="utf-8") == "one\none\n"
    resumed.journal.close()
    second.close()


@pytest.mark.asyncio
async def test_read_only_tool_is_safe_to_repeat(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("existing\n", encoding="utf-8")
    memory = MemoryStore(tmp_path / "memory.sqlite")
    tools = _tools(ledger)
    tools.invoke = _dead_invoke  # type: ignore[method-assign]
    loop = _loop(
        tmp_path,
        provider=FakeProvider([READ_CALL]),
        memory=memory,
        journal_name="crash-4",
        tools=tools,
    )
    with pytest.raises(RuntimeError):
        await loop.run()
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([READ_CALL, ANSWER]),
        memory=second,
        journal_name="resume-4",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    orphan = _steps_for(second, "read_ledger")[0]
    assert orphan.status is StepStatus.FAILED
    assert result.uncertain_steps == []
    repeated = _steps_for(second, "read_ledger")[1]
    assert repeated.status is StepStatus.DONE
    resumed.journal.close()
    second.close()


@pytest.mark.asyncio
async def test_completed_checkpoint_leaves_nothing_uncertain(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    loop = _loop(
        tmp_path,
        # The tool completes and checkpoints; the next model call kills us.
        provider=FakeProvider([WRITE_CALL, "unused"], errors={2: RuntimeError("death")}),
        memory=memory,
        journal_name="crash-5",
        tools=_tools(ledger),
    )
    with pytest.raises(RuntimeError, match="death"):
        await loop.run()

    assert _steps_for(memory, "append_line")[0].status is StepStatus.DONE
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([ANSWER]),
        memory=second,
        journal_name="resume-5",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    assert result.uncertain_steps == []
    assert ledger.read_text(encoding="utf-8") == "one\n"
    resumed.journal.close()
    second.close()


def test_tool_spec_defaults_to_mutating() -> None:
    """Fail-closed: an undeclared tool is treated as one with side effects."""
    spec = ToolSpec(name="x", description="", parameters={"type": "object"})
    assert spec.side_effect is SideEffect.MUTATING


def test_side_effect_is_not_part_of_the_model_declaration() -> None:
    """The prefix is sacred: side-effect metadata is harness-internal."""
    spec = ToolSpec(
        name="x", description="d", parameters={"type": "object"},
        side_effect=SideEffect.NONE,
    )
    declaration = spec.to_openai_tool()
    assert "side_effect" not in str(declaration)


@pytest.mark.asyncio
async def test_guard_survives_a_second_crash_during_recovery(tmp_path: Path) -> None:
    """A death right after resume must not drop the protection resume added."""
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL]),
        memory=memory,
        journal_name="crash-6",
        tools=_tools(ledger),
    )

    def dead_checkpoint(**kwargs: Any) -> None:
        raise RuntimeError("simulated process death")

    memory.save_tool_checkpoint = dead_checkpoint  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await loop.run()
    loop.journal.close()
    memory.close()

    # Resume, then die again before the model ever produced an action.
    second = MemoryStore(tmp_path / "memory.sqlite")
    dying = _loop(
        tmp_path,
        provider=FakeProvider(["unused"], errors={1: RuntimeError("death again")}),
        memory=second,
        journal_name="resume-6a",
        tools=_tools(ledger),
    )
    with pytest.raises(RuntimeError, match="death again"):
        await dying.run()
    dying.journal.close()
    second.close()

    third = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL, ANSWER]),
        memory=third,
        journal_name="resume-6b",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    assert ledger.read_text(encoding="utf-8") == "one\n"
    refused = _steps_for(third, "append_line")[-1]
    assert refused.error_kind == "uncertain_side_effect"
    # The guard fired, so the ambiguity is no longer open at the end of the run.
    assert result.uncertain_steps == []
    resumed.journal.close()
    third.close()


async def _crash_after_side_effect(tmp_path: Path, ledger: Path, journal: str) -> None:
    """Leave the database with a mutating step stuck at RUNNING."""
    memory = MemoryStore(tmp_path / "memory.sqlite")
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL]),
        memory=memory,
        journal_name=journal,
        tools=_tools(ledger),
    )

    def dead_checkpoint(**kwargs: Any) -> None:
        raise RuntimeError("simulated process death")

    memory.save_tool_checkpoint = dead_checkpoint  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await loop.run()
    loop.journal.close()
    memory.close()


@pytest.mark.asyncio
async def test_recovery_is_all_or_nothing(tmp_path: Path) -> None:
    """A death *inside* recovery must not leave the step marked but unguarded.

    Committing the UNCERTAIN status without its guard would produce a database
    that looks recovered while the protection against repeating the side effect
    is gone — the next resume would replay it silently.
    """
    ledger = tmp_path / "ledger.txt"
    await _crash_after_side_effect(tmp_path, ledger, "crash-7")

    second = MemoryStore(tmp_path / "memory.sqlite")
    dying = _loop(
        tmp_path,
        provider=FakeProvider(["unused"]),
        memory=second,
        journal_name="resume-7a",
        tools=_tools(ledger),
    )

    def dead_commit(**kwargs: Any) -> None:
        raise RuntimeError("died during recovery")

    second.commit_resume_recovery = dead_commit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="died during recovery"):
        await dying.run()

    # Nothing was written: the step is still the orphan it was.
    assert _steps_for(second, "append_line")[0].status is StepStatus.RUNNING
    dying.journal.close()
    second.close()

    third = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL, ANSWER]),
        memory=third,
        journal_name="resume-7b",
        tools=_tools(ledger),
    )
    await resumed.run()

    assert ledger.read_text(encoding="utf-8") == "one\n"
    assert _steps_for(third, "append_line")[0].status is StepStatus.UNCERTAIN
    resumed.journal.close()
    third.close()


@pytest.mark.asyncio
async def test_warning_is_not_repeated_on_a_later_resume(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.txt"
    await _crash_after_side_effect(tmp_path, ledger, "crash-8")

    second = MemoryStore(tmp_path / "memory.sqlite")
    dying = _loop(
        tmp_path,
        provider=FakeProvider(["unused"], errors={1: RuntimeError("death again")}),
        memory=second,
        journal_name="resume-8a",
        tools=_tools(ledger),
    )
    with pytest.raises(RuntimeError, match="death again"):
        await dying.run()
    dying.journal.close()
    second.close()

    third = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        provider=FakeProvider([ANSWER]),
        memory=third,
        journal_name="resume-8b",
        tools=_tools(ledger),
    )
    await resumed.run()

    warnings = [
        message
        for message in resumed.assembler.append_messages
        if message.content.startswith("[resume]")
    ]
    assert len(warnings) == 1
    resumed.journal.close()
    third.close()


@pytest.mark.asyncio
async def test_refusal_costs_neither_budget_nor_progress(tmp_path: Path) -> None:
    """The refusal must not be the thing that ends the run it interrupted.

    A refused call did not execute, so counting it as a repeat would let the
    no-progress rule trip on the very retry the guard exists to permit.
    """
    ledger = tmp_path / "ledger.txt"
    memory = MemoryStore(tmp_path / "memory.sqlite")
    loop = _loop(
        tmp_path,
        provider=FakeProvider([WRITE_CALL, WRITE_CALL]),
        memory=memory,
        journal_name="crash-9",
        tools=_tools(ledger),
    )
    original = memory.save_tool_checkpoint
    calls = {"n": 0}

    def die_on_second(**kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("simulated process death")
        original(**kwargs)

    memory.save_tool_checkpoint = die_on_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await loop.run()
    loop.journal.close()
    memory.close()

    second = MemoryStore(tmp_path / "memory.sqlite")
    resumed = _loop(
        tmp_path,
        # refused, then the model insists, then it answers
        provider=FakeProvider([WRITE_CALL, WRITE_CALL, ANSWER]),
        memory=second,
        journal_name="resume-9",
        tools=_tools(ledger),
    )
    result = await resumed.run()

    assert result.stop_reason is StopReason.ANSWERED
    assert result.tool_calls == 2  # one before the crash, one deliberate repeat
    resumed.journal.close()
    second.close()
