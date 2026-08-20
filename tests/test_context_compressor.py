from pathlib import Path

import pytest

from harness.agent import AgentLoop
from harness.context import CacheEconomics, PromptAssembler, TokenBudget
from harness.context.compressor import (
    CompressionState,
    RetrieveAgain,
    SummarizeOldConversation,
    escalate,
)
from harness.core import Budget, ContextOverflow, Message, StopReason
from harness.memory import MemoryStore
from harness.models import FakeProvider
from harness.protocol.constrained import ConstrainedJsonCodec
from harness.telemetry import RunJournal
from harness.tools import ToolRegistry


def _loop(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    assembler: PromptAssembler,
    ladder: list[object],
    retrieve: object = None,
) -> AgentLoop:
    return AgentLoop(
        run_id="run-1",
        goal="fix overflow",
        workspace=tmp_path,
        provider=provider,
        codec=ConstrainedJsonCodec(),
        tools=ToolRegistry(),
        assembler=assembler,
        token_budget=TokenBudget(),
        memory=MemoryStore(tmp_path / "memory.sqlite"),
        journal=RunJournal(tmp_path / "run", "run-1"),
        budget=Budget(max_steps=20),
        economics=CacheEconomics(safety_margin=0.001),
        compression_ladder=ladder,
        retrieve=retrieve,
    )


def _close(loop: AgentLoop) -> None:
    loop.memory.close()
    loop.journal.close()


@pytest.mark.asyncio
async def test_retrieval_alone_does_not_retry_context_overflow(tmp_path: Path) -> None:
    provider = FakeProvider(
        ["unused", '{"action":"answer","content":"done"}'],
        errors={1: ContextOverflow("too large")},
    )
    assembler = PromptAssembler(system="system", task="task")
    loop = _loop(
        tmp_path,
        provider,
        assembler=assembler,
        ladder=[RetrieveAgain()],
        retrieve=lambda _query: "more context",
    )

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_real_compression_retries_context_overflow(tmp_path: Path) -> None:
    provider = FakeProvider(
        ["unused", '{"action":"answer","content":"done"}'],
        errors={1: ContextOverflow("too large")},
    )
    assembler = PromptAssembler(system="system", task="task")
    for index in range(8):
        assembler.append(Message(role="user", content=f"{index}:" + "x" * 2_000))
    loop = _loop(
        tmp_path,
        provider,
        assembler=assembler,
        ladder=[SummarizeOldConversation(keep_recent=2, min_span=2)],
    )

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.ANSWERED
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_unchanged_compression_ladder_stops_overflow(tmp_path: Path) -> None:
    provider = FakeProvider(
        ["unused", '{"action":"answer","content":"done"}'],
        errors={1: ContextOverflow("too large")},
    )
    loop = _loop(
        tmp_path,
        provider,
        assembler=PromptAssembler(system="system", task="task"),
        ladder=[SummarizeOldConversation()],
    )

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert provider.calls == 1


def test_retrieve_again_remains_available_outside_overflow() -> None:
    assembler = PromptAssembler(system="system", task="task")
    budget = TokenBudget()
    outcomes = escalate(
        [RetrieveAgain()],
        CompressionState(
            assembler=assembler,
            budget=budget,
            retrieve=lambda _query: "fresh context",
            retrieval_query="query",
        ),
        CacheEconomics(),
        current_step=1,
        max_steps=10,
    )

    assert outcomes[0].applied is True
    assert outcomes[0].freed_tokens == 0
    assert outcomes[0].resolves_overflow is False
    assert assembler.append_messages[-1].content.endswith("fresh context")


def test_token_budget_is_the_only_prompt_limit_authority() -> None:
    assert "max_prompt_tokens" not in Budget.model_fields

    report = TokenBudget(
        context_window=100,
        soft_ceiling=80,
        generation_reserve=20,
        token_counter=lambda text: len(text),
    ).report_tokens(prefix_tokens=40, append_tokens=41)

    assert report.append.within_budget is False
    assert report.over_hard_ceiling is True
