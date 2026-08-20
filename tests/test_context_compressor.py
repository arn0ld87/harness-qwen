from pathlib import Path

import pytest

from harness.agent import AgentLoop
from harness.context import CacheEconomics, PromptAssembler, TokenBudget
from harness.context.compressor import (
    CompressionState,
    RetrieveAgain,
    SummarizeOldConversation,
    _AppendRewriteStrategy,
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
    token_budget: TokenBudget | None = None,
    economics: CacheEconomics | None = None,
) -> AgentLoop:
    return AgentLoop(
        run_id="run-1",
        goal="fix overflow",
        workspace=tmp_path,
        provider=provider,
        codec=ConstrainedJsonCodec(),
        tools=ToolRegistry(),
        assembler=assembler,
        token_budget=token_budget or TokenBudget(),
        memory=MemoryStore(tmp_path / "memory.sqlite"),
        journal=RunJournal(tmp_path / "run", "run-1"),
        budget=Budget(max_steps=20),
        economics=economics or CacheEconomics(safety_margin=0.001),
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


# --- Issue #6: ContextOverflow-Recovery netto korrekt ---------------------


class _LyingGrowStrategy(_AppendRewriteStrategy):
    """Reports a fake positive local saving but grows the append on apply.

    Exactly the failure mode issue #6 names: a rung that looks like it frees
    tokens locally while leaving the prompt larger than before. The loop must
    measure the net effect, not trust the rung's self-reported ``freed``.
    """

    name = "lying_grow"

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        return [
            *state.assembler.append_messages,
            Message(role="user", content="Y" * 4_000),
        ]

    def estimate_freed_tokens(self, state: CompressionState) -> int:
        return 500  # lies: claims to free, but _candidate grows the append


@pytest.mark.asyncio
async def test_retrieve_again_is_skipped_at_context_overflow(tmp_path: Path) -> None:
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
        retrieve=lambda _query: "fresh context",
    )

    result = await loop.run()
    _close(loop)

    assert result.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert provider.calls == 1
    # The retrieved message must not have been appended: at overflow a pure
    # append cannot relieve the condition, so RetrieveAgain is skipped, not
    # merely run-and-deemed-insufficient.
    assert not any(
        "fresh context" in msg.content for msg in assembler.append_messages
    )


@pytest.mark.asyncio
async def test_net_negative_compression_is_not_overflow_recovery(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        ["unused", '{"action":"answer","content":"done"}'],
        errors={1: ContextOverflow("too large")},
    )
    assembler = PromptAssembler(system="system", task="task")
    assembler.append(Message(role="user", content="x" * 180))
    loop = _loop(
        tmp_path,
        provider,
        assembler=assembler,
        ladder=[_LyingGrowStrategy()],
    )

    result = await loop.run()
    _close(loop)

    # The rung applied (it clears the economics bar) and self-reports freed
    # tokens, but the prompt grew net — recovery must be refused.
    assert result.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_compression_still_over_hard_ceiling_is_not_recovery(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        ["unused", '{"action":"answer","content":"done"}'],
        errors={1: ContextOverflow("too large")},
    )
    assembler = PromptAssembler(system="system", task="task")
    # 12 messages ~54 tokens each (~648 append tokens) over a 400-token window.
    for _ in range(12):
        assembler.append(Message(role="user", content="x" * 180))
    loop = _loop(
        tmp_path,
        provider,
        assembler=assembler,
        ladder=[SummarizeOldConversation(keep_recent=6, min_span=2)],
        token_budget=TokenBudget(
            context_window=400, soft_ceiling=320, generation_reserve=30,
        ),
    )

    result = await loop.run()
    _close(loop)

    # Summarizing shrinks the prompt net, but the tail alone still exceeds the
    # hard ceiling — so the overflow is not actually resolved.
    assert result.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert provider.calls == 1
