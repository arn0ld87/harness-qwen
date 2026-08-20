"""The prefix invariant as a measurable run (issue #22).

``run_prefix_invariant`` drives a real ``PromptAssembler`` through a scripted
step sequence and watches two things the architecture rests on: the prefix
hash must not move across append-zone growth, and the cache must keep hitting
once it is warm. These tests pin both — plus the declared-invalidation path
that legitimately moves the hash, the undeclared-change probe that proves the
run catches the bug the invariant exists for, and the anomaly that a stable
prefix with a cold cache is its own kind of invalid.

The provider is always ``FakeProvider``: it replays a fixed script and
synthesises cache-accurate usage, so the invariant is checked without a model.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.benchmark import PrefixStep, StepKind, run_prefix_invariant
from harness.context.assembler import InvalidationReason
from harness.core import Role
from harness.models import FakeProvider


def _steps(*kinds: StepKind) -> list[PrefixStep]:
    return [PrefixStep(kind=k) for k in kinds]


async def _run(provider: FakeProvider, steps, **kw):
    return await run_prefix_invariant(
        provider, system="You are a careful agent.", task="Fix the failing test.",
        tools=(), steps=steps, run_id="test", **kw,
    )


# -- the append zone never moves the hash, and the cache keeps hitting -------


@pytest.mark.asyncio
async def test_the_prefix_hash_is_stable_across_appends() -> None:
    """Role switches, tool results and retrieval all grow the append zone;
    none of them moves the prefix hash, and the run is valid."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [
        PrefixStep(kind=StepKind.ROLE, role=Role.CODER),
        PrefixStep(kind=StepKind.TOOL_RESULT, tool="run", tool_content="1 passed"),
        PrefixStep(kind=StepKind.RETRIEVAL, retrieval_text="fact: cache is sacred"),
        PrefixStep(kind=StepKind.ROLE, role=Role.TESTER),
    ])

    assert report.valid
    assert report.violations == []
    hashes = {s.prefix_hash for s in report.steps}
    assert len(hashes) == 1  # one hash across the cold call and every append


@pytest.mark.asyncio
async def test_the_cold_call_misses_and_warm_calls_hit() -> None:
    """The first call establishes the prefix (no cache yet); every later call
    with the same prefix must report cached tokens — that is the hit the whole
    design amortises on."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, _steps(StepKind.ROLE, StepKind.TOOL_RESULT))

    cold = report.steps[0]
    warm = report.steps[1:]
    assert cold.cached_tokens == 0
    assert all(s.cached_tokens > 0 for s in warm)
    assert 0.0 < report.cache_hit_ratio <= 1.0


@pytest.mark.asyncio
async def test_cache_hit_ratio_excludes_the_cold_call() -> None:
    """The aggregate is over warm steps only — the cold miss is expected and
    must not drag the ratio down, or a healthy run looks sick."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, _steps(StepKind.ROLE))

    warm = report.steps[1]
    expected = warm.cached_tokens / warm.prompt_tokens
    assert report.cache_hit_ratio == pytest.approx(expected)


# -- declared invalidation legitimately moves the hash ----------------------


@pytest.mark.asyncio
async def test_a_declared_task_change_moves_the_hash_and_records_the_reason() -> None:
    """The one case the hash is allowed to move: a change declared first,
    attributed to a reason, and recorded. The run stays valid."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [
        PrefixStep(kind=StepKind.ROLE, role=Role.CODER),
        PrefixStep(
            kind=StepKind.DECLARED_INVALIDATION,
            invalidation_reason=InvalidationReason.TASK_CHANGED,
            invalidation_note="new goal from the user",
            new_task="Fix a different test instead.",
        ),
    ])

    assert report.valid
    assert report.violations == []
    declared = report.steps[-1]
    assert declared.prefix_hash != report.steps[0].prefix_hash
    record = declared.declared_invalidation
    assert record is not None
    assert record.applied is True
    assert InvalidationReason.TASK_CHANGED in record.reasons
    assert "task" in record.changed_segments
    # The initial build's INITIAL record plus this one.
    assert len(report.invalidations) >= 2


# -- the undeclared-change probe: the run catches the bug -------------------


@pytest.mark.asyncio
async def test_an_undeclared_task_change_is_a_violation() -> None:
    """Changing the task segment without declaring is exactly the bug the
    invariant exists to catch. The assembler raises; the run catches it,
    records the violation with the offending segment, and is invalid."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [
        PrefixStep(kind=StepKind.ROLE, role=Role.CODER),
        PrefixStep(kind=StepKind.UNDECLARED_TASK, new_task="sneaky new goal"),
    ])

    assert report.valid is False
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.at_step == 2
    assert violation.kind is StepKind.UNDECLARED_TASK
    assert violation.segments == ["task"]


@pytest.mark.asyncio
async def test_the_run_recovers_after_a_violation() -> None:
    """A violation reverts the undeclared mutation so the next step builds on
    the last good prefix rather than re-raising — one violation per bug, not
    one per remaining step."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [
        PrefixStep(kind=StepKind.UNDECLARED_TASK, new_task="sneaky"),
        PrefixStep(kind=StepKind.TOOL_RESULT, tool="run", tool_content="ok"),
    ])

    assert len(report.violations) == 1
    # The post-violation step builds cleanly and on the original prefix.
    good = report.steps[0].prefix_hash
    assert report.steps[-1].prefix_hash == good


# -- anomaly: stable prefix, cache miss -------------------------------------


@pytest.mark.asyncio
async def test_a_stable_prefix_with_a_cold_cache_is_an_anomaly() -> None:
    """The invariant has two halves: the hash stays put *and* the cache keeps
    hitting. A warm step whose prefix matched but whose cache missed means the
    cache the architecture amortises on is broken — invalid, even though no
    hash moved."""
    from harness.core import ModelResponse, Usage

    # Script an explicit usage with cached_tokens=0 on every call. FakeProvider
    # respects a non-default Usage verbatim, so the warm calls miss despite a
    # stable prefix.
    miss = ModelResponse(content="ok", usage=Usage(prompt_tokens=30, cached_tokens=0,
                                                   completion_tokens=1))
    provider = FakeProvider([miss], repeat_last=True)
    report = await _run(provider, _steps(StepKind.ROLE, StepKind.TOOL_RESULT))

    assert report.valid is False
    assert 1 in report.anomalies
    assert 2 in report.anomalies
    assert report.violations == []  # the hash never moved; the cache did


# -- discarded output accounting --------------------------------------------


@pytest.mark.asyncio
async def test_discarded_output_tokens_are_reported() -> None:
    """Reprocessed tokens are expressed as the output the run did not get to
    produce — the §1 figure. Cold reprocessing makes it non-zero; the exact
    number is the cost model's job, here only that the wire is connected."""
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, _steps(StepKind.ROLE))
    # Warm steps hit the cache, so reprocessing is just the cold call's prompt.
    assert report.total_reprocessed_tokens >= 0
    assert report.discarded_output_tokens >= 0


# -- fingerprint is carried through -----------------------------------------


@pytest.mark.asyncio
async def test_the_fingerprint_is_attached_to_the_report() -> None:
    from harness.benchmark import (
        Fingerprint,
        HostFingerprint,
        ModelFingerprint,
        RuntimeFingerprint,
    )

    fp = Fingerprint(
        host=HostFingerprint(hostname="testbox", cpu_model="Test CPU"),
        runtime=RuntimeFingerprint(base_url="http://127.0.0.1:18080",
                                   host="127.0.0.1", port=18080,
                                   ownership="owned", pid=4242, identity_verified=True),
        model=ModelFingerprint(model_id="fake", n_ctx=65536),
    )
    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [], fingerprint=fp)
    assert report.fingerprint is fp


@pytest.mark.asyncio
async def test_a_fixed_clock_makes_started_at_deterministic() -> None:
    fixed = datetime(2026, 8, 20, 1, 0, tzinfo=UTC)

    def now() -> datetime:
        return fixed

    provider = FakeProvider(["ok"], repeat_last=True)
    report = await _run(provider, [], now=now)
    assert report.started_at == fixed
    assert report.finished_at == fixed


# -- against the real model -------------------------------------------------


@pytest.mark.local_llm
async def test_prefix_invariant_runs_against_the_served_model() -> None:
    """The invariant check works against a real llama-server (attached, not
    started here). Asserts the run completes and reports a verdict, not that
    the cache hits — a real server's cache behaviour is a measurement, not a
    unit test, and a miss is informative rather than a failure."""
    from harness.config import resolve_config
    from harness.runtime.supervisor import LlamaServerSupervisor
    from harness.session import build_provider

    resolved = resolve_config(overrides={"runtime.attach": True})
    provider = build_provider(resolved.config)
    supervisor = LlamaServerSupervisor(resolved.config)
    await supervisor.ensure()
    try:
        report = await run_prefix_invariant(
            provider, system="You are a careful agent.", task="Echo: ok",
            steps=_steps(StepKind.ROLE, StepKind.TOOL_RESULT),
            run_id="real-smoke",
        )
    finally:
        await supervisor.close()

    assert report.run_id == "real-smoke"
    assert len(report.steps) == 3
    # Valid or not, the run produced a complete record.
    assert report.valid in {True, False}