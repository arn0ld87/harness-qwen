"""The prefix contract as a measurable benchmark rule (issue #22).

The contract itself is enforced in ``harness.context.assembler``: the stable
prefix is hashed on every build and an undeclared mismatch raises. This module
turns that assertion into something a run can *report* — it drives a real
``PromptAssembler`` through a scripted sequence of append-zone actions (role
switches, tool results, retrieval) and one declared invalidation, calls the
provider at each step, and records the prefix hash and the cache accounting
side by side.

What that exposes is the invariant the project rests on: across every append
the prefix hash must not move and the cache must keep hitting; the only time
either changes is a declared, reasoned invalidation. An undeclared change is
caught rather than raised past the caller, so a benchmark run records it and
is marked invalid instead of aborting — the run is the evidence, not the
exception. A step whose prefix stayed stable but whose cache missed is an
anomaly: the cache the whole architecture amortises on is broken, and a 0 exit
code must not paper over that.

Out of scope (per the issue): optimising the prefix design. This measures and
watches; it does not change what the prefix contains.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness.benchmark.models import Fingerprint
from harness.context.assembler import (
    InvalidationReason,
    InvalidationRecord,
    PrefixViolation,
    PromptAssembler,
)
from harness.context.budget import TokenBudget
from harness.context.economics import CacheEconomics
from harness.core import GenerationRequest, Role, ToolSpec

if TYPE_CHECKING:
    from harness.models.base import ModelProvider

PREFIX_REPORT_SCHEMA_VERSION = 1


class StepKind(enum.StrEnum):
    """What a step did to the assembler before the model was called.

    ``initial`` is the cold call that establishes the prefix; every later step
    appends to the append zone or declares a prefix change, and the invariant
    is that only the latter moves the hash.
    """

    INITIAL = "initial"
    ROLE = "role"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    DECLARED_INVALIDATION = "declared_invalidation"
    UNDECLARED_TASK = "undeclared_task"


class PrefixStep(BaseModel):
    """One scripted action on the assembler, followed by a model call.

    Only the fields relevant to ``kind`` are read; the rest are ignored. Kept
    as data rather than callables so a step sequence can be serialised and a
    suite file can declare one without writing Python.
    """

    kind: StepKind
    label: str = ""
    # role
    role: Role | None = None
    # tool_result
    tool: str | None = None
    tool_content: str | None = None
    # retrieval
    retrieval_text: str | None = None
    retrieval_source: str = "retrieval"
    # declared_invalidation
    invalidation_reason: InvalidationReason | None = None
    invalidation_note: str = ""
    new_task: str | None = None
    new_repo_map: str | None = None


class PrefixStepResult(BaseModel):
    """The prefix hash and the cache accounting for one model call."""

    index: int
    kind: StepKind
    label: str
    prefix_hash: str
    prefix_tokens: int
    prompt_tokens: int
    cached_tokens: int
    reprocessed_tokens: int
    cache_hit_ratio: float
    declared_invalidation: InvalidationRecord | None = None
    """Set on a ``declared_invalidation`` step: the record the assembler
    produced, with its reason codes and whether the change actually applied."""


class PrefixViolationRecord(BaseModel):
    """An undeclared prefix change, caught so the run records it."""

    at_step: int
    kind: StepKind
    expected: str
    actual: str
    segments: list[str]


class PrefixInvariantReport(BaseModel):
    """The outcome of one prefix-invariant run."""

    schema_version: int = PREFIX_REPORT_SCHEMA_VERSION
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    fingerprint: Fingerprint | None = None
    steps: list[PrefixStepResult] = Field(default_factory=list)
    invalidations: list[InvalidationRecord] = Field(default_factory=list)
    """Every invalidation the assembler recorded, declared or initial."""
    violations: list[PrefixViolationRecord] = Field(default_factory=list)
    """Undeclared prefix changes. Even one invalidates the run."""
    anomalies: list[int] = Field(default_factory=list)
    """Step indices (after the cold call) whose prefix stayed stable but whose
    cache missed — the invariant says that should not happen."""
    cache_hit_ratio: float = 0.0
    """Share of prompt tokens served from cache, aggregated over the warm
    (post-cold) steps. The cold call is excluded: it is supposed to miss."""
    total_reprocessed_tokens: int = 0
    discarded_output_tokens: int = 0
    """Reprocessed tokens expressed as the output the run did not get to
    produce — the §1 figure, via :class:`CacheEconomics`."""
    valid: bool = True
    warnings: list[str] = Field(default_factory=list)


def _apply_step(asm: PromptAssembler, step: PrefixStep) -> None:
    if step.kind is StepKind.ROLE:
        asm.append_role_directive(step.role or Role.PLANNER)
    elif step.kind is StepKind.TOOL_RESULT:
        from harness.core import ToolResult

        asm.append_tool_result(
            ToolResult(tool=step.tool or "run", ok=True,
                       content=step.tool_content or "ok")
        )
    elif step.kind is StepKind.RETRIEVAL:
        asm.append_retrieved(step.retrieval_text or "", source=step.retrieval_source)
    elif step.kind is StepKind.DECLARED_INVALIDATION:
        # Declared before the change, not discovered as a diff afterward
        # (CONTEXT.md §4). A None reason is a misuse; treat it as a no-op
        # rather than silently authorising an undeclared change.
        reason = step.invalidation_reason
        if reason is not None:
            asm.invalidate(reason, note=step.invalidation_note)
        if step.new_task is not None:
            asm.set_task(step.new_task)
        if step.new_repo_map is not None:
            asm.set_repo_map(step.new_repo_map)
    elif step.kind is StepKind.UNDECLARED_TASK:
        # The violation probe: change the task segment *without* declaring,
        # exactly the bug the invariant exists to catch. The assembler raises
        # on the next build; the runner catches it and records the violation.
        asm.set_task(step.new_task or "undeclared new goal")


async def run_prefix_invariant(
    provider: ModelProvider,
    *,
    system: str,
    task: str,
    tools: Sequence[ToolSpec] = (),
    repo_map: str = "",
    steps: Sequence[PrefixStep] = (),
    max_tokens: int = 64,
    run_id: str | None = None,
    fingerprint: Fingerprint | None = None,
    budget: TokenBudget | None = None,
    economics: CacheEconomics | None = None,
    now: Callable[[], datetime] | None = None,
) -> PrefixInvariantReport:
    """Drive the assembler through ``steps`` and check the prefix invariant.

    Each step mutates the assembler's append zone (or declares a prefix
    change), builds, and calls the provider. The prefix hash and the server's
    cache accounting are recorded per call. An undeclared prefix change is
    caught and recorded as a violation rather than raised — the run is the
    evidence. The report is marked invalid on any violation or any cache
    anomaly (stable prefix, cache miss after the cold call).
    """
    asm = PromptAssembler(
        system=system, task=task, tools=tools, repo_map=repo_map,
        budget=budget or TokenBudget(),
    )
    eco = economics or CacheEconomics()
    started = (now or datetime.now)()
    rid = run_id or f"prefix-{started.strftime('%Y%m%dT%H%M%SZ')}"

    results: list[PrefixStepResult] = []
    violations: list[PrefixViolationRecord] = []
    anomalies: list[int] = []
    prev_hash: str | None = None
    total_prompt = 0
    total_cached = 0
    total_reprocessed = 0
    peak_context = 0

    async def _call_and_record(index: int, kind: StepKind, label: str) -> None:
        nonlocal prev_hash, total_prompt, total_cached, total_reprocessed, peak_context
        built = asm.build()
        request = GenerationRequest(
            messages=built.messages, max_tokens=max_tokens,
            prefix_token_estimate=built.prefix_tokens,
        )
        response = await provider.generate(request)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        cached_tokens = usage.cached_tokens
        reprocessed = max(prompt_tokens - cached_tokens, 0)
        ratio = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        peak_context = max(peak_context, prompt_tokens)

        record: InvalidationRecord | None = None
        if kind is StepKind.DECLARED_INVALIDATION and asm.invalidations:
            record = asm.invalidations[-1]

        results.append(PrefixStepResult(
            index=index, kind=kind, label=label,
            prefix_hash=built.prefix_hash, prefix_tokens=built.prefix_tokens,
            prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
            reprocessed_tokens=reprocessed, cache_hit_ratio=ratio,
            declared_invalidation=record,
        ))

        # The invariant: the hash must not move between append steps. After the
        # cold call, a stable hash must also keep hitting the cache — a miss
        # there means the cache the architecture amortises on is broken.
        if prev_hash is not None and kind is not StepKind.DECLARED_INVALIDATION:
            if built.prefix_hash != prev_hash:
                violations.append(PrefixViolationRecord(
                    at_step=index, kind=kind, expected=prev_hash,
                    actual=built.prefix_hash, segments=[],
                ))
            elif cached_tokens == 0 and prompt_tokens > 0:
                anomalies.append(index)

        prev_hash = built.prefix_hash
        total_prompt += prompt_tokens
        total_cached += cached_tokens
        total_reprocessed += reprocessed

    # The cold call establishes the prefix. Every later step is warm.
    try:
        await _call_and_record(0, StepKind.INITIAL, "cold")
    except PrefixViolation as exc:
        violations.append(PrefixViolationRecord(
            at_step=0, kind=StepKind.INITIAL, expected=exc.expected,
            actual=exc.actual, segments=exc.segments,
        ))

    for index, step in enumerate(steps, start=1):
        # Snapshot the segment an undeclared-change probe is about to mutate,
        # so a violation can be reverted and later steps build on the last good
        # prefix instead of re-raising on every subsequent build.
        saved_task = asm.task if step.kind is StepKind.UNDECLARED_TASK else None
        _apply_step(asm, step)
        try:
            await _call_and_record(index, step.kind, step.label or step.kind.value)
        except PrefixViolation as exc:
            violations.append(PrefixViolationRecord(
                at_step=index, kind=step.kind, expected=exc.expected,
                actual=exc.actual, segments=exc.segments,
            ))
            if saved_task is not None:
                asm.set_task(saved_task)
            else:
                _revert_last(asm, step)

    warm_steps = [r for r in results if r.kind is not StepKind.INITIAL]
    warm_prompt = sum(r.prompt_tokens for r in warm_steps)
    warm_cached = sum(r.cached_tokens for r in warm_steps)
    aggregate_ratio = warm_cached / warm_prompt if warm_prompt > 0 else 0.0
    discarded = eco.discarded_output_tokens(
        eco.reprocess_cost(total_reprocessed, context_tokens=peak_context or 1)
    )

    valid = not violations and not anomalies
    finished = (now or datetime.now)()
    return PrefixInvariantReport(
        run_id=rid, started_at=started, finished_at=finished,
        fingerprint=fingerprint, steps=results, invalidations=asm.invalidations,
        violations=violations, anomalies=anomalies,
        cache_hit_ratio=aggregate_ratio, total_reprocessed_tokens=total_reprocessed,
        discarded_output_tokens=discarded, valid=valid,
    )


def _revert_last(asm: PromptAssembler, step: PrefixStep) -> None:
    """Best-effort revert of an undeclared mutation after a violation.

    The assembler does not remember the pre-mutation value (it is left
    untouched on a ``PrefixViolation``, so the *build* state is consistent —
    the drifted segment is the one the caller set). For the declared-invalidation
    path a violation should not occur; for appends it cannot. This guards the
    undeclared-set case by restoring the last good append zone and dropping the
    pending segment change is not possible without the old value, so we only
    rewind the append zone when the step was an append.
    """
    append_kinds = (StepKind.ROLE, StepKind.TOOL_RESULT, StepKind.RETRIEVAL)
    if step.kind in append_kinds and asm.append_messages:
        # Drop the message this step just appended (the last one).
        asm.restore_append(asm.append_messages[:-1])


__all__ = [
    "PREFIX_REPORT_SCHEMA_VERSION",
    "PrefixInvariantReport",
    "PrefixStep",
    "PrefixStepResult",
    "PrefixViolationRecord",
    "StepKind",
    "run_prefix_invariant",
]