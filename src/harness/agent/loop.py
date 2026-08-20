"""The agent loop (ARCHITECTURE.md, "Agent loop"): build context, call the
model, validate the action, execute or answer, check progress, repeat.

Every bound in ``core.Budget`` is enforced. Every step is written to
``MemoryStore`` *before* it executes, so a killed process resumes from the
last committed step instead of from zero — the entire reason the store
exists (a cold prompt costs ~25 s on this machine).

The ``PromptAssembler`` and ``ActionCodec`` passed in must already be fully
configured: any codec-specific tool-rendering text
(``ActionCodec.render_tools``) belongs in the assembler's prefix content
before it reaches this loop, because building that text is a one-time setup
concern, not a per-step one, and the loop never rewrites the prefix except
through the compression ladder's declared invalidations.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.agent.overflow import recover_overflow
from harness.agent.retry import ErrorCategory, RetryPolicy
from harness.agent.roles import RoleSequencer
from harness.agent.verifier import ExecutedStep, Verifier, capture_workspace_baseline
from harness.context.assembler import AssembledPrompt, PrefixViolation, PromptAssembler
from harness.context.budget import TokenBudget
from harness.context.compressor import (
    CompressionState,
    CompressionStrategy,
    RetrieveFn,
    default_ladder,
    escalate,
)
from harness.context.economics import CacheEconomics
from harness.core import (
    Action,
    AnswerAction,
    Budget,
    GenerationRequest,
    Message,
    ParseError,
    ProviderError,
    Role,
    RunResult,
    RunRuntimeState,
    StepStatus,
    StopReason,
    TaskState,
    ToolAction,
    Usage,
)
from harness.memory.migrations import StoreError
from harness.memory.store import MemoryStore
from harness.models.base import ModelProvider
from harness.protocol.codec import ActionCodec
from harness.telemetry.journal import RunJournal
from harness.tools.registry import ToolRegistry

NO_PROGRESS_REPEAT = 3
"""Consecutive identical (tool, arguments) calls that count as no progress.
Checked before execution, so the repeated call never runs and never draws
against the tool-call budget — a stuck run stops instead of burning it."""


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class _StepOutcome:
    """One call to :meth:`AgentLoop._run_step`'s result: exactly one of the
    two fields is set — a stop reason ends the run, an action hands control
    back to the loop to execute or answer."""

    stop: StopReason | None = None
    action: Action | None = None


class AgentLoop:
    """Runs one task to completion, budget exhaustion, or a stop condition."""

    def __init__(
        self,
        *,
        run_id: str,
        goal: str,
        workspace: str | Path,
        provider: ModelProvider,
        codec: ActionCodec,
        tools: ToolRegistry,
        assembler: PromptAssembler,
        token_budget: TokenBudget,
        memory: MemoryStore,
        journal: RunJournal,
        budget: Budget | None = None,
        economics: CacheEconomics | None = None,
        compression_ladder: Sequence[CompressionStrategy] | None = None,
        retry_policy: RetryPolicy | None = None,
        role_sequencer: RoleSequencer | None = None,
        verifier: Verifier | None = None,
        retrieve: RetrieveFn | None = None,
    ) -> None:
        self.run_id = run_id
        self.goal = goal
        self.workspace = Path(workspace)
        self.provider = provider
        self.codec = codec
        self.tools = tools
        self.assembler = assembler
        self.token_budget = token_budget
        self.memory = memory
        self.journal = journal
        self.budget = budget or Budget()
        self.economics = economics or CacheEconomics()
        self.ladder: list[CompressionStrategy] = list(compression_ladder or default_ladder())
        self.retries = retry_policy or RetryPolicy(self.budget.max_retries)
        self.roles = role_sequencer or RoleSequencer()
        self.verifier = verifier or Verifier()
        self.retrieve = retrieve

        self._executed: list[ExecutedStep] = []
        self._recent_calls: deque[tuple[str, str]] = deque(maxlen=NO_PROGRESS_REPEAT)
        self._tool_calls = 0
        self._retries_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_tokens = 0
        self._elapsed_offset = 0.0
        self._active_step_index: int | None = None
        self._started_monotonic = time.monotonic()
        self._cancelled = False

    def cancel(self) -> None:
        """Request a cooperative stop, honoured at the top of the next step.

        Not a signal handler and not ``asyncio.CancelledError`` — a process
        kill cannot be intercepted at all, which is exactly the case
        ``MemoryStore`` exists to survive instead. This is for a caller that
        wants the run to stop cleanly and still get a ``RunResult`` back.
        """
        self._cancelled = True

    # -- top level -----------------------------------------------------

    async def run(self) -> RunResult:
        state = self._load_or_start_state()
        self._started_monotonic = time.monotonic()

        while True:
            if self._cancelled:
                return self._finalize(state, stop_reason=StopReason.CANCELLED)

            outcome = await self._run_step(state)
            if outcome.stop is not None:
                return self._finalize(state, stop_reason=outcome.stop)

            action = outcome.action
            assert action is not None  # `_run_step` guarantees one of the two

            if isinstance(action, AnswerAction):
                return self._answer(state, action)

            assert isinstance(action, ToolAction)
            if self._is_no_progress(action.tool, action.arguments):
                self.journal.log_event("no_progress", tool=action.tool, step=state.step_index)
                return self._finalize(state, stop_reason=StopReason.NO_PROGRESS)

            await self._execute_tool(state, action)

    def _answer(self, state: TaskState, action: AnswerAction) -> RunResult:
        outcome = self.verifier.verify(
            action,
            history=self._executed,
            run_id=self.run_id,
            workspace=self.workspace,
            baseline=state.workspace_baseline,
        )
        if not outcome.verified:
            self.journal.log_event("verification_failed", notes=outcome.notes)
        # The answer is itself a completed step, counted the same way a
        # completed tool step is before the run's counters are read.
        state.step_index += 1
        return self._finalize(
            state, stop_reason=StopReason.ANSWERED, answer=action.content,
            verified=outcome.verified, notes=outcome.notes,
        )

    # -- one step: build, call the model, retry until a valid action -----

    async def _run_step(self, state: TaskState) -> _StepOutcome:
        if self._active_step_index == state.step_index:
            role = self.roles.role_for_step(state.step_index)
        else:
            role = self.roles.apply(self.assembler, state.step_index)
            self._active_step_index = state.step_index
        overflow_compressed = False

        while True:
            exhausted = self.budget.exhausted(
                steps=state.step_index, elapsed_s=self._elapsed(),
                tool_calls=self._tool_calls, retries=self._retries_used,
            )
            if exhausted:
                self.journal.log_event("budget_exhausted", bound=exhausted, step=state.step_index)
                return _StepOutcome(stop=StopReason.BUDGET_EXHAUSTED)

            try:
                assembled = self.assembler.build()
            except PrefixViolation as exc:
                self.journal.log_event("prefix_violation", detail=str(exc))
                return _StepOutcome(stop=StopReason.UNRECOVERABLE_ERROR)

            step_id = self.memory.append_step(
                self.run_id, step_index=state.step_index, role=str(role), action="model_call",
            )
            started = time.monotonic()
            try:
                response = await self.provider.generate(self._build_request(assembled))
            except ProviderError as exc:
                latency = (time.monotonic() - started) * 1000.0
                self.memory.complete_step(
                    step_id, status=StepStatus.FAILED,
                    error_kind=type(exc).__name__, duration_ms=latency,
                )
                self._log_model_call(role, latency, error_kind=type(exc).__name__)
                decision = self.retries.for_provider_error(exc, attempt=self._retries_used)

                if decision.category is ErrorCategory.CONTEXT_OVERFLOW and not overflow_compressed:
                    overflow_compressed = True
                    if self._try_compress(state, trigger="context_overflow"):
                        continue
                    return _StepOutcome(stop=StopReason.UNRECOVERABLE_ERROR)

                if not decision.retry:
                    return _StepOutcome(stop=decision.stop_reason or StopReason.UNRECOVERABLE_ERROR)
                self._apply_retry_feedback(decision.category, decision.feedback or "")
                continue

            self._accumulate_usage(response.usage)
            latency = (time.monotonic() - started) * 1000.0
            action = self.codec.parse(response)
            context_size = assembled.prefix_tokens + assembled.append_tokens

            if isinstance(action, ParseError):
                self.memory.complete_step(
                    step_id, status=StepStatus.FAILED,
                    error_kind="parse_error", duration_ms=latency,
                )
                self._log_model_call(
                    role, latency, response=response, context_size=context_size,
                    prefix_hash=assembled.prefix_hash, error_kind="parse_error",
                )
                decision = self.retries.for_parse_error(action, attempt=self._retries_used)
                if not decision.retry:
                    return _StepOutcome(stop=decision.stop_reason or StopReason.UNRECOVERABLE_ERROR)
                self._apply_retry_feedback(decision.category, decision.feedback or "")
                continue

            self.memory.complete_step(step_id, status=StepStatus.DONE, duration_ms=latency)
            self._log_model_call(
                role, latency, response=response, context_size=context_size,
                prefix_hash=assembled.prefix_hash,
            )
            return _StepOutcome(action=action)

    # -- tool execution --------------------------------------------------

    async def _execute_tool(self, state: TaskState, action: ToolAction) -> None:
        step_id = self.memory.append_step(
            self.run_id,
            step_index=state.step_index,
            role=str(self.roles.role_for_step(state.step_index)),
            action="tool_call",
            tool=action.tool,
            arguments=action.arguments,
        )
        result = await self.tools.invoke(action.tool, action.arguments)
        self._tool_calls += 1
        self._executed.append(
            ExecutedStep(
                id=step_id,
                run_id=self.run_id,
                step_index=state.step_index,
                tool=action.tool,
                arguments=dict(action.arguments), result=result,
            )
        )
        self.assembler.append_tool_result(result)
        self.journal.log_step(
            step=f"tool:{action.tool}", tool=action.tool, tool_ms=result.duration_ms,
            exit_code=result.exit_code, error_kind=result.error_kind,
        )
        if not result.ok:
            note = f"step {state.step_index}: {action.tool} failed: {result.content[:200]}"
            state.open_problems.append(note)
        state.step_index += 1
        state.updated_at = _now()
        self._active_step_index = None
        self._maybe_compress(state)
        self.memory.save_tool_checkpoint(
            step_id=step_id,
            status=StepStatus.DONE if result.ok else StepStatus.FAILED,
            task_state=state,
            runtime_state=self._runtime_state(),
            exit_code=result.exit_code,
            error_kind=result.error_kind,
            duration_ms=result.duration_ms,
        )

    # -- context compression ---------------------------------------------

    def _compression_state(self) -> CompressionState:
        return CompressionState(
            assembler=self.assembler, budget=self.token_budget,
            retrieve=self.retrieve, retrieval_query=self.goal,
        )

    def _try_compress(self, state: TaskState, *, trigger: str) -> bool:
        """Run the ladder and report whether it resolved the trigger.

        ``context_overflow`` is delegated to :func:`recover_overflow`, which
        climbs the ladder until the prompt is net smaller *and* back under the
        hard ceiling (issue #6) — a rung that frees tokens locally but leaves
        the prompt larger, or still over the window, has not resolved it.
        Outside overflow (``append_budget``) the ladder runs once and any
        applied rung is enough: the soft ceiling is advisory, not a hard
        failure there, and ungated rungs (retrieval) stay available.
        """
        if trigger == "context_overflow":
            return recover_overflow(
                state=self._compression_state(),
                economics=self.economics,
                ladder=self.ladder,
                journal=self.journal,
                current_step=state.step_index,
                max_steps=self.budget.max_steps,
                persist=self._persist_runtime_state,
            )
        outcomes = escalate(
            self.ladder, self._compression_state(), self.economics,
            current_step=state.step_index, max_steps=self.budget.max_steps,
            trigger=trigger,
        )
        for outcome in outcomes:
            self.journal.log_event(
                "compression", trigger=trigger, strategy=outcome.strategy,
                applied=outcome.applied, freed_tokens=outcome.freed_tokens, reason=outcome.reason,
            )
        return any(outcome.applied for outcome in outcomes)

    def _maybe_compress(self, state: TaskState) -> None:
        report = self.token_budget.report(
            self.assembler.prefix_messages(), self.assembler.append_messages,
        )
        if report.append.within_budget and not report.over_hard_ceiling:
            return
        self._try_compress(state, trigger="append_budget")

    # -- no-progress detection --------------------------------------------

    def _is_no_progress(self, tool: str, arguments: dict[str, Any]) -> bool:
        signature = (tool, json.dumps(arguments, sort_keys=True, default=str))
        self._recent_calls.append(signature)
        return (
            len(self._recent_calls) == self._recent_calls.maxlen
            and len(set(self._recent_calls)) == 1
        )

    # -- retry bookkeeping --------------------------------------------------

    def _apply_retry_feedback(self, category: ErrorCategory, feedback: str) -> None:
        self._retries_used += 1
        self.assembler.append(Message(role="user", content=f"[retry: {category}]\n{feedback}"))
        self._persist_runtime_state()

    # -- request building, usage accounting, state I/O ----------------------

    def _log_model_call(
        self, role: Role, latency_ms: float, *,
        response: Any = None, context_size: int | None = None,
        prefix_hash: str | None = None, error_kind: str | None = None,
    ) -> None:
        """Shared shape for the three ``model_call`` outcomes (provider
        error, parse error, success) — one place that can never let the
        fields logged for one drift from the others."""
        kwargs: dict[str, Any] = {"role": str(role), "latency_ms": latency_ms}
        if response is not None:
            kwargs.update(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=response.usage.cached_tokens,
                context_size=context_size, prefix_hash=prefix_hash,
            )
        if error_kind is not None:
            kwargs["error_kind"] = error_kind
            kwargs["retry"] = self._retries_used
        self.journal.log_step(step="model_call", **kwargs)

    def _build_request(self, assembled: AssembledPrompt) -> GenerationRequest:
        kwargs = self.codec.request_kwargs(self.tools.specs())
        return GenerationRequest(
            messages=assembled.messages, max_tokens=self.budget.max_output_tokens,
            prefix_token_estimate=assembled.prefix_tokens, **kwargs,
        )

    def _accumulate_usage(self, usage: Usage) -> None:
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._cached_tokens += usage.cached_tokens
        self._persist_runtime_state()

    def _elapsed(self) -> float:
        return self._elapsed_offset + time.monotonic() - self._started_monotonic

    def _load_or_start_state(self) -> TaskState:
        existing = self.memory.load_task_state(self.run_id)
        if existing is not None:
            workspace_changed = (
                Path(existing.workspace).resolve() != self.workspace.resolve()
            )
            if existing.goal != self.goal or workspace_changed:
                raise StoreError(f"run {self.run_id!r} does not match goal or workspace")
            self._restore_run_state()
            for step in self.memory.unfinished_steps(self.run_id):
                self.memory.complete_step(
                    step.id, status=StepStatus.FAILED,
                    note="orphaned: run resumed after interruption",
                )
            self.journal.log_event("run_resumed", step_index=existing.step_index)
            return existing
        now = _now()
        state = TaskState(
            run_id=self.run_id,
            goal=self.goal,
            workspace=str(self.workspace),
            workspace_baseline=capture_workspace_baseline(self.workspace),
            created_at=now,
            updated_at=now,
        )
        self.memory.start_run(state)
        self.memory.save_task_state(state)
        self._persist_runtime_state()
        self.journal.log_event("run_started", goal=self.goal)
        return state

    def _restore_run_state(self) -> None:
        runtime = self.memory.load_runtime_state(self.run_id)
        if runtime is None:
            raise StoreError(f"run {self.run_id!r} has no persisted runtime state")
        self._executed = list(runtime.executed_steps)
        self._recent_calls.extend(runtime.recent_calls)
        self.assembler.restore_append(runtime.append_history)
        self._active_step_index = runtime.active_step_index
        self._tool_calls = runtime.tool_calls
        self._retries_used = runtime.retries_used
        self._prompt_tokens = runtime.prompt_tokens
        self._completion_tokens = runtime.completion_tokens
        self._cached_tokens = runtime.cached_tokens
        self._elapsed_offset = runtime.elapsed_s

    def _persist_runtime_state(self) -> None:
        self.memory.save_runtime_state(self._runtime_state())

    def _runtime_state(self) -> RunRuntimeState:
        return RunRuntimeState(
            run_id=self.run_id,
            executed_steps=self._executed,
            recent_calls=list(self._recent_calls),
            append_history=self.assembler.append_messages,
            active_step_index=self._active_step_index,
            tool_calls=self._tool_calls,
            retries_used=self._retries_used,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            cached_tokens=self._cached_tokens,
            elapsed_s=self._elapsed(),
        )

    def _finalize(
        self, state: TaskState, *, stop_reason: StopReason,
        answer: str | None = None, verified: bool = False, notes: list[str] | None = None,
    ) -> RunResult:
        state.updated_at = _now()
        self.memory.save_task_state(state)
        self._persist_runtime_state()
        result = RunResult(
            run_id=self.run_id, stop_reason=stop_reason, answer=answer, verified=verified,
            verification_notes=notes or [], steps_taken=state.step_index,
            tool_calls=self._tool_calls, retries=self._retries_used,
            total_prompt_tokens=self._prompt_tokens,
            total_completion_tokens=self._completion_tokens,
            total_cached_tokens=self._cached_tokens,
            elapsed_s=self._elapsed(),
        )
        self.memory.finish_run(result)
        self.journal.log_event(
            "run_finished", stop_reason=str(stop_reason), steps=state.step_index,
            tool_calls=self._tool_calls, retries=self._retries_used, verified=verified,
        )
        return result
