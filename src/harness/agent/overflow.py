"""Context-overflow recovery (issue #6).

The compression ladder's per-rung ``resolves_overflow`` only proves a single
rung freed tokens locally — not that the prompt is net smaller, or back under
the hard ceiling, after the edit. Recovery is decided here by the token count
the budget actually sees: the ladder is climbed until the prompt shrinks *and*
fits the window again. Ungated rungs (pure appends such as ``RetrieveAgain``)
are skipped on this path — a retrieval adds tokens and so cannot relieve an
overflow; running it there would only mask whether recovery happened.

Extracted from ``agent.loop`` so that module stays under the ~500-line limit
(AGENTS.md "Code style"): overflow recovery is one decision, owned here. The
loop supplies the fully wired :class:`CompressionState` (retrieval and
summarize callables included), so this module owns neither capability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from harness.context.compressor import CompressionState, CompressionStrategy, escalate
from harness.context.economics import CacheEconomics
from harness.telemetry.journal import RunJournal


def recover_overflow(
    *,
    state: CompressionState,
    economics: CacheEconomics,
    ladder: Sequence[CompressionStrategy],
    journal: RunJournal,
    current_step: int,
    max_steps: int,
    persist: Callable[[], None],
) -> bool:
    """Climb the ladder until the prompt is net smaller and within the hard
    ceiling. Returns ``True`` only when the overflow is genuinely resolved.

    Each iteration applies at most one paying rung, then re-measures. The loop
    terminates because every applied rung strictly shrinks the prompt, so the
    token count is a non-negative, strictly decreasing bound: it either lands
    under the ceiling (``True``), stops shrinking (``False``), or no rung
    applies (``False``). Before/after counts are logged every iteration so the
    measurement is auditable, not just the terminal verdict.
    """
    assembler = state.assembler
    budget = state.budget
    before = budget.count_messages(
        [*assembler.prefix_messages(), *assembler.append_messages]
    )
    while True:
        outcomes = escalate(
            ladder, state, economics,
            current_step=current_step, max_steps=max_steps,
            trigger="context_overflow",
        )
        for outcome in outcomes:
            journal.log_event(
                "compression", trigger="context_overflow",
                strategy=outcome.strategy, applied=outcome.applied,
                freed_tokens=outcome.freed_tokens, reason=outcome.reason,
            )
        if not any(o.applied for o in outcomes):
            journal.log_event(
                "overflow_recovery", resolved=False, before_tokens=before,
                reason="no rung applied",
            )
            persist()
            return False
        after = budget.count_messages(
            [*assembler.prefix_messages(), *assembler.append_messages]
        )
        if after >= before:
            journal.log_event(
                "overflow_recovery", resolved=False,
                before_tokens=before, after_tokens=after,
                reason="net-negative: recovery did not shrink the prompt",
            )
            persist()
            return False
        report = budget.report(assembler.prefix_messages(), assembler.append_messages)
        if not report.over_hard_ceiling:
            journal.log_event(
                "overflow_recovery", resolved=True,
                before_tokens=before, after_tokens=after,
            )
            persist()
            return True
        # Net progress made but still over the hard ceiling: keep climbing.
        journal.log_event(
            "overflow_recovery", resolved=False, pending=True,
            before_tokens=before, after_tokens=after,
            reason="still over hard ceiling; climbing again",
        )
        before = after