"""Bounded, input-changing retries (ARCHITECTURE.md, "Action protocol").

This module governs only the two ways a step can fail to produce a usable
*action*: the response didn't parse, or the provider itself failed to answer.
A tool call that runs and returns ``ok=False`` is not retried here — its
result is simply the outcome of the step, fed back into context the same way
a successful result would be (``PromptAssembler.append_tool_result``), and
the model chooses its next move on the next step. What this module bounds is
re-asking the *same* step for a valid action, which the action protocol
requires to change the input every time it happens: repair once from the
parse error, then re-ask with that error as feedback, counted against
``core.Budget.max_retries``. A retry that cannot change the input is not
worth spending budget on and is refused instead.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from harness.core import (
    ContextOverflow,
    ParseError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    StopReason,
)


class ErrorCategory(enum.StrEnum):
    """What kind of failure a :class:`RetryDecision` was classifying."""

    PARSE_FAILURE = "parse_failure"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    CONTEXT_OVERFLOW = "context_overflow"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Whether to retry, the feedback that makes the retry a different
    request, and why.

    ``feedback`` is only ever set alongside ``retry=True`` — it is the text
    the loop appends to context before the next attempt. ``detail`` is set on
    every decision and is what the loop should record as the reason when it
    stops the run. ``stop_reason`` is set only alongside ``retry=False`` and
    tells the loop *which* ``core.StopReason`` applies: running out of the
    shared retry budget is ``BUDGET_EXHAUSTED``, a category this policy
    considers unrecoverable on its own terms (an unavailable provider, an
    overflow no retry can fix, a parse error marked non-recoverable) is
    ``UNRECOVERABLE_ERROR``. Deciding this here, rather than letting the loop
    re-derive it from ``category`` alone, keeps the two refusal reasons from
    drifting apart as new categories are added.
    """

    category: ErrorCategory
    retry: bool
    feedback: str | None = None
    detail: str = ""
    stop_reason: StopReason | None = None


class RetryPolicy:
    """Classifies a step failure and decides whether another attempt is
    warranted, bounded by a single run-wide retry budget.

    ``max_retries`` is cumulative across the whole run, matching
    ``core.Budget.max_retries`` — not a per-step allowance. A step that fails
    once and a step that fails three times draw from the same pool: a run
    that keeps needing repair is not making progress, regardless of which
    step is doing the failing.

    ``ContextOverflow`` is classified but never made retryable here: fixing
    it means freeing tokens, which only the context subsystem's compression
    ladder can do, and that ladder is not this policy's to own. The loop
    decides, using this classification as the trigger, whether a compression
    pass changed the input enough to justify trying again — a retry this
    policy did not authorise on its own, but one the compression makes
    legitimate by actually changing what gets sent.
    """

    def __init__(self, max_retries: int) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.max_retries = max_retries

    def _budget_left(self, attempt: int) -> bool:
        return attempt < self.max_retries

    def for_parse_error(self, error: ParseError, *, attempt: int) -> RetryDecision:
        """Decide whether to re-ask after a response that did not parse."""
        if not error.recoverable:
            return RetryDecision(
                ErrorCategory.PARSE_FAILURE, retry=False,
                detail=f"parse error marked unrecoverable: {error.reason}",
                stop_reason=StopReason.UNRECOVERABLE_ERROR,
            )
        if not self._budget_left(attempt):
            detail = f"retry budget exhausted ({attempt}/{self.max_retries}) after: {error.reason}"
            return RetryDecision(
                ErrorCategory.PARSE_FAILURE, retry=False,
                detail=detail, stop_reason=StopReason.BUDGET_EXHAUSTED,
            )
        return RetryDecision(
            ErrorCategory.PARSE_FAILURE, retry=True,
            feedback=error.as_feedback(), detail=error.reason,
        )

    def for_provider_error(self, exc: ProviderError, *, attempt: int) -> RetryDecision:
        """Decide whether to retry a provider-level failure.

        ``ProviderUnavailable`` is the run's unrecoverable end of the
        provider interface: an endpoint that is not answering will not start
        answering because the loop asked twice. ``ContextOverflow`` is
        reported as not retryable *by this method* — see the class
        docstring for what the loop does with that classification instead.
        Everything else — chiefly ``ProviderTimeout`` — is worth one more
        attempt inside the budget, with the timeout itself phrased as the
        feedback that makes the retry a different request.
        """
        if isinstance(exc, ProviderUnavailable):
            return RetryDecision(
                ErrorCategory.UNAVAILABLE, retry=False,
                detail=f"provider unavailable: {exc}",
                stop_reason=StopReason.UNRECOVERABLE_ERROR,
            )
        if isinstance(exc, ContextOverflow):
            return RetryDecision(
                ErrorCategory.CONTEXT_OVERFLOW, retry=False,
                detail=f"context overflow: {exc}",
                stop_reason=StopReason.UNRECOVERABLE_ERROR,
            )
        if not self._budget_left(attempt):
            return RetryDecision(
                _category_of(exc), retry=False,
                detail=f"retry budget exhausted ({attempt}/{self.max_retries}) after: {exc}",
                stop_reason=StopReason.BUDGET_EXHAUSTED,
            )
        if isinstance(exc, ProviderTimeout):
            feedback = f"Your previous response timed out ({exc}). Reply with one complete action."
            return RetryDecision(
                ErrorCategory.TIMEOUT, retry=True, feedback=feedback, detail=str(exc),
            )
        # An unrecognised ProviderError subclass: retrying blind is not a
        # safe default, so this end of the classification is closed rather
        # than left open to whatever subclass shows up next.
        return RetryDecision(
            ErrorCategory.UNKNOWN, retry=False, detail=f"unrecognised provider error: {exc}",
            stop_reason=StopReason.UNRECOVERABLE_ERROR,
        )


def _category_of(exc: ProviderError) -> ErrorCategory:
    return ErrorCategory.TIMEOUT if isinstance(exc, ProviderTimeout) else ErrorCategory.UNKNOWN
