"""The compression escalation ladder (CONTEXT.md section 5).

The nominal (cloud API) order — drop irrelevant tool output, summarize old
tool output, summarize old conversation, retrieve again, grow the window — is
reordered here by what each rung actually costs on this machine: everything
after an edit point must be reprocessed regardless of which zone it nominally
belongs to, so the cheapest rungs are the ones that touch the least of the
append zone. Growing the context window (the nominal list's last rung) is not
implemented here: it is a runtime restart, not a context-layer rewrite, and
belongs to whatever supervises the ``llama-server`` process.

Every rung is a pure function of :class:`CompressionState` until
:meth:`CompressionStrategy.apply` actually runs — :func:`escalate` always
prices a rung with :class:`~harness.context.economics.CacheEconomics` first,
per CONTEXT.md's "compression is a purchase."
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from harness.context.assembler import PromptAssembler
from harness.context.budget import TokenBudget
from harness.context.economics import CacheEconomics, CompressionDecision
from harness.core import Message
from harness.tools.compression import compress_output

# Mirrors PromptAssembler.append_retrieved's tag format exactly. Not exported
# by assembler.py, so duplicated here — classification needs to recognise a
# retrieved-context message, not just produce one.
_RETRIEVED_PREFIX = "[retrieved: "

RetrieveFn = Callable[[str], str]
"""query -> retrieved text. Injected: the FTS5 index lives behind the
Retriever interface (CONTEXT.md section 7), a separate subsystem this module
does not own or assume exists."""

SummarizeFn = Callable[[Sequence[Message]], str]
"""messages -> summary text. Injected for the same reason: producing a real
summary needs a model call, and this module has to stay usable — and
testable — without one."""


def default_summarize(messages: Sequence[Message]) -> str:
    """Deterministic fallback: extractive, not generative.

    Keeps the first message (what the span was working toward) and the last
    (its most recent outcome) verbatim, naming how much sat between. A real
    Summarizer is injected via :attr:`CompressionState.summarize`; nothing
    downstream cares which one produced the text.
    """
    if not messages:
        return "[summary: nothing to summarize]"
    if len(messages) == 1:
        return f"[summary: 1 message elided] {messages[0].content.strip()[:200]}"
    head, tail = messages[0], messages[-1]
    return (
        f"[summary: {len(messages)} messages elided] "
        f"first: {head.content.strip()[:160]} … last: {tail.content.strip()[:160]}"
    )


@dataclass(slots=True)
class CompressionState:
    """What a strategy needs: the assembler it may rewrite, the budget it
    counts tokens with, and the two capabilities this layer does not own
    itself (retrieval, summarization) — supplied by the agent loop, not
    constructed here."""

    assembler: PromptAssembler
    budget: TokenBudget
    retrieve: RetrieveFn | None = None
    retrieval_query: str = ""
    summarize: SummarizeFn = default_summarize


@runtime_checkable
class CompressionStrategy(Protocol):
    """One rung of the ladder.

    ``reprocess_tokens`` is what lets :func:`escalate` feed CacheEconomics
    correctly: the token count of the append zone *after* the point where a
    rewrite first diverges from current content — exactly what a KV cache
    forces the runtime to resend. ``gated`` marks a rung cheap enough to skip
    that pricing question entirely (see :class:`RetrieveAgain`).
    """

    name: str
    invalidates_prefix: bool
    gated: bool

    def estimate_freed_tokens(self, state: CompressionState) -> int: ...
    def reprocess_tokens(self, state: CompressionState) -> int: ...
    def apply(self, state: CompressionState) -> None: ...


def _first_diff(old: Sequence[Message], new: Sequence[Message]) -> int:
    """Index of the first differing message, or the shared length if one
    sequence is simply a prefix of the other (a pure append)."""
    limit = min(len(old), len(new))
    for i in range(limit):
        if old[i] != new[i]:
            return i
    return limit


def _reprocess_span(old: Sequence[Message], new: Sequence[Message], budget: TokenBudget) -> int:
    """Tokens of ``new`` from its first divergence from ``old`` onward — the
    span a KV cache forces the runtime to resend if this edit is applied."""
    return budget.count_messages(new[_first_diff(old, new):])


def _freed(old: Sequence[Message], new: Sequence[Message], budget: TokenBudget) -> int:
    """Tokens saved, floored at zero: an edit that adds more than it removes
    (a pure retrieval append) frees nothing, never a negative amount."""
    return max(0, budget.count_messages(old) - budget.count_messages(new))


class _AppendRewriteStrategy:
    """Shared plumbing for a strategy that replaces the append zone in place.

    Subclasses implement only :meth:`_candidate` — the replacement message
    list, or ``None`` when there is nothing to do — so the estimate a rung
    reports and the edit it makes can never drift apart; both derive from the
    same candidate.
    """

    name: str
    invalidates_prefix: bool = False
    gated: bool = True

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        raise NotImplementedError

    def _note(self, candidate: list[Message]) -> str:
        return self.name

    def estimate_freed_tokens(self, state: CompressionState) -> int:
        candidate = self._candidate(state)
        if candidate is None:
            return 0
        return _freed(state.assembler.append_messages, candidate, state.budget)

    def reprocess_tokens(self, state: CompressionState) -> int:
        candidate = self._candidate(state)
        if candidate is None:
            return 0
        return _reprocess_span(state.assembler.append_messages, candidate, state.budget)

    def apply(self, state: CompressionState) -> None:
        candidate = self._candidate(state)
        if candidate is None:
            return
        state.assembler.rewrite_append(candidate, note=self._note(candidate))


# -- the four rungs, cheapest first ----------------------------------------


@dataclass(slots=True)
class RetrieveAgain(_AppendRewriteStrategy):
    """Rung 1: fetch fresh context at the tail.

    A pure append — no existing message is touched, so there is nothing for
    CacheEconomics to price (``gated = False``): the only tokens processed
    are the ones added, the same ~0.8 s any append costs regardless of prompt
    size (CONTEXT.md section 1). It frees nothing itself; it exists so a
    later rung need not keep dragging stale retrieved context along just to
    answer the same question.
    """

    name: str = "retrieve_again"
    gated: bool = False
    source: str = "retry"

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        if state.retrieve is None or not state.retrieval_query.strip():
            return None
        text = state.retrieve(state.retrieval_query)
        if not text.strip():
            return None
        addition = Message(role="user", content=f"{_RETRIEVED_PREFIX}{self.source}]\n{text}")
        return [*state.assembler.append_messages, addition]

    def _note(self, candidate: list[Message]) -> str:
        return f"{self.name}: fetched fresh context for {self.source!r}"


@dataclass(slots=True)
class DropSupersededToolOutputs(_AppendRewriteStrategy):
    """Rung 2: drop a tool result a later call to the same tool replaced.

    The only identity a message carries once in the append zone is its tool
    name (arguments are not retained) — coarse, but conservative: this only
    ever drops a tool's *own* earlier output, keeping the most recent call
    for every name, never touching another tool's history.
    """

    name: str = "drop_superseded_tool_output"
    marker: str = "[superseded: a later call to {tool!r} replaced this output]"

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        old = state.assembler.append_messages
        last_index: dict[str, int] = {}
        for i, msg in enumerate(old):
            if msg.role == "tool" and msg.name:
                last_index[msg.name] = i

        changed = False
        new: list[Message] = []
        for i, msg in enumerate(old):
            if msg.role == "tool" and msg.name and last_index.get(msg.name) != i:
                new.append(Message(
                    role="tool", content=self.marker.format(tool=msg.name),
                    name=msg.name, tool_call_id=msg.tool_call_id,
                ))
                changed = True
            else:
                new.append(msg)
        return new if changed else None

    def _note(self, candidate: list[Message]) -> str:
        return f"{self.name}: dropped superseded tool output"


OLD_TOOL_OUTPUT_KEEP_RECENT = 3
"""Most-recent tool results left untouched — exactly what the next step is
likely to act on, so never a summarization candidate."""

SUMMARY_MAX_LINES = 12
SUMMARY_MAX_BYTES = 800
"""Recompression budget for a tool result CONTEXT.md section 6 already
compressed once. Tighter than tools.compression's own defaults on purpose:
this is a second pass on content the run has already moved past."""


@dataclass(slots=True)
class SummarizeOldToolOutputs(_AppendRewriteStrategy):
    """Rung 3: recompress tool results that have scrolled past the recent
    horizon.

    Same mechanism as rung 2 —
    :func:`~harness.tools.compression.compress_output` is the identical pass
    every tool result already went through once, aimed at a tighter budget —
    but on a broader, more conservative region: a result nothing later
    superseded is kept, just smaller.
    """

    name: str = "summarize_old_tool_output"
    keep_recent: int = OLD_TOOL_OUTPUT_KEEP_RECENT
    max_lines: int = SUMMARY_MAX_LINES
    max_bytes: int = SUMMARY_MAX_BYTES

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        old = state.assembler.append_messages
        tool_idx = [i for i, m in enumerate(old) if m.role == "tool"]
        stale = set(tool_idx[: max(0, len(tool_idx) - self.keep_recent)])
        if not stale:
            return None

        changed = False
        new: list[Message] = []
        for i, msg in enumerate(old):
            if i in stale and len(msg.content.encode("utf-8")) > self.max_bytes:
                recompressed = compress_output(
                    msg.content, max_lines=self.max_lines, max_bytes=self.max_bytes
                )
                if recompressed.truncated:
                    new.append(Message(
                        role="tool", content=recompressed.text,
                        name=msg.name, tool_call_id=msg.tool_call_id,
                    ))
                    changed = True
                    continue
            new.append(msg)
        return new if changed else None

    def _note(self, candidate: list[Message]) -> str:
        return f"{self.name}: recompressed old tool output"


OLD_CONVERSATION_KEEP_RECENT_MESSAGES = 6
"""Verbatim tail left untouched — recent steps are what the next action
depends on."""

MIN_OLD_CONVERSATION_MESSAGES = 4
"""Below this span, summarizing buys back too little to be worth a rewrite
that reprocesses the whole append zone — the most expensive rung in it
(CONTEXT.md section 5)."""


@dataclass(slots=True)
class SummarizeOldConversation(_AppendRewriteStrategy):
    """Rung 4: collapse the oldest span of the append zone into one summary
    message.

    The most expensive in-append-zone edit: by construction it targets the
    content closest to the prefix boundary, so applying it forces the
    largest possible span to be reprocessed (CONTEXT.md section 5). Reached
    last, and only when the span is long enough to be worth that cost.
    """

    name: str = "summarize_old_conversation"
    keep_recent: int = OLD_CONVERSATION_KEEP_RECENT_MESSAGES
    min_span: int = MIN_OLD_CONVERSATION_MESSAGES

    def _candidate(self, state: CompressionState) -> list[Message] | None:
        old = state.assembler.append_messages
        span = old[: max(0, len(old) - self.keep_recent)]
        if len(span) < self.min_span:
            return None
        summary_text = state.summarize(span)
        summary = Message(
            role="user", content=f"[summary: {len(span)} earlier messages]\n{summary_text}"
        )
        return [summary, *old[len(span):]]

    def _note(self, candidate: list[Message]) -> str:
        return f"{self.name}: collapsed old conversation"


def default_ladder() -> list[CompressionStrategy]:
    """The escalation ladder in cache-cost order (CONTEXT.md section 5)."""
    return [
        RetrieveAgain(),
        DropSupersededToolOutputs(),
        SummarizeOldToolOutputs(),
        SummarizeOldConversation(),
    ]


class StrategyOutcome(BaseModel):
    """One rung's numbers and verdict, applied or not — for the run journal.

    Recorded either way: refusing every rung is as informative as accepting
    one, and ``decision`` is what makes that refusal auditable rather than
    asserted.
    """

    strategy: str
    freed_tokens: int
    reprocess_tokens: int
    applied: bool
    resolves_overflow: bool
    reason: str
    decision: CompressionDecision | None = None


def escalate(
    strategies: Sequence[CompressionStrategy],
    state: CompressionState,
    economics: CacheEconomics,
    *,
    current_step: int,
    max_steps: int,
    trigger: str = "",
) -> list[StrategyOutcome]:
    """Walk the ladder in order, applying the first rung that pays for
    itself, and stop there.

    An ungated rung (:class:`RetrieveAgain`) is tried unconditionally and
    never stops the walk — it has nothing to weigh against reprocess cost, so
    it does not compete with the rungs that do. A gated rung goes through
    :class:`~harness.context.economics.CacheEconomics` before
    :meth:`CompressionStrategy.apply` is ever called; the walk stops at the
    first one whose numbers clear the bar.

    ``trigger="context_overflow"`` skips ungated rungs: a pure append adds
    tokens, so it cannot relieve an overflow — running it there would only
    grow the prompt and mask whether recovery actually happened (issue #6).
    Outside that trigger (e.g. ``append_budget``) ungated rungs stay available.
    """
    overflow = trigger == "context_overflow"
    outcomes: list[StrategyOutcome] = []
    for strategy in strategies:
        freed = strategy.estimate_freed_tokens(state)
        reprocess = strategy.reprocess_tokens(state)

        if not strategy.gated:
            if overflow:
                outcomes.append(StrategyOutcome(
                    strategy=strategy.name, freed_tokens=freed, reprocess_tokens=reprocess,
                    applied=False, resolves_overflow=False,
                    reason="skipped at context_overflow: pure append cannot relieve overflow",
                ))
                continue
            before = state.assembler.append_messages
            strategy.apply(state)
            applied = state.assembler.append_messages != before
            outcomes.append(StrategyOutcome(
                strategy=strategy.name, freed_tokens=freed, reprocess_tokens=reprocess,
                applied=applied,
                resolves_overflow=applied and freed > 0,
                reason="ungated: pure tail append" if applied else "nothing to do",
            ))
            continue

        if freed <= 0:
            outcomes.append(StrategyOutcome(
                strategy=strategy.name, freed_tokens=freed, reprocess_tokens=reprocess,
                applied=False, resolves_overflow=False, reason="nothing to do",
            ))
            continue

        context_tokens = state.budget.count_messages(
            [*state.assembler.prefix_messages(), *state.assembler.append_messages]
        )
        decision = economics.evaluate(
            freed_tokens=freed, new_prefix_tokens=reprocess,
            current_step=current_step, max_steps=max_steps, context_tokens=context_tokens,
        )
        if decision.compress:
            strategy.apply(state)
        outcomes.append(StrategyOutcome(
            strategy=strategy.name, freed_tokens=freed, reprocess_tokens=reprocess,
            applied=decision.compress,
            resolves_overflow=decision.compress and freed > 0,
            reason=decision.rationale,
            decision=decision,
        ))
        if decision.compress:
            break
    return outcomes
