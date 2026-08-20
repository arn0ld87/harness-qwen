"""Token accounting for the two prompt zones plus generation headroom.

The window is split ``C = P + A + G`` (CONTEXT.md §3): the stable prefix ``P``
is measured, never chosen as a fraction, because its size is dictated by the
tool set and repository map. Only ``G`` and the soft ceiling on ``A`` are
levers.

Counting is an estimate unless a provider tokenizer is supplied. The estimate
is deliberately visible in the report (``exact``) so nobody reads a budget
decision as a measurement when it was arithmetic on character counts.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from harness.core import Message

CHARS_PER_TOKEN = 3.6
"""Characters per token for the heuristic estimate.

Mixed English prose and source code on a byte-level BPE vocabulary lands
between 3.3 and 4.0 chars/token; 3.6 sits in the middle. It is an estimate,
good to roughly ±15%, and must never be presented as a token count."""

PER_MESSAGE_OVERHEAD_TOKENS = 4
"""Chat-template scaffolding per message (role marker plus delimiters).

Ignoring it under-counts a long step history by hundreds of tokens, which is
exactly the direction that silently overflows a window."""

DEFAULT_CONTEXT_WINDOW = 65536
"""``--ctx-size`` from the current launch flags: a configured maximum, not a
validated working target (CONTEXT.md §3)."""

DEFAULT_SOFT_CEILING = 16000
"""PROVISIONAL, pending the flag sweep. Last point in the measured pp-rate
profile before the fall to ~120 tok/s past 18k — a hint about where processing
cost accelerates, not a benchmarked optimum."""

DEFAULT_GENERATION_RESERVE_RATIO = 0.15
"""PROVISIONAL, unmeasured. Conservative starting fraction of the soft
ceiling withheld from input growth."""

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` from its character count.

    This is a heuristic, not a measurement: it never sees the model's
    vocabulary. Use it for budget headroom and compression sizing, never to
    decide whether a request fits a hard context limit — for that, pass a
    provider tokenizer to :class:`TokenBudget`.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class Zone(enum.StrEnum):
    PREFIX = "prefix"
    APPEND = "append"
    GENERATION = "generation"


class ZoneUsage(BaseModel):
    zone: Zone
    tokens: int
    limit: int

    @property
    def over_by(self) -> int:
        return max(0, self.tokens - self.limit)

    @property
    def within_budget(self) -> bool:
        return self.tokens <= self.limit


class BudgetReport(BaseModel):
    """Per-zone usage of one assembled prompt."""

    prefix: ZoneUsage
    append: ZoneUsage
    generation: ZoneUsage
    soft_ceiling: int
    context_window: int
    exact: bool
    """False when the counts came from :func:`estimate_tokens`."""

    @property
    def zones(self) -> tuple[ZoneUsage, ZoneUsage, ZoneUsage]:
        return (self.prefix, self.append, self.generation)

    @property
    def input_tokens(self) -> int:
        return self.prefix.tokens + self.append.tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.generation.tokens

    @property
    def over_budget_zone(self) -> Zone | None:
        """The zone to act on, or None.

        The prefix is reported first when both are over: an oversized prefix
        leaves no room to allocate and no compression of the append zone can
        fix it.
        """
        for usage in self.zones:
            if not usage.within_budget:
                return usage.zone
        return None

    @property
    def over_hard_ceiling(self) -> bool:
        """True when the prompt plus its reserve does not fit the window."""
        return self.total_tokens > self.context_window

    @property
    def append_headroom(self) -> int:
        return max(0, self.append.limit - self.append.tokens)

    def summary(self) -> str:
        """One-line form for the run journal."""
        zones = " ".join(f"{u.zone}={u.tokens}/{u.limit}" for u in self.zones)
        kind = "exact" if self.exact else "estimated"
        over = self.over_budget_zone or "none"
        return f"{zones} ceiling={self.soft_ceiling} over={over} ({kind})"


class TokenBudget:
    """Allocates the context window across prefix, append zone and generation.

    ``token_counter`` is the provider's exact tokenizer when one is available.
    Without it every count is the documented character heuristic.
    """

    def __init__(
        self,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        soft_ceiling: int = DEFAULT_SOFT_CEILING,
        generation_reserve_ratio: float = DEFAULT_GENERATION_RESERVE_RATIO,
        generation_reserve: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if soft_ceiling <= 0 or soft_ceiling > context_window:
            raise ValueError("soft_ceiling must be in (0, context_window]")
        if generation_reserve is None:
            if not 0.0 < generation_reserve_ratio < 1.0:
                raise ValueError("generation_reserve_ratio must be in (0, 1)")
            generation_reserve = int(soft_ceiling * generation_reserve_ratio)
        if not 0 < generation_reserve < soft_ceiling:
            raise ValueError("generation_reserve must be in (0, soft_ceiling)")

        self.context_window = context_window
        self.soft_ceiling = soft_ceiling
        self.generation_reserve = generation_reserve
        self._counter = token_counter

    @property
    def exact(self) -> bool:
        return self._counter is not None

    def count_text(self, text: str) -> int:
        if self._counter is not None:
            return self._counter(text)
        return estimate_tokens(text)

    def count_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for message in messages:
            total += self.count_text(message.content) + PER_MESSAGE_OVERHEAD_TOKENS
            if message.name:
                total += self.count_text(message.name)
        return total

    def append_limit(self, prefix_tokens: int) -> int:
        """``A = C - P - G``, floored at zero when the prefix already ate it."""
        return max(0, self.soft_ceiling - prefix_tokens - self.generation_reserve)

    def prefix_limit(self) -> int:
        """The largest prefix that still leaves room to generate.

        A prefix at this limit leaves an append zone of zero, so this is an
        upper bound on the pathological case, not a target.
        """
        return self.soft_ceiling - self.generation_reserve

    def report(
        self,
        prefix_messages: Sequence[Message],
        append_messages: Sequence[Message],
        *,
        generation_tokens: int | None = None,
    ) -> BudgetReport:
        """Report usage per zone for one assembled prompt."""
        prefix_tokens = self.count_messages(prefix_messages)
        append_tokens = self.count_messages(append_messages)
        return self.report_tokens(
            prefix_tokens=prefix_tokens,
            append_tokens=append_tokens,
            generation_tokens=generation_tokens,
        )

    def report_tokens(
        self,
        *,
        prefix_tokens: int,
        append_tokens: int,
        generation_tokens: int | None = None,
    ) -> BudgetReport:
        """Report usage from counts already taken elsewhere."""
        planned_generation = (
            self.generation_reserve if generation_tokens is None else generation_tokens
        )
        return BudgetReport(
            prefix=ZoneUsage(
                zone=Zone.PREFIX, tokens=prefix_tokens, limit=self.prefix_limit()
            ),
            append=ZoneUsage(
                zone=Zone.APPEND,
                tokens=append_tokens,
                limit=self.append_limit(prefix_tokens),
            ),
            generation=ZoneUsage(
                zone=Zone.GENERATION,
                tokens=planned_generation,
                limit=self.generation_reserve,
            ),
            soft_ceiling=self.soft_ceiling,
            context_window=self.context_window,
            exact=self.exact,
        )

    def would_exceed(self, *, prefix_tokens: int, append_tokens: int, addition: int) -> bool:
        """Whether appending ``addition`` tokens pushes past the soft ceiling.

        The trigger for a :mod:`harness.context.economics` check, per
        CONTEXT.md §3 — not a licence to compress on its own.
        """
        return append_tokens + addition > self.append_limit(prefix_tokens)
