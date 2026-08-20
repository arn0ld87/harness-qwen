"""Cost model deciding whether a compression pass pays for itself.

Compression is a purchase (CONTEXT.md §5). Rewriting content forces every
token after the edit point to be reprocessed, and prompt-processing throughput
degrades as context grows, so the same rewrite gets more expensive the longer
a run lasts. The decision is arithmetic on measured rates, and the arithmetic
is returned alongside it so the journal records why a run did or did not pay.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

GENERATION_RATE_TOK_S = 16.5
"""Measured single-slot generation rate (DISCOVERY.md). Used to express a
reprocess cost as the output tokens it displaces — a second slot does not add
capacity here, it halves throughput, so there is nothing to hide the cost
behind."""

DEFAULT_SAFETY_MARGIN = 1.5
"""PLACEHOLDER (CONTEXT.md §5). A compression must beat its own cost by this
factor before it is worth the risk of a rewrite."""

MIN_PP_RATE_TOK_S = 20.0
"""Floor for extrapolation past the last measured point. Linear continuation
of the measured slope crosses zero around 25k tokens, which is nonsense; the
floor keeps extrapolation pessimistic without becoming absurd."""

COLD_PROBE_PP_RATE_TOK_S = 194.0
"""The cold 4816-token probe (DISCOVERY.md). Deliberately not a point on the
interpolation curve: a single cold reprocess measures something different from
steady-state throughput at that size, and mixing the two makes the curve
non-monotonic (194 tok/s at 4.8k against ~170 tok/s across the 2k-6k band)."""

MEASURED_PP_PROFILE: tuple[tuple[int, float], ...] = (
    (2000, 170.0),
    (6000, 170.0),
    (16000, 154.0),
    (18000, 120.0),
)
"""Steady-state prompt-processing rate by context size (DISCOVERY.md §1)."""


class PpPoint(BaseModel):
    """One measured (context size, prompt-processing rate) observation."""

    context_tokens: int = Field(ge=1)
    tokens_per_second: float = Field(gt=0.0)


class PpRateProfile(BaseModel):
    """Prompt-processing rate as a function of context size.

    Between measured points the rate is linearly interpolated. Below the first
    point it is clamped rather than extrapolated upward: predicting a rate
    faster than anything measured is the optimistic direction, and an
    optimistic pp_rate makes compression look cheaper than it is.
    """

    points: list[PpPoint]
    min_rate: float = Field(default=MIN_PP_RATE_TOK_S, gt=0.0)

    @field_validator("points")
    @classmethod
    def _non_empty(cls, points: list[PpPoint]) -> list[PpPoint]:
        if not points:
            raise ValueError("a pp-rate profile needs at least one measured point")
        return sorted(points, key=lambda p: p.context_tokens)

    @model_validator(mode="after")
    def _distinct_sizes(self) -> PpRateProfile:
        sizes = [p.context_tokens for p in self.points]
        if len(set(sizes)) != len(sizes):
            raise ValueError("pp-rate profile has two rates for one context size")
        return self

    @classmethod
    def measured(cls) -> PpRateProfile:
        """The profile measured on this machine (DISCOVERY.md §1)."""
        return cls(
            points=[
                PpPoint(context_tokens=size, tokens_per_second=rate)
                for size, rate in MEASURED_PP_PROFILE
            ]
        )

    def rate_at(self, context_tokens: int) -> float:
        """Prompt-processing rate at ``context_tokens``, in tokens per second."""
        if context_tokens < 0:
            raise ValueError("context_tokens must not be negative")
        points = self.points
        if context_tokens <= points[0].context_tokens or len(points) == 1:
            return points[0].tokens_per_second

        for lower, upper in zip(points, points[1:], strict=False):
            if context_tokens <= upper.context_tokens:
                span = upper.context_tokens - lower.context_tokens
                travelled = context_tokens - lower.context_tokens
                delta = upper.tokens_per_second - lower.tokens_per_second
                return lower.tokens_per_second + delta * (travelled / span)

        return self._extrapolate(context_tokens)

    def _extrapolate(self, context_tokens: int) -> float:
        """Continue the last measured slope downward, floored at ``min_rate``.

        Extrapolation only ever makes the rate worse: past the last measured
        point the run is in territory where the only evidence is that
        throughput keeps falling.
        """
        last = self.points[-1]
        if len(self.points) == 1:
            return max(self.min_rate, last.tokens_per_second)
        previous = self.points[-2]
        span = last.context_tokens - previous.context_tokens
        slope = (last.tokens_per_second - previous.tokens_per_second) / span
        projected = last.tokens_per_second + slope * (context_tokens - last.context_tokens)
        return max(self.min_rate, min(projected, last.tokens_per_second))


class CompressionDecision(BaseModel):
    """A compression verdict together with the arithmetic behind it."""

    compress: bool
    rationale: str
    pp_rate: float
    context_tokens: int
    freed_tokens: int
    new_prefix_tokens: int
    reprocess_cost_s: float
    saving_per_step_s: float
    remaining_steps: int
    projected_saving_s: float
    threshold_s: float
    safety_margin: float
    discarded_output_tokens: int
    """Generation the reprocess displaces, at the measured single-slot rate."""


class CacheEconomics:
    """Decides whether freeing tokens is worth the reprocess it triggers."""

    def __init__(
        self,
        profile: PpRateProfile | None = None,
        *,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        generation_rate: float = GENERATION_RATE_TOK_S,
    ) -> None:
        if safety_margin <= 0:
            raise ValueError("safety_margin must be positive")
        if generation_rate <= 0:
            raise ValueError("generation_rate must be positive")
        self.profile = profile or PpRateProfile.measured()
        self.safety_margin = safety_margin
        self.generation_rate = generation_rate

    def pp_rate(self, context_tokens: int) -> float:
        return self.profile.rate_at(context_tokens)

    def reprocess_cost(self, new_prefix_tokens: int, *, context_tokens: int) -> float:
        """Seconds to reprocess everything after the edit point."""
        return max(0, new_prefix_tokens) / self.pp_rate(context_tokens)

    def saving_per_step(self, freed_tokens: int, *, context_tokens: int) -> float:
        """Seconds saved per later step, only meaningful if the span shrinks."""
        return max(0, freed_tokens) / self.pp_rate(context_tokens)

    def discarded_output_tokens(self, seconds: float) -> int:
        """Output tokens a stall of ``seconds`` costs the run.

        50 s at 8k context is ~825 tokens of real model output the run does not
        get to produce (CONTEXT.md §1). Wall clock is abstract; this is not.
        """
        return int(seconds * self.generation_rate)

    def evaluate(
        self,
        *,
        freed_tokens: int,
        new_prefix_tokens: int,
        current_step: int,
        max_steps: int,
        context_tokens: int,
    ) -> CompressionDecision:
        """Full cost model for one candidate compression."""
        remaining_steps = max(0, max_steps - current_step)
        rate = self.pp_rate(context_tokens)
        reprocess_cost = self.reprocess_cost(new_prefix_tokens, context_tokens=context_tokens)
        saving = self.saving_per_step(freed_tokens, context_tokens=context_tokens)
        projected = saving * remaining_steps
        threshold = reprocess_cost * self.safety_margin
        compress = projected > threshold and freed_tokens > 0 and remaining_steps > 0

        arithmetic = (
            f"pp_rate={rate:.1f} tok/s at {context_tokens} tok; "
            f"reprocess_cost={max(0, new_prefix_tokens)}/{rate:.1f}={reprocess_cost:.1f} s; "
            f"saving_per_step={max(0, freed_tokens)}/{rate:.1f}={saving:.1f} s; "
            f"remaining_steps={remaining_steps}; "
            f"projected={projected:.1f} s vs "
            f"threshold={reprocess_cost:.1f}*{self.safety_margin:.2f}={threshold:.1f} s"
        )
        if freed_tokens <= 0:
            verdict = "refuse: frees nothing"
        elif remaining_steps <= 0:
            verdict = "refuse: no steps left to amortise over"
        elif compress:
            verdict = "compress: amortises"
        else:
            verdict = "refuse: does not amortise"

        return CompressionDecision(
            compress=compress,
            rationale=f"{verdict} | {arithmetic}",
            pp_rate=rate,
            context_tokens=context_tokens,
            freed_tokens=freed_tokens,
            new_prefix_tokens=new_prefix_tokens,
            reprocess_cost_s=reprocess_cost,
            saving_per_step_s=saving,
            remaining_steps=remaining_steps,
            projected_saving_s=projected,
            threshold_s=threshold,
            safety_margin=self.safety_margin,
            discarded_output_tokens=self.discarded_output_tokens(reprocess_cost),
        )

    def should_compress(
        self,
        *,
        freed_tokens: int,
        new_prefix_tokens: int,
        current_step: int,
        max_steps: int,
        context_tokens: int,
    ) -> tuple[bool, str]:
        """Decision plus the arithmetic that produced it, for the journal."""
        decision = self.evaluate(
            freed_tokens=freed_tokens,
            new_prefix_tokens=new_prefix_tokens,
            current_step=current_step,
            max_steps=max_steps,
            context_tokens=context_tokens,
        )
        return decision.compress, decision.rationale
