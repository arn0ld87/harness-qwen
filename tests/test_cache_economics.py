"""The compression cost model against the DISCOVERY §5 worked example (issue #22).

The numbers in CONTEXT.md §5 are not illustrative — they are the measured
pp-rate profile applied to a concrete step. The decision the model returns has
to match the decision the document reaches, or the model and the document have
drifted apart and one of them is wrong.
"""

from __future__ import annotations

from harness.context.economics import CacheEconomics, PpRateProfile


def test_the_measured_pp_rate_at_18k_is_120_tok_s() -> None:
    profile = PpRateProfile.measured()
    assert profile.rate_at(18_000) == 120.0


def test_step_20_of_40_amortises_so_compress() -> None:
    """18k context, 4000 freed, 14000 reprocessed, 20 steps left → compress."""
    eco = CacheEconomics()  # default safety_margin 1.5, measured profile
    decision = eco.evaluate(
        freed_tokens=4_000,
        new_prefix_tokens=14_000,
        current_step=20,
        max_steps=40,
        context_tokens=18_000,
    )
    assert decision.compress is True
    assert decision.remaining_steps == 20
    # The arithmetic the document walks through, to the precision that matters:
    # 14000/120 ≈ 116.7 s reprocess, 4000/120 ≈ 33.3 s/step, 33.3*20 = 666.7 s
    # projected against 116.7*1.5 = 175.0 s threshold.
    assert round(decision.reprocess_cost_s, 1) == 116.7
    assert round(decision.saving_per_step_s, 1) == 33.3
    assert round(decision.projected_saving_s, 1) == 666.7
    assert round(decision.threshold_s, 1) == 175.0


def test_step_38_does_not_amortise_so_refuse() -> None:
    """Same candidate, only 2 steps left — the run finishes on current context."""
    eco = CacheEconomics()
    decision = eco.evaluate(
        freed_tokens=4_000,
        new_prefix_tokens=14_000,
        current_step=38,
        max_steps=40,
        context_tokens=18_000,
    )
    assert decision.compress is False
    assert decision.remaining_steps == 2
    assert "does not amortise" in decision.rationale


def test_freed_tokens_of_zero_is_refused_regardless_of_steps() -> None:
    eco = CacheEconomics()
    decision = eco.evaluate(
        freed_tokens=0,
        new_prefix_tokens=14_000,
        current_step=20,
        max_steps=40,
        context_tokens=18_000,
    )
    assert decision.compress is False
    assert "frees nothing" in decision.rationale


def test_fifty_seconds_at_8k_is_about_825_discarded_output_tokens() -> None:
    """The §1 figure: a 50 s stall at the single-slot generation rate costs the
    run ~825 tokens of output it did not get to produce."""
    eco = CacheEconomics()
    assert eco.discarded_output_tokens(50.0) == 825


def test_reprocess_cost_uses_the_interpolated_rate_not_a_constant() -> None:
    """pp-rate falls as context grows (DISCOVERY §1); the cost model must read
    it at the current size, not assume the cold-probe 194 tok/s everywhere."""
    eco = CacheEconomics()
    cheap = eco.reprocess_cost(1_000, context_tokens=2_000)
    dear = eco.reprocess_cost(1_000, context_tokens=18_000)
    assert dear > cheap