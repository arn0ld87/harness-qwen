"""Summary statistics for benchmark samples (issue #19).

The properties asserted here are the ones that decide whether a published
number is honest: a mean hides the 25 s cold prompt inside a row of 0.5 s
cached ones, and silently dropping the outlier that produced it turns a
measurement into an opinion.
"""

from __future__ import annotations

import math

import pytest

from harness.benchmark.stats import (
    distribution,
    mad,
    median,
    outlier_indices,
    percentile,
)

# -- percentiles -----------------------------------------------------------


def test_percentile_interpolates_between_ranks() -> None:
    values = [float(n) for n in range(1, 11)]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 90) == pytest.approx(9.1)


def test_percentile_endpoints_are_the_extremes() -> None:
    values = [3.0, 1.0, 2.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 3.0


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    assert percentile([7.5], 99) == 7.5


def test_percentile_refuses_an_empty_series() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_refuses_a_rank_outside_the_range() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_median_of_an_even_series_averages_the_middle_pair() -> None:
    assert median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


# -- dispersion and outliers -----------------------------------------------


def test_mad_is_the_median_absolute_deviation() -> None:
    assert mad([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(1.0)


def test_a_cold_prompt_among_cached_ones_is_flagged() -> None:
    # The shape this exists for: one 25 s reprocess in a row of cache hits.
    values = [0.5, 0.49, 0.52, 0.51, 24.8]
    assert outlier_indices(values) == [4]


def test_a_constant_series_has_no_outliers() -> None:
    # Zero dispersion would divide by zero in a naive modified z-score.
    assert outlier_indices([2.0, 2.0, 2.0, 2.0]) == []


def test_a_flat_majority_with_one_spike_still_flags_the_spike() -> None:
    # MAD is exactly 0 here, so the fallback dispersion has to carry it.
    assert outlier_indices([1.0, 1.0, 1.0, 1.0, 9.0]) == [4]


def test_a_short_series_is_never_called_an_outlier() -> None:
    # Two samples cannot establish what "typical" means.
    assert outlier_indices([1.0, 40.0]) == []


# -- the reported distribution ---------------------------------------------


def test_distribution_reports_percentiles_and_keeps_every_sample() -> None:
    values = [0.5, 0.49, 0.52, 0.51, 24.8]
    dist = distribution(values)
    assert dist is not None
    assert dist.count == 5
    assert dist.min == pytest.approx(0.49)
    assert dist.max == pytest.approx(24.8)
    assert dist.p50 == pytest.approx(0.51)
    assert dist.outliers == [4]
    # The outlier is labelled, not removed: count still covers all samples.
    assert dist.count == len(values)


def test_distribution_of_nothing_is_none_rather_than_zero() -> None:
    # A case where every repetition failed has no latency, and reporting 0.0
    # would read as "instant" instead of "no measurement".
    assert distribution([]) is None


def test_distribution_ignores_non_finite_samples() -> None:
    dist = distribution([1.0, math.inf, 2.0, math.nan])
    assert dist is not None
    assert dist.count == 2
