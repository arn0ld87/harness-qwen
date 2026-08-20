"""Summary statistics for benchmark samples.

Percentiles rather than means, for a measured reason. A cold prompt on this
hardware costs 24.83 s where a cached one costs 0.49 s (DISCOVERY.md section
4). A mean over five repetitions containing one cold call reports ~5 s — a
number that describes none of the five and that changes with the warmup
policy rather than with the runtime. The median and the tail percentiles each
describe an actual sample.

Outliers are *labelled*, never dropped. Removing the 25 s sample would hide
exactly the event worth investigating, and a benchmark that quietly discards
its inconvenient data is not evidence of anything.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

MODIFIED_Z_THRESHOLD = 3.5
"""Iglewicz-Hoaglin cut-off for the modified z-score. Robust because it is
built on the median and the MAD: one 25 s sample cannot inflate the dispersion
it is being judged against, which is what happens with mean and stddev."""

MIN_SAMPLES_FOR_OUTLIERS = 3
"""Below this, "typical" is not established. Calling one of two samples an
outlier is a coin toss with extra arithmetic."""

_MAD_TO_SIGMA = 0.6745
"""Consistency constant: MAD * this approximates the standard deviation of a
normal distribution."""


class Distribution(BaseModel):
    """What a series of samples is reported as.

    ``count`` covers every finite sample, including the ones listed in
    ``outliers`` — the flag is an annotation on the data, not a filter applied
    before it.
    """

    count: int
    min: float
    max: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    mad: float
    """Median absolute deviation: dispersion that a single extreme sample
    cannot inflate."""
    outliers: list[int] = Field(default_factory=list)
    """Indices into the finite samples, in input order."""


def _finite(values: Iterable[float]) -> list[float]:
    """Drop NaN and infinity.

    A provider that reported no timing must not turn into a percentile. These
    are absent measurements, not extreme ones, so they are excluded from the
    count rather than treated as outliers.
    """
    return [float(v) for v in values if math.isfinite(v)]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between the two closest ranks.

    The same definition numpy uses by default, spelled out here rather than
    importing numpy for eleven lines of arithmetic — the project adds a
    dependency only after measuring that it is needed.
    """
    if not values:
        raise ValueError("percentile of an empty series is undefined")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile rank must be within 0..100, got {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def median(values: Sequence[float]) -> float:
    """The 50th percentile, named for the times that is what is meant."""
    return percentile(values, 50)


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation from the median."""
    if not values:
        raise ValueError("mad of an empty series is undefined")
    centre = median(values)
    return median([abs(value - centre) for value in values])


def outlier_indices(values: Sequence[float],
                    threshold: float = MODIFIED_Z_THRESHOLD) -> list[int]:
    """Indices whose modified z-score exceeds ``threshold``.

    The MAD is zero whenever more than half the samples are identical — the
    normal shape for a cached prompt repeated five times. Dividing by it would
    either crash or call every non-identical sample an outlier, so the mean
    absolute deviation stands in for that case. If both are zero the series is
    constant and nothing is unusual in it.
    """
    finite = _finite(values)
    if len(finite) < MIN_SAMPLES_FOR_OUTLIERS:
        return []

    centre = median(finite)
    deviations = [abs(value - centre) for value in finite]
    scale = median(deviations) / _MAD_TO_SIGMA
    if scale == 0.0:
        scale = sum(deviations) / len(deviations)
    if scale == 0.0:
        return []

    return [index for index, deviation in enumerate(deviations)
            if deviation / scale > threshold]


def distribution(values: Iterable[float]) -> Distribution | None:
    """Summarise a series, or ``None`` when there is nothing to summarise.

    ``None`` rather than a zeroed record: a case whose every repetition failed
    has no latency, and printing ``0.0 ms`` for it reads as "instant".
    """
    finite = _finite(values)
    if not finite:
        return None

    return Distribution(
        count=len(finite),
        min=min(finite),
        max=max(finite),
        mean=sum(finite) / len(finite),
        p50=percentile(finite, 50),
        p90=percentile(finite, 90),
        p95=percentile(finite, 95),
        p99=percentile(finite, 99),
        mad=mad(finite),
        outliers=outlier_indices(finite),
    )


__all__ = [
    "MIN_SAMPLES_FOR_OUTLIERS",
    "MODIFIED_Z_THRESHOLD",
    "Distribution",
    "distribution",
    "mad",
    "median",
    "outlier_indices",
    "percentile",
]
