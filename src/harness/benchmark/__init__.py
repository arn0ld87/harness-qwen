"""Reproducible capability and performance runs.

This is the layer the rest of the repository's claims rest on: every
structural decision in `docs/DISCOVERY.md` and `CONTEXT.md` is derived from a
measurement, and this is what produces measurements. That obliges it to be
harder on itself than the code it measures — warmup separated from the
measured phase, percentiles instead of means, the serving process proved to be
the one that was started, and a fingerprint on every run so a number stays
comparable after the hardware, the flags or the model file have moved.

A run is a library object: assemble a suite, hand it a provider, get a typed
record back. The CLI that prints it is a separate concern (#27).
"""

from harness.benchmark.identity import probe_identity, redact_argv
from harness.benchmark.models import (
    RUN_SCHEMA_VERSION,
    BenchmarkRun,
    CaseResult,
    Fingerprint,
    HostFingerprint,
    Invalidation,
    ModelFingerprint,
    Phase,
    ResourceSample,
    RuntimeFingerprint,
    SampleMetrics,
)
from harness.benchmark.report import (
    artifact_path,
    invalidate_artifact,
    read_run,
    render_summary,
    write_run,
)
from harness.benchmark.resources import sample_resources
from harness.benchmark.runner import IDENTITY_REASON, BenchmarkRunner
from harness.benchmark.stats import Distribution, distribution, median, percentile
from harness.benchmark.suite import (
    SUITE_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkSuite,
    CaseDefaults,
    Expectation,
    ExpectationKind,
    OutputMode,
    load_suite,
)

__all__ = [
    "IDENTITY_REASON",
    "RUN_SCHEMA_VERSION",
    "SUITE_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkRun",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "CaseDefaults",
    "CaseResult",
    "Distribution",
    "Expectation",
    "ExpectationKind",
    "Fingerprint",
    "HostFingerprint",
    "Invalidation",
    "ModelFingerprint",
    "OutputMode",
    "Phase",
    "ResourceSample",
    "RuntimeFingerprint",
    "SampleMetrics",
    "artifact_path",
    "distribution",
    "invalidate_artifact",
    "load_suite",
    "median",
    "percentile",
    "probe_identity",
    "read_run",
    "redact_argv",
    "render_summary",
    "sample_resources",
    "write_run",
]
