"""Typed record of one benchmark run.

Every field here exists so a number can be compared to another number months
later. A measurement without its fingerprint is not a slow measurement or a
fast one — it is an anecdote, because the hardware, the launch flags, the
model file and the effective configuration all move independently and each of
them moves the result.

Derived rates (cache hit ratio, tokens per second of wall clock) are
properties rather than stored fields. An artefact then cannot contain
arithmetic that disagrees with the samples it was computed from.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.benchmark.stats import Distribution
from harness.runtime import Ownership

RUN_SCHEMA_VERSION = 1
"""Bumped when the artefact layout changes incompatibly. A reader that finds
a version it does not know must say so rather than guess."""


class Phase(enum.StrEnum):
    """Which half of a case a sample belongs to.

    The separation is the whole point: the first call after a prefix change
    reprocesses the entire prompt — 24.83 s for 4816 tokens against 0.49 s for
    a byte-identical repeat (DISCOVERY.md section 4). Folding one warmup into
    five measurements moves the mean by seconds.
    """

    WARMUP = "warmup"
    MEASURE = "measure"


class SampleMetrics(BaseModel):
    """One model call: what it cost and whether it answered the question."""

    case_id: str
    index: int
    phase: Phase
    ok: bool = True
    error_kind: str | None = None
    """Exception class name when the call failed. A failed repetition is a
    result — it is recorded and counted, never retried into silence."""

    wall_clock_ms: float = 0.0
    """Measured by the harness around the call, so it includes queueing and
    transport. The server's own ``prompt_ms``/``predicted_ms`` do not."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens the server served from cache. The gap to
    ``prompt_tokens`` is exactly what this call's prefix change cost."""

    prompt_ms: float | None = None
    predicted_ms: float | None = None
    prompt_per_second: float | None = None
    predicted_per_second: float | None = None
    time_to_first_token_ms: float | None = None

    finish_reason: str | None = None
    expectation_met: bool | None = None
    """None when the case states no expectation — an unasked question has no
    answer, and defaulting it to True would invent a pass rate."""
    expectation_detail: str | None = None

    @property
    def reprocessed_tokens(self) -> int:
        """Prompt tokens the server had to process again."""
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_hit_ratio(self) -> float:
        """Share of the prompt served from cache; 0.0 for an empty prompt."""
        if self.prompt_tokens <= 0:
            return 0.0
        return min(self.cached_tokens / self.prompt_tokens, 1.0)

    @property
    def wall_clock_tokens_per_s(self) -> float | None:
        """Generation throughput as the caller experiences it."""
        if self.wall_clock_ms <= 0 or self.completion_tokens <= 0:
            return None
        return self.completion_tokens / (self.wall_clock_ms / 1000.0)


class ResourceSample(BaseModel):
    """Memory and VRAM at one point in the run.

    zram is tracked separately from disk swap because it is compressed RAM:
    a configuration that only fits by filling it has traded memory for the CPU
    time the model is already saturating (DISCOVERY.md section 1). A benchmark
    that ignores it can report a "successful" configuration that is quietly
    paying for itself in compression.
    """

    label: str = ""
    """Where in the run this was taken: before, after, or a case boundary."""
    at: datetime
    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None
    swap_total_bytes: int | None = None
    swap_used_bytes: int | None = None
    swap_is_zram: bool = False
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None


class HostFingerprint(BaseModel):
    """The machine. Two runs on different hosts are not comparable."""

    hostname: str | None = None
    os_release: str | None = None
    kernel: str | None = None
    cpu_model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    ram_total_bytes: int | None = None
    swap_is_zram: bool = False
    gpus: list[str] = Field(default_factory=list)
    vram_total_bytes: int | None = None


class RuntimeFingerprint(BaseModel):
    """The serving process, and how sure we are that it is the right one.

    ``identity_verified`` is the field that decides whether the rest of the
    artefact means anything. A benchmark run against a server this harness did
    not start produces figures in the normal range and no error at all, which
    is precisely what makes it expensive (#11).
    """

    base_url: str
    host: str = "127.0.0.1"
    port: int
    ownership: Ownership | None = None
    pid: int | None = None
    process_name: str | None = None
    port_state: str | None = None
    identity_verified: bool = False
    identity_detail: str | None = None
    started_at: datetime | None = None
    launch_argv: list[str] | None = None
    """The command line this run's server was started with, credentials
    hidden. ``None`` for an attached server: this harness did not choose those
    flags and must not imply that it did — and on this hardware the flags are
    the single largest lever on the result (DISCOVERY.md section 5.2)."""


class ModelFingerprint(BaseModel):
    """The weights, as the runtime and the GGUF file report them."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = None
    model_path: str | None = None
    n_ctx: int | None = None
    total_slots: int | None = None
    build_info: str | None = None
    chat_template_caps: dict[str, bool] = Field(default_factory=dict)
    architecture: str | None = None
    quantization: str | None = None
    size_label: str | None = None
    file_size_bytes: int | None = None


class Fingerprint(BaseModel):
    """Everything that has to match for two runs to be comparable."""

    host: HostFingerprint = Field(default_factory=HostFingerprint)
    runtime: RuntimeFingerprint
    model: ModelFingerprint = Field(default_factory=ModelFingerprint)
    harness_version: str = ""
    python_version: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    """The *effective* configuration, secrets redacted — what the run actually
    used, not what a file somewhere asks for."""
    config_provenance: dict[str, str] = Field(default_factory=dict)
    """Dotted field path -> "origin: source". Which layer set a value is often
    the answer to why two runs on one machine disagree."""


class Invalidation(BaseModel):
    """A reason this run should not be compared against others.

    Appended, never applied destructively: the samples stay exactly as they
    were recorded. A run that turned out to be untrustworthy is still the
    record of what happened, and deleting it removes the evidence for the very
    claim that it was untrustworthy.
    """

    reason: str
    detail: str | None = None
    at: datetime


class CaseResult(BaseModel):
    """Everything one case produced, warmup included."""

    case_id: str
    description: str = ""
    repetitions: int
    warmup: int
    warmup_samples: list[SampleMetrics] = Field(default_factory=list)
    """Kept, not discarded. The cold call is the most interesting sample in
    the run for anything cache-related; it is merely not a measurement of the
    steady state."""
    samples: list[SampleMetrics] = Field(default_factory=list)
    failures: int = 0
    success_rate: float | None = None
    """Share of measured samples meeting the case's expectation, or None when
    the case states none."""

    wall_clock_ms: Distribution | None = None
    generation_tokens_per_s: Distribution | None = None
    prompt_tokens_per_s: Distribution | None = None
    time_to_first_token_ms: Distribution | None = None
    cache_hit_ratio: Distribution | None = None
    resources: list[ResourceSample] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    """One execution of one suite against one runtime."""

    schema_version: int = RUN_SCHEMA_VERSION
    run_id: str
    suite: str
    started_at: datetime
    finished_at: datetime | None = None
    fingerprint: Fingerprint
    cases: list[CaseResult] = Field(default_factory=list)
    resources: list[ResourceSample] = Field(default_factory=list)
    invalidations: list[Invalidation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    """Conditions that weaken the run without voiding it — an attached server,
    a missing hardware profile. Loud in the summary, kept in the artefact."""

    @property
    def valid(self) -> bool:
        """False once anything has been recorded against this run."""
        return not self.invalidations

    def invalidated(self, reason: str, *, detail: str | None = None,
                    at: datetime | None = None) -> BenchmarkRun:
        """A copy carrying one more invalidation and the same samples."""
        entry = Invalidation(reason=reason, detail=detail, at=at or datetime.now(UTC))
        return self.model_copy(update={"invalidations": [*self.invalidations, entry]})


__all__ = [
    "RUN_SCHEMA_VERSION",
    "BenchmarkRun",
    "CaseResult",
    "Fingerprint",
    "HostFingerprint",
    "Invalidation",
    "ModelFingerprint",
    "Phase",
    "ResourceSample",
    "RuntimeFingerprint",
    "SampleMetrics",
]
