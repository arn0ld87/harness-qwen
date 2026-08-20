"""Executing a suite and recording what it cost.

Three rules govern the order of operations here, and all three come from a
failure that produces plausible numbers rather than an error:

1. **Identity before measurement.** The port is checked against the pid this
   harness started before a single call is made, and again after the last
   one. A run whose subject changed halfway through is invalidated, not
   averaged.
2. **Warmup before measurement, and kept.** The first call after a prefix
   change reprocesses the whole prompt — 24.83 s against 0.49 s for a
   byte-identical repeat. Warmup samples are recorded in full and excluded
   from the statistics; deleting them would hide the most informative call in
   the run, folding them in would move every number.
3. **Failures are results.** A repetition that raised is recorded with its
   error class and counted, never retried until it agrees.

Nothing here records model output. The samples carry token counts, timings
and a deterministic verdict — not the response text, and not reasoning.
"""

from __future__ import annotations

import platform
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from harness import __version__
from harness.benchmark.identity import (
    InspectPort,
    VerifyOwner,
    identity_changed,
    probe_identity,
)
from harness.benchmark.models import (
    BenchmarkRun,
    CaseResult,
    Fingerprint,
    HostFingerprint,
    ModelFingerprint,
    Phase,
    ResourceSample,
    SampleMetrics,
)
from harness.benchmark.resources import sample_resources as default_resources
from harness.benchmark.stats import distribution
from harness.benchmark.suite import BenchmarkCase, BenchmarkSuite, ExpectationKind
from harness.config.resolve import ResolvedConfig
from harness.core import ProviderError
from harness.discovery.hardware import probe_hardware
from harness.discovery.models import HardwareProfile
from harness.models.base import ModelProvider
from harness.runtime import RuntimeHandle
from harness.runtime.port import RuntimeIdentityMismatch, inspect_port_async, verify_owner

IDENTITY_REASON = "runtime_identity"
"""Invalidation reason for a runtime that changed under the run."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BenchmarkRunner:
    """Runs one suite against one provider and returns a typed record.

    Everything that touches the world outside the process is injectable — the
    clock, the resource probe, both port probes — so the whole layer is
    exercised in CI against ``FakeProvider`` without a model, a GPU or a
    listening socket.
    """

    def __init__(
        self,
        provider: ModelProvider,
        resolved: ResolvedConfig,
        *,
        handle: RuntimeHandle | None = None,
        profile: HardwareProfile | None = None,
        sample_resources: Callable[[], ResourceSample] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = _utcnow,
        inspect: InspectPort = inspect_port_async,
        verify: VerifyOwner = verify_owner,
    ) -> None:
        self.provider = provider
        self.resolved = resolved
        self.handle = handle
        self.clock = clock
        self.now = now
        self._profile = profile
        self._sample_resources = sample_resources or default_resources
        self._inspect = inspect
        self._verify = verify

    # -- public API --------------------------------------------------------

    async def run(self, suite: BenchmarkSuite, *, run_id: str | None = None) -> BenchmarkRun:
        """Execute every case and return the record.

        Raises :class:`RuntimeIdentityMismatch` before measuring anything when
        the server on the configured port is not the one this harness started.
        """
        started_at = self.now()
        identity = await probe_identity(
            self.resolved.config, self.handle, inspect=self._inspect, verify=self._verify
        )

        warnings = list(self.resolved.warnings)
        if identity.identity_detail:
            warnings.append(identity.identity_detail)
        model, model_warning = await self._model_fingerprint()
        if model_warning:
            warnings.append(model_warning)

        run = BenchmarkRun(
            run_id=run_id or self._new_run_id(started_at),
            suite=suite.name,
            started_at=started_at,
            fingerprint=Fingerprint(
                host=self._host_fingerprint(),
                runtime=identity,
                model=model,
                harness_version=__version__,
                python_version=platform.python_version(),
                config=self.resolved.as_dict(),
                config_provenance={
                    path: f"{origin}: {source}"
                    for path, (origin, source) in self.resolved.provenance().items()
                },
            ),
            warnings=warnings,
        )

        resources = [self._resource("before")]
        cases = [await self._run_case(case) for case in suite.cases]
        resources.append(self._resource("after"))

        run = run.model_copy(update={
            "cases": cases, "resources": resources, "finished_at": self.now(),
        })
        return await self._confirm_identity(run)

    # -- one case ----------------------------------------------------------

    async def _run_case(self, case: BenchmarkCase) -> CaseResult:
        resources = [self._resource("before")]
        warmup = [await self._measure(case, index, Phase.WARMUP)
                  for index in range(case.warmup)]
        samples = [await self._measure(case, index, Phase.MEASURE)
                   for index in range(case.repetitions)]
        resources.append(self._resource("after"))

        ok = [sample for sample in samples if sample.ok]
        judged = [sample for sample in ok if sample.expectation_met is not None]

        return CaseResult(
            case_id=case.id,
            description=case.description,
            repetitions=case.repetitions,
            warmup=case.warmup,
            warmup_samples=warmup,
            samples=samples,
            failures=len(samples) - len(ok),
            success_rate=(
                sum(1 for s in judged if s.expectation_met) / len(judged) if judged else None
            ),
            wall_clock_ms=distribution(s.wall_clock_ms for s in ok),
            # Harness-measured rather than server-reported: it is the number a
            # caller experiences, and it exists even when the runtime sends no
            # timings at all. The server's own view is kept per sample.
            generation_tokens_per_s=distribution(
                s.wall_clock_tokens_per_s for s in ok if s.wall_clock_tokens_per_s is not None
            ),
            # Only the server can measure this one; absent means absent.
            prompt_tokens_per_s=distribution(
                s.prompt_per_second for s in ok if s.prompt_per_second is not None
            ),
            time_to_first_token_ms=distribution(
                s.time_to_first_token_ms for s in ok if s.time_to_first_token_ms is not None
            ),
            cache_hit_ratio=distribution(s.cache_hit_ratio for s in ok),
            resources=resources,
        )

    async def _measure(self, case: BenchmarkCase, index: int, phase: Phase) -> SampleMetrics:
        """One call, timed from outside the provider."""
        request = case.to_request()
        started = self.clock()
        try:
            response = await self.provider.generate(request)
        except ProviderError as exc:
            return SampleMetrics(
                case_id=case.id, index=index, phase=phase, ok=False,
                error_kind=type(exc).__name__,
                wall_clock_ms=(self.clock() - started) * 1000.0,
            )
        elapsed_ms = (self.clock() - started) * 1000.0

        met, detail = case.expect.check(response)
        judged = case.expect.kind is not ExpectationKind.NONE
        return SampleMetrics(
            case_id=case.id,
            index=index,
            phase=phase,
            wall_clock_ms=elapsed_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=response.usage.cached_tokens,
            prompt_ms=response.timings.prompt_ms,
            predicted_ms=response.timings.predicted_ms,
            prompt_per_second=response.timings.prompt_per_second,
            predicted_per_second=response.timings.predicted_per_second,
            time_to_first_token_ms=response.timings.time_to_first_token_ms,
            finish_reason=response.finish_reason,
            expectation_met=met if judged else None,
            expectation_detail=detail if judged else None,
        )

    # -- fingerprinting ----------------------------------------------------

    def _new_run_id(self, started_at: datetime) -> str:
        """Sortable by time, unique by construction.

        Two runs a second apart on one machine must not collide, and a
        directory of artefacts should list in the order they were made.
        """
        return f"bench-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    def _host_fingerprint(self) -> HostFingerprint:
        profile = self.profile
        vram = [gpu.vram_total_bytes for gpu in profile.gpus if gpu.vram_total_bytes]
        return HostFingerprint(
            hostname=profile.hostname,
            os_release=profile.os_release,
            kernel=profile.kernel,
            cpu_model=profile.cpu.model_name,
            physical_cores=profile.cpu.physical_cores,
            logical_cores=profile.cpu.logical_cores,
            ram_total_bytes=profile.memory.total_bytes,
            swap_is_zram=profile.memory.swap_is_zram,
            gpus=[gpu.name for gpu in profile.gpus if gpu.name],
            vram_total_bytes=max(vram) if vram else None,
        )

    @property
    def profile(self) -> HardwareProfile:
        """The hardware description, probed once if none was supplied."""
        if self._profile is None:
            self._profile = HardwareProfile(created_at=self.now(), **probe_hardware())
        return self._profile

    async def _model_fingerprint(self) -> tuple[ModelFingerprint, str | None]:
        """What the runtime says it is serving, plus what the GGUF said.

        A runtime that will not answer does not stop the run: the failure
        shows up in every sample anyway, and a fingerprint that says "unknown"
        is more useful than no artefact at all.
        """
        gguf = self.profile.model
        try:
            info = await self.provider.model_info()
        except ProviderError as exc:
            return ModelFingerprint(), f"model info unavailable: {type(exc).__name__}: {exc}"

        return ModelFingerprint(
            model_id=info.model_id,
            model_path=info.model_path or (str(gguf.path) if gguf and gguf.path else None),
            n_ctx=info.n_ctx,
            total_slots=info.total_slots,
            build_info=info.build_info,
            chat_template_caps=dict(info.chat_template_caps),
            architecture=gguf.architecture if gguf else None,
            quantization=gguf.quantization if gguf else None,
            size_label=gguf.size_label if gguf else None,
            file_size_bytes=gguf.file_size_bytes if gguf else None,
        ), None

    def _resource(self, label: str) -> ResourceSample:
        return self._sample_resources().model_copy(update={"label": label})

    async def _confirm_identity(self, run: BenchmarkRun) -> BenchmarkRun:
        """Re-check the port after the last sample.

        Invalidating rather than raising: the samples are already recorded and
        they are the evidence that something moved. Throwing them away would
        leave nobody able to see what the run actually measured.
        """
        try:
            after = await probe_identity(
                self.resolved.config, self.handle, inspect=self._inspect, verify=self._verify
            )
        except RuntimeIdentityMismatch as exc:
            return run.invalidated(IDENTITY_REASON, detail=str(exc), at=self.now())

        changed = identity_changed(run.fingerprint.runtime, after)
        if changed:
            return run.invalidated(IDENTITY_REASON, detail=changed, at=self.now())
        return run


__all__ = ["IDENTITY_REASON", "BenchmarkRunner"]
