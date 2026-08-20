"""Reproducible benchmark runs against a scripted provider (issue #19).

Two failures shape these tests. The first is a run that measures a process
nobody started: it produces figures in the normal range and no error, so
`verify_owner` is a precondition here rather than a diagnostic. The second is
a warmup silently folded into the result: the cold prompt costs ~25 s on this
hardware against ~0.5 s cached, so one unlabelled warmup moves every mean in
the report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.benchmark import (
    BenchmarkCase,
    BenchmarkRunner,
    BenchmarkSuite,
    Expectation,
    ExpectationKind,
    OutputMode,
    Phase,
    ResourceSample,
    load_suite,
)
from harness.config import Origin, resolve_config
from harness.config.resolve import ResolvedConfig
from harness.core import ModelResponse, ProviderTimeout, ToolCall
from harness.discovery.models import CpuInfo, HardwareProfile, MemoryInfo
from harness.models.fake import FakeProvider
from harness.runtime import Ownership, PortReport, PortState, RuntimeHandle
from harness.runtime.port import RuntimeIdentityMismatch

CAPABILITY_SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "model-capabilities.json"


# -- fixtures and doubles --------------------------------------------------


def _resolved(**overrides: object) -> ResolvedConfig:
    base: dict[str, object] = {"runtime.port": 18080, "runtime.host": "127.0.0.1"}
    base.update(overrides)
    return resolve_config(overrides=base)


def _handle(*, owned: bool = True, pid: int | None = 4242) -> RuntimeHandle:
    return RuntimeHandle(
        base_url="http://127.0.0.1:18080",
        ownership=Ownership.OWNED if owned else Ownership.ATTACHED,
        pid=pid,
        started_at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
    )


def _profile() -> HardwareProfile:
    return HardwareProfile(
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        hostname="testbox",
        kernel="7.1.6",
        cpu=CpuInfo(model_name="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=20 * 1024**3,
                          swap_total_bytes=8 * 1024**3, swap_used_bytes=4 * 1024**3,
                          swap_is_zram=True),
    )


class _Ok:
    """Identity probes that agree the configured pid holds the port."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.verify_calls = 0
        self.inspect_calls = 0

    def report(self) -> PortReport:
        return PortReport(port=18080, state=PortState.INFERENCE_SERVER, pid=self.pid,
                          process_name="llama-server")

    async def verify(self, port: int, pid: int, host: str = "127.0.0.1") -> PortReport:
        self.verify_calls += 1
        return self.report()

    async def inspect(self, port: int, host: str = "127.0.0.1") -> PortReport:
        self.inspect_calls += 1
        return self.report()


class _ChangesAfterwards(_Ok):
    """Verifies once, then reports a different process on the port."""

    async def verify(self, port: int, pid: int, host: str = "127.0.0.1") -> PortReport:
        self.verify_calls += 1
        if self.verify_calls == 1:
            return self.report()
        raise RuntimeIdentityMismatch(f"port {port} is held by pid 9999, not by pid {pid}")


class _Clock:
    """Monotonic fake clock: every read advances by a fixed step."""

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.t = 0.0

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _resources() -> ResourceSample:
    return ResourceSample(
        at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
        ram_total_bytes=32 * 1024**3,
        ram_available_bytes=20 * 1024**3,
        swap_used_bytes=4 * 1024**3,
        swap_total_bytes=8 * 1024**3,
        swap_is_zram=True,
        vram_used_bytes=2831 * 1024**2,
        vram_total_bytes=6144 * 1024**2,
    )


_UNSET = object()
"""Distinguishes "no handle given" from a deliberate handle=None."""


def _runner(provider: FakeProvider, *, handle: RuntimeHandle | None | object = _UNSET,
            probes: _Ok | None = None, resolved: ResolvedConfig | None = None,
            **kwargs: object) -> BenchmarkRunner:
    probes = probes or _Ok()
    return BenchmarkRunner(
        provider,
        resolved or _resolved(),
        handle=_handle() if handle is _UNSET else handle,
        profile=_profile(),
        sample_resources=_resources,
        clock=_Clock(),
        inspect=probes.inspect,
        verify=probes.verify,
        **kwargs,
    )


def _suite(**case: object) -> BenchmarkSuite:
    defaults: dict[str, object] = {"id": "echo", "prompt": "ping", "system": "be brief",
                                   "repetitions": 3, "warmup": 1}
    defaults.update(case)
    return BenchmarkSuite(name="unit", cases=[BenchmarkCase(**defaults)])


# -- warmup is separated from measurement ----------------------------------


async def test_warmup_samples_are_recorded_but_kept_out_of_the_statistics() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider).run(_suite(repetitions=3, warmup=2))

    case = run.cases[0]
    assert len(case.warmup_samples) == 2
    assert len(case.samples) == 3
    assert provider.calls == 5
    assert case.wall_clock_ms is not None
    assert case.wall_clock_ms.count == 3
    assert all(s.phase is Phase.WARMUP for s in case.warmup_samples)
    assert all(s.phase is Phase.MEASURE for s in case.samples)


async def test_a_case_without_warmup_still_runs_its_repetitions() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider).run(_suite(repetitions=2, warmup=0))
    assert run.cases[0].warmup_samples == []
    assert len(run.cases[0].samples) == 2


async def test_the_cold_first_call_lands_in_warmup_where_it_belongs() -> None:
    # FakeProvider reports cached_tokens only once a prefix repeats, which is
    # the same signal the real server gives. With warmup, no measured sample
    # carries the cold reprocess.
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider).run(_suite(repetitions=2, warmup=1))
    assert run.cases[0].warmup_samples[0].cached_tokens == 0
    assert all(s.cached_tokens > 0 for s in run.cases[0].samples)


# -- the fingerprint -------------------------------------------------------


async def test_a_run_carries_an_id_timestamps_and_a_host_fingerprint() -> None:
    run = await _runner(FakeProvider(["pong"], repeat_last=True)).run(_suite())
    assert run.run_id
    assert run.started_at is not None and run.finished_at is not None
    assert run.fingerprint.host.hostname == "testbox"
    assert run.fingerprint.host.swap_is_zram is True
    assert run.fingerprint.harness_version


async def test_two_runs_do_not_share_an_id() -> None:
    runner = _runner(FakeProvider(["pong"], repeat_last=True))
    first = await runner.run(_suite())
    second = await runner.run(_suite())
    assert first.run_id != second.run_id


async def test_the_effective_configuration_is_recorded_with_its_provenance() -> None:
    resolved = _resolved(**{"runtime.port": 19999})
    run = await _runner(FakeProvider(["pong"], repeat_last=True),
                        resolved=resolved).run(_suite())
    assert run.fingerprint.config["runtime"]["port"] == 19999
    assert run.fingerprint.config_provenance["runtime.port"].startswith(Origin.CLI)


async def test_the_recorded_configuration_does_not_contain_the_api_key() -> None:
    resolved = _resolved(**{"runtime.api_key": "sk-do-not-log-me"})
    run = await _runner(FakeProvider(["pong"], repeat_last=True),
                        resolved=resolved).run(_suite())
    assert "sk-do-not-log-me" not in run.model_dump_json()


async def test_a_credential_in_extra_flags_is_hidden_in_the_recorded_argv(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")
    resolved = _resolved(**{
        "runtime.server_binary": str(binary),
        "model.path": str(tmp_path / "model.gguf"),
        "model.extra_flags": ["--api-key", "sk-secret", "--jinja"],
    })
    run = await _runner(FakeProvider(["pong"], repeat_last=True),
                        resolved=resolved).run(_suite())
    argv = run.fingerprint.runtime.launch_argv
    assert argv is not None
    assert "--jinja" in argv
    assert "sk-secret" not in " ".join(argv)


async def test_the_model_fingerprint_comes_from_the_serving_runtime() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider).run(_suite())
    assert run.fingerprint.model.model_id == "fake"
    assert run.fingerprint.model.n_ctx == 65536


# -- process identity ------------------------------------------------------


async def test_an_owned_runtime_is_verified_before_anything_is_measured() -> None:
    probes = _Ok()
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider, probes=probes).run(_suite())
    assert probes.verify_calls >= 1
    assert run.fingerprint.runtime.identity_verified is True
    assert run.fingerprint.runtime.pid == 4242


async def test_a_foreign_process_on_the_port_aborts_the_run_before_measuring() -> None:
    # The whole point of the check: numbers from somebody else's process look
    # exactly like ours. Nothing may be measured once identity fails.
    provider = FakeProvider(["pong"], repeat_last=True)

    async def verify(port: int, pid: int, host: str = "127.0.0.1") -> PortReport:
        raise RuntimeIdentityMismatch(f"port {port} is held by pid 9999, not pid {pid}")

    async def inspect(port: int, host: str = "127.0.0.1") -> PortReport:
        return PortReport(port=port, state=PortState.INFERENCE_SERVER, pid=9999)

    runner = BenchmarkRunner(
        provider, _resolved(), handle=_handle(), profile=_profile(),
        sample_resources=_resources, clock=_Clock(), inspect=inspect, verify=verify,
    )
    with pytest.raises(RuntimeIdentityMismatch):
        await runner.run(_suite())
    assert provider.calls == 0


async def test_the_real_verifier_refuses_a_port_nothing_is_listening_on(
    unused_tcp_port: int,
) -> None:
    # Exercises the shipped default probes, not a stub: an owned handle whose
    # server is gone must not produce a benchmark run.
    provider = FakeProvider(["pong"], repeat_last=True)
    runner = BenchmarkRunner(
        provider,
        _resolved(**{"runtime.port": unused_tcp_port}),
        handle=_handle(),
        profile=_profile(),
        sample_resources=_resources,
        clock=_Clock(),
    )
    with pytest.raises(RuntimeIdentityMismatch):
        await runner.run(_suite())
    assert provider.calls == 0


async def test_an_attached_runtime_is_measured_but_marked_unverified() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider, handle=_handle(owned=False)).run(_suite())
    assert run.fingerprint.runtime.identity_verified is False
    assert run.fingerprint.runtime.launch_argv is None
    assert any("attach" in warning for warning in run.warnings)
    assert run.cases[0].samples


async def test_a_run_without_a_handle_at_all_is_marked_unverified() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider, handle=None).run(_suite())
    assert run.fingerprint.runtime.identity_verified is False
    assert run.warnings


async def test_the_process_changing_under_the_run_invalidates_it_without_data_loss(
) -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    run = await _runner(provider, probes=_ChangesAfterwards()).run(_suite())
    assert not run.valid
    assert run.invalidations[0].reason == "runtime_identity"
    # The samples survive: an invalid run is still evidence of what happened.
    assert len(run.cases[0].samples) == 3


# -- metrics ---------------------------------------------------------------


async def test_prompt_generation_and_cache_metrics_are_recorded_per_sample() -> None:
    provider = FakeProvider(["pong pong pong"], repeat_last=True)
    run = await _runner(provider).run(_suite(repetitions=2, warmup=1))
    sample = run.cases[0].samples[0]
    assert sample.prompt_tokens > 0
    assert sample.completion_tokens > 0
    assert sample.cached_tokens > 0
    assert sample.reprocessed_tokens == sample.prompt_tokens - sample.cached_tokens
    assert 0.0 <= sample.cache_hit_ratio <= 1.0
    assert sample.wall_clock_ms == pytest.approx(250.0)
    assert sample.wall_clock_tokens_per_s is not None


async def test_resource_use_is_sampled_around_the_run_and_around_each_case() -> None:
    # DISCOVERY.md section 1: zram is compressed RAM, so a configuration that
    # "fits" by filling it has traded memory for the CPU the model needs.
    run = await _runner(FakeProvider(["pong"], repeat_last=True)).run(_suite())
    assert [s.label for s in run.resources] == ["before", "after"]
    assert [s.label for s in run.cases[0].resources] == ["before", "after"]
    assert run.resources[0].swap_is_zram is True


async def test_a_failing_repetition_is_recorded_and_excluded_from_latency() -> None:
    provider = FakeProvider(["pong"], repeat_last=True)
    provider.fail_on_call(3, ProviderTimeout("too slow"))
    run = await _runner(provider).run(_suite(repetitions=3, warmup=1))

    case = run.cases[0]
    assert case.failures == 1
    assert [s.ok for s in case.samples] == [True, False, True]
    assert case.samples[1].error_kind == "ProviderTimeout"
    assert case.wall_clock_ms is not None and case.wall_clock_ms.count == 2


async def test_a_case_where_everything_failed_reports_no_distribution() -> None:
    provider = FakeProvider([], repeat_last=False)
    run = await _runner(provider).run(_suite(repetitions=2, warmup=0))
    case = run.cases[0]
    assert case.failures == 2
    assert case.wall_clock_ms is None


# -- expectations ----------------------------------------------------------


async def test_a_capability_case_reports_a_success_rate() -> None:
    provider = FakeProvider(['{"answer": 4}', "not json", '{"answer": 4}'])
    suite = _suite(
        repetitions=3, warmup=0,
        expect=Expectation(kind=ExpectationKind.JSON_OBJECT, required_keys=["answer"]),
    )
    run = await _runner(provider).run(suite)
    assert run.cases[0].success_rate == pytest.approx(2 / 3)
    assert run.cases[0].samples[1].expectation_met is False


async def test_a_case_without_an_expectation_reports_no_success_rate() -> None:
    run = await _runner(FakeProvider(["pong"], repeat_last=True)).run(_suite())
    assert run.cases[0].success_rate is None


async def test_a_tool_case_sends_tool_definitions_and_scores_the_call() -> None:
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    provider = FakeProvider(
        [ModelResponse(content="", tool_calls=[
            ToolCall(id="1", name="read_file", arguments={"path": "x"})])],
        repeat_last=True,
    )
    suite = _suite(
        repetitions=2, warmup=0, output_mode=OutputMode.TOOLS, tools=tools,
        expect=Expectation(kind=ExpectationKind.TOOL_CALL, value="read_file"),
    )
    run = await _runner(provider).run(suite)
    assert provider.last_request.tools == tools
    assert run.cases[0].success_rate == 1.0


# -- suites with several cases ---------------------------------------------


async def test_every_case_in_the_shipped_suite_produces_a_result() -> None:
    suite = load_suite(CAPABILITY_SUITE)
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    run = await _runner(provider).run(suite)
    assert [c.case_id for c in run.cases] == [c.id for c in suite.cases]
    assert run.suite == suite.name


@pytest.mark.local_llm
async def test_the_capability_suite_runs_against_the_served_model() -> None:
    """The same framework against the real 35B model.

    Excluded from the default suite: a single cold prompt costs about 25 s on
    this hardware, and the whole suite is minutes.
    """
    from harness.models.llamacpp import LlamaCppProvider

    resolved = _resolved()
    provider = LlamaCppProvider(resolved.config.runtime.base_url)
    runner = BenchmarkRunner(provider, resolved, handle=None)
    run = await runner.run(load_suite(CAPABILITY_SUITE))
    assert run.cases
    assert all(case.samples for case in run.cases)
