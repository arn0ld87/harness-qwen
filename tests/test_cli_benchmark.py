"""`harness benchmark` end to end (issue #27).

Every run uses a scripted ``FakeProvider`` behind a stubbed ``build_runner``:
the benchmark framework, the suite loader and the process-level contract are
exercised, the 35B model and the real llama-server are not. What is tested
here is the CLI surface — which subcommand runs which suite, what gets written,
and which exit code a measurement that was not trustworthy returns.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from harness.benchmark import BenchmarkRunner
from harness.cli import app
from harness.cli_benchmark import EXIT_CONFIG, EXIT_ERROR, EXIT_INVALID, EXIT_OK
from harness.config import resolve_config
from harness.config.resolve import ResolvedConfig
from harness.discovery.models import CpuInfo, HardwareProfile, MemoryInfo
from harness.models import FakeProvider
from harness.runtime import (
    Ownership,
    PortReport,
    PortState,
    RuntimeHandle,
    RuntimeIdentityMismatch,
)

CAPABILITY_SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "model-capabilities.json"
runner = CliRunner()


# -- stub runtime (no model, no socket) ------------------------------------


def _resolved() -> ResolvedConfig:
    return resolve_config(overrides={"runtime.port": 18080, "runtime.host": "127.0.0.1"})


def _handle() -> RuntimeHandle:
    return RuntimeHandle(
        base_url="http://127.0.0.1:18080",
        ownership=Ownership.OWNED,
        pid=4242,
        started_at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
    )


def _profile() -> HardwareProfile:
    return HardwareProfile(
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        hostname="testbox",
        kernel="7.1.6",
        cpu=CpuInfo(model_name="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=20 * 1024**3),
    )


class _Ok:
    """Identity probes that agree the configured pid holds the port."""

    def __init__(self) -> None:
        self.verify_calls = 0

    def report(self) -> PortReport:
        return PortReport(port=18080, state=PortState.INFERENCE_SERVER, pid=4242,
                          process_name="llama-server")

    async def verify(self, port: int, pid: int, host: str = "127.0.0.1") -> PortReport:
        self.verify_calls += 1
        return self.report()

    async def inspect(self, port: int, host: str = "127.0.0.1") -> PortReport:
        return self.report()


class _ChangesAfterwards(_Ok):
    """Verifies once, then reports a different process — invalidates the run."""

    async def verify(self, port: int, pid: int, host: str = "127.0.0.1") -> PortReport:
        self.verify_calls += 1
        if self.verify_calls == 1:
            return self.report()
        raise RuntimeIdentityMismatch(f"port {port} is held by pid 9999, not by pid {pid}")


class _Clock:
    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.t = 0.0

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _resources():
    from harness.benchmark import ResourceSample

    return ResourceSample(
        at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
        ram_total_bytes=32 * 1024**3,
        ram_available_bytes=20 * 1024**3,
        vram_used_bytes=2831 * 1024**2,
        vram_total_bytes=6144 * 1024**2,
    )


RunnerFactory = Callable[
    [ResolvedConfig], Awaitable[tuple[BenchmarkRunner, Callable[[], Awaitable[None]]]]
]


def _factory(provider: FakeProvider, *, probes: _Ok | None = None) -> RunnerFactory:
    """A ``build_runner`` that returns a real ``BenchmarkRunner`` wired to a
    scripted provider and stub probes — the same shape the CLI's real
    ``build_runner`` returns, only nothing touches the network."""
    probes = probes or _Ok()

    async def make(
        resolved: ResolvedConfig,
    ) -> tuple[BenchmarkRunner, Callable[[], Awaitable[None]]]:
        r = BenchmarkRunner(
            provider,
            _resolved(),
            handle=_handle(),
            profile=_profile(),
            sample_resources=_resources,
            clock=_Clock(),
            inspect=probes.inspect,
            verify=probes.verify,
        )

        async def cleanup() -> None:
            return None

        return r, cleanup

    return make


def _invoke(*args: str, factory: RunnerFactory, env: dict[str, str] | None = None) -> Result:
    import harness.cli_benchmark as cli_benchmark

    original = cli_benchmark.build_runner
    cli_benchmark.build_runner = factory  # type: ignore[assignment]
    try:
        return runner.invoke(app, ["benchmark", *args], env=env)
    finally:
        cli_benchmark.build_runner = original  # type: ignore[assignment]


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "HARNESS_WORKSPACE": str(tmp_path),
        "HARNESS_DATABASE": str(tmp_path / "memory.sqlite"),
    }


# -- the run subcommands ---------------------------------------------------


def test_capability_runs_the_shipped_suite_and_exits_zero(tmp_path: Path) -> None:
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "capability", "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_OK
    assert "benchmark run" in result.stdout
    assert "model-capabilities" in result.stdout
    artefacts = list((tmp_path / "runs").glob("*.json"))
    assert len(artefacts) == 1
    payload = json.loads(artefacts[0].read_text(encoding="utf-8"))
    assert payload["suite"] == "model-capabilities"


def test_an_invalid_run_exits_three_not_zero(tmp_path: Path) -> None:
    """A run whose serving process moved under it is not a success, and the
    exit code must say so — a caller that treats 0 as "good measurement" has
    bought exactly the assurance the identity check exists to refuse."""
    provider = FakeProvider(["pong"], repeat_last=True)
    result = _invoke(
        "capability", "--out", str(tmp_path / "runs"),
        factory=_factory(provider, probes=_ChangesAfterwards()), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_INVALID
    assert "INVALID" in result.stdout
    assert "runtime_identity" in result.stdout


def test_a_missing_suite_exits_one(tmp_path: Path) -> None:
    """``flags`` and ``tasks`` are wired subcommands; their canonical suites
    land with #20 and #21. Until then the command says "not found" and fails
    loudly rather than running an empty, successful-looking run."""
    provider = FakeProvider(["pong"], repeat_last=True)
    result = _invoke(
        "flags", "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_ERROR
    assert "not found" in result.stdout.lower()


def test_a_suite_override_runs_any_suite_file(tmp_path: Path) -> None:
    """Every run subcommand accepts ``--suite`` so a custom or not-yet-shipped
    suite is runnable through any of them — the subcommand is the default
    path, not a constraint."""
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "tasks", "--suite", str(CAPABILITY_SUITE), "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_OK
    assert "model-capabilities" in result.stdout


def test_run_id_and_output_directory_are_controllable(tmp_path: Path) -> None:
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "capability", "--run-id", "bench-fixed", "--out", str(tmp_path / "deep" / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_OK
    artefact = tmp_path / "deep" / "runs" / "bench-fixed.json"
    assert artefact.exists()
    assert json.loads(artefact.read_text(encoding="utf-8"))["run_id"] == "bench-fixed"


def test_json_output_emits_the_run(tmp_path: Path) -> None:
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "capability", "--json", "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_OK
    payload = json.loads(result.stdout)
    assert payload["run_id"]
    assert payload["suite"] == "model-capabilities"


def test_case_selection_narrows_the_suite(tmp_path: Path) -> None:
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "capability", "--case", "json_schema_constrained",
        "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_OK
    artefact = json.loads(next((tmp_path / "runs").glob("*.json")).read_text("utf-8"))
    assert [c["case_id"] for c in artefact["cases"]] == ["json_schema_constrained"]


def test_an_unknown_case_id_is_refused(tmp_path: Path) -> None:
    """A mistyped ``--case`` producing an empty run is the silent success the
    suite loader exists to refuse (see ``BenchmarkSuite.select``)."""
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    result = _invoke(
        "capability", "--case", "no-such-case", "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=_env(tmp_path),
    )

    assert result.exit_code == EXIT_ERROR
    assert "no-such-case" in result.stdout


def test_a_broken_configuration_exits_two(tmp_path: Path) -> None:
    provider = FakeProvider(['{"answer": 4}'], repeat_last=True)
    env = _env(tmp_path)
    env["HARNESS_RUNTIME_PORT"] = "nope"
    result = _invoke(
        "capability", "--out", str(tmp_path / "runs"),
        factory=_factory(provider), env=env,
    )

    assert result.exit_code == EXIT_CONFIG
    assert "Traceback" not in result.stdout


def test_an_identity_mismatch_before_measuring_exits_one(tmp_path: Path) -> None:
    """A runtime that is not the one this harness started must stop the run
    before any sample is taken — an execution failure, not an invalid run,
    because there is no run to invalidate."""
    import harness.cli_benchmark as cli_benchmark

    class _RaisingRunner:
        async def run(self, suite, *, run_id: str | None = None):  # type: ignore[no-untyped-def]
            raise RuntimeIdentityMismatch("port 18080 is held by pid 9999")

    async def make(resolved: ResolvedConfig):
        async def cleanup() -> None:
            return None
        return _RaisingRunner(), cleanup

    original = cli_benchmark.build_runner
    cli_benchmark.build_runner = make  # type: ignore[assignment]
    try:
        result = runner.invoke(app, ["benchmark", "capability", "--out", str(tmp_path / "runs")],
                               env=_env(tmp_path))
    finally:
        cli_benchmark.build_runner = original  # type: ignore[assignment]

    assert result.exit_code == EXIT_ERROR
    assert "identity" in result.stdout.lower() or "9999" in result.stdout


# -- compare ---------------------------------------------------------------


def _write_run(tmp_path: Path, name: str, *, invalid: bool = False) -> Path:
    from harness.benchmark import (
        BenchmarkRun,
        CaseResult,
        Fingerprint,
        HostFingerprint,
        ModelFingerprint,
        Phase,
        RuntimeFingerprint,
        SampleMetrics,
    )
    from harness.benchmark.stats import distribution

    sample = SampleMetrics(case_id="echo", index=0, phase=Phase.MEASURE,
                           wall_clock_ms=500.0, prompt_tokens=100,
                           completion_tokens=20, cached_tokens=90)
    case = CaseResult(case_id="echo", repetitions=1, warmup=0,
                      samples=[sample], wall_clock_ms=distribution([500.0]))
    run = BenchmarkRun(
        run_id=name, suite="model-capabilities",
        started_at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, 1, 5, tzinfo=UTC),
        fingerprint=Fingerprint(
            host=HostFingerprint(hostname="testbox", cpu_model="Test CPU"),
            runtime=RuntimeFingerprint(base_url="http://127.0.0.1:18080",
                                       host="127.0.0.1", port=18080,
                                       ownership="owned", pid=4242, identity_verified=True),
            model=ModelFingerprint(model_id="fake", n_ctx=65536),
        ),
        cases=[case],
    )
    if invalid:
        run = run.invalidated("runtime_identity", detail="process moved")
    path = tmp_path / f"{name}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_compare_two_valid_artefacts_exits_zero(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "run-aaa")
    b = _write_run(tmp_path, "run-bbb")
    result = runner.invoke(app, ["benchmark", "compare", str(a), str(b)])

    assert result.exit_code == EXIT_OK
    assert "run-aaa" in result.stdout
    assert "run-bbb" in result.stdout
    assert "comparable" in result.stdout.lower()


def test_compare_flags_an_invalid_run_exits_three(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "run-aaa", invalid=True)
    b = _write_run(tmp_path, "run-bbb")
    result = runner.invoke(app, ["benchmark", "compare", str(a), str(b)])

    assert result.exit_code == EXIT_INVALID
    assert "INVALID" in result.stdout


def test_compare_a_missing_artefact_exits_one(tmp_path: Path) -> None:
    b = _write_run(tmp_path, "run-bbb")
    result = runner.invoke(app, ["benchmark", "compare", str(tmp_path / "nope.json"), str(b)])

    assert result.exit_code == EXIT_ERROR
    assert "Traceback" not in result.stdout


# -- against the real model ------------------------------------------------


@pytest.mark.local_llm
def test_capability_runs_against_the_served_model(tmp_path: Path) -> None:
    """The CLI wiring works against a real llama-server (attached, not started
    here — the running server on this machine stays untouched). Asserts nothing
    about the model's answers; capability is a benchmark question, not a unit
    test, and the framework's correctness does not depend on the scores."""
    result = runner.invoke(
        app,
        ["benchmark", "capability", "--attach", "--out", str(tmp_path / "runs")],
        env={**_env(tmp_path), "HARNESS_RUNTIME_ATTACH": "true"},
    )
    assert result.exit_code in {EXIT_OK, EXIT_INVALID, EXIT_ERROR}
    assert "benchmark run" in result.stdout or "identity" in result.stdout.lower()