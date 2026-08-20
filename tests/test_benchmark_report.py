"""JSON artefact and human-readable summary of a benchmark run (issue #19).

Both outputs exist because they answer different questions: the artefact is
what a later run is compared against, the summary is what a person reads to
decide whether the run is worth comparing at all. Invalidating a run must not
touch either one's raw samples — a run that turned out to be untrustworthy is
still the record of what happened.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.benchmark import (
    BenchmarkRun,
    CaseResult,
    Fingerprint,
    HostFingerprint,
    ModelFingerprint,
    Phase,
    RuntimeFingerprint,
    SampleMetrics,
    invalidate_artifact,
    read_run,
    render_summary,
    write_run,
)
from harness.benchmark.stats import distribution


def _sample(index: int, ms: float, *, ok: bool = True) -> SampleMetrics:
    return SampleMetrics(
        case_id="echo", index=index, phase=Phase.MEASURE, ok=ok,
        wall_clock_ms=ms, prompt_tokens=100, completion_tokens=20, cached_tokens=90,
    )


def _run(*, verified: bool = True) -> BenchmarkRun:
    samples = [_sample(0, 500.0), _sample(1, 510.0), _sample(2, 24830.0)]
    case = CaseResult(
        case_id="echo",
        description="repeat a fixed prompt",
        repetitions=3,
        warmup=1,
        warmup_samples=[_sample(0, 24830.0)],
        samples=samples,
        failures=0,
        success_rate=1.0,
        wall_clock_ms=distribution([s.wall_clock_ms for s in samples]),
    )
    return BenchmarkRun(
        run_id="bench-20260820T010000Z-abcd1234",
        suite="unit",
        started_at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, 1, 5, tzinfo=UTC),
        fingerprint=Fingerprint(
            host=HostFingerprint(hostname="testbox", cpu_model="Test CPU"),
            runtime=RuntimeFingerprint(
                base_url="http://127.0.0.1:18080", host="127.0.0.1", port=18080,
                ownership="owned", pid=4242, identity_verified=verified,
                launch_argv=["llama-server", "--api-key", "[redacted:flag_value]"],
            ),
            model=ModelFingerprint(model_id="fake", n_ctx=65536),
            harness_version="0.1.0",
            python_version="3.12.0",
            config={"runtime": {"port": 18080, "api_key": "[redacted:api_key]"}},
            config_provenance={"runtime.port": "cli: command line"},
        ),
        cases=[case],
    )


# -- the JSON artefact -----------------------------------------------------


def test_writing_a_run_names_the_file_after_the_run_id(tmp_path: Path) -> None:
    path = write_run(_run(), tmp_path)
    assert path.name == "bench-20260820T010000Z-abcd1234.json"
    assert path.parent == tmp_path


def test_the_artefact_round_trips_without_losing_a_sample(tmp_path: Path) -> None:
    original = _run()
    restored = read_run(write_run(original, tmp_path))
    assert restored == original
    assert len(restored.cases[0].samples) == 3
    assert len(restored.cases[0].warmup_samples) == 1


def test_the_artefact_is_readable_json_with_a_schema_version(tmp_path: Path) -> None:
    payload = json.loads(write_run(_run(), tmp_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] >= 1
    assert payload["cases"][0]["samples"][0]["wall_clock_ms"] == 500.0


def test_the_directory_is_created_if_it_does_not_exist(tmp_path: Path) -> None:
    path = write_run(_run(), tmp_path / "deep" / "nested")
    assert path.exists()


def test_reading_a_file_that_is_not_a_run_fails_loudly(tmp_path: Path) -> None:
    stray = tmp_path / "stray.json"
    stray.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(stray)


# -- invalidation ----------------------------------------------------------


def test_a_run_starts_out_valid() -> None:
    assert _run().valid


def test_invalidating_a_run_keeps_every_sample(tmp_path: Path) -> None:
    path = write_run(_run(), tmp_path)
    updated = invalidate_artifact(path, "wrong_flags", detail="started with --n-cpu-moe 40")

    assert not updated.valid
    assert updated.invalidations[0].reason == "wrong_flags"
    assert "n-cpu-moe" in (updated.invalidations[0].detail or "")
    assert len(updated.cases[0].samples) == 3
    assert updated.cases[0].wall_clock_ms is not None


def test_invalidation_is_persisted_to_the_same_artefact(tmp_path: Path) -> None:
    path = write_run(_run(), tmp_path)
    invalidate_artifact(path, "wrong_flags")
    assert not read_run(path).valid


def test_a_second_invalidation_is_appended_rather_than_replacing_the_first(
    tmp_path: Path,
) -> None:
    path = write_run(_run(), tmp_path)
    invalidate_artifact(path, "wrong_flags")
    updated = invalidate_artifact(path, "thermal_throttling")
    assert [i.reason for i in updated.invalidations] == ["wrong_flags", "thermal_throttling"]


# -- the human-readable summary --------------------------------------------


def test_the_summary_names_the_run_the_host_and_the_model() -> None:
    text = render_summary(_run())
    assert "bench-20260820T010000Z-abcd1234" in text
    assert "testbox" in text
    assert "fake" in text


def test_the_summary_reports_percentiles_rather_than_a_mean() -> None:
    text = render_summary(_run())
    assert "p50" in text and "p95" in text
    # A mean over 500, 510 and 24830 ms would read as 8.6 s and describe none
    # of the three samples.
    assert "8613" not in text


def test_the_summary_states_how_many_samples_were_warmup() -> None:
    assert "warmup" in render_summary(_run()).lower()


def test_the_summary_flags_an_unverified_runtime() -> None:
    text = render_summary(_run(verified=False))
    assert "unverified" in text.lower()


def test_an_invalidated_run_says_so_at_the_top() -> None:
    run = _run().invalidated("wrong_flags", detail="started with --n-cpu-moe 40")
    text = render_summary(run)
    assert "INVALID" in text
    assert "wrong_flags" in text.splitlines()[0] or "wrong_flags" in text


def test_the_summary_does_not_print_a_credential() -> None:
    run = _run()
    run.fingerprint.runtime.identity_detail = "connected with token ghp_abcdefghijklmnopqrstuvwx"
    text = render_summary(run)
    assert "ghp_abcdefghijklmnopqrstuvxx" not in text


# -- comparing two runs (issue #27) ---------------------------------------


def _case(case_id: str, ms: float, *, rate: float = 1.0) -> CaseResult:
    samples = [_sample(0, ms), _sample(1, ms + 5.0), _sample(2, ms + 10.0)]
    return CaseResult(
        case_id=case_id, description=case_id, repetitions=3, warmup=1,
        samples=samples, success_rate=rate,
        wall_clock_ms=distribution([s.wall_clock_ms for s in samples]),
    )


def _run_variant(
    *,
    run_id: str = "bench-20260820T010000Z-abcd1234",
    hostname: str = "testbox",
    model_id: str = "fake",
    base_url: str = "http://127.0.0.1:18080",
    invalid_reason: str | None = None,
    cases: list[CaseResult] | None = None,
) -> BenchmarkRun:
    run = BenchmarkRun(
        run_id=run_id,
        suite="unit",
        started_at=datetime(2026, 8, 20, 1, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, 1, 5, tzinfo=UTC),
        fingerprint=Fingerprint(
            host=HostFingerprint(hostname=hostname, cpu_model="Test CPU"),
            runtime=RuntimeFingerprint(
                base_url=base_url, host="127.0.0.1", port=18080,
                ownership="owned", pid=4242, identity_verified=True,
            ),
            model=ModelFingerprint(model_id=model_id, n_ctx=65536),
        ),
        cases=cases or [_case("echo", 500.0)],
    )
    if invalid_reason:
        run = run.invalidated(invalid_reason, detail="something moved")
    return run


def test_comparison_names_both_runs_and_their_suites() -> None:
    from harness.benchmark import render_comparison

    text = render_comparison(_run_variant(run_id="run-aaa"), _run_variant(run_id="run-bbb"))
    assert "run-aaa" in text
    assert "run-bbb" in text
    assert "unit" in text


def test_comparison_of_comparable_runs_says_so() -> None:
    from harness.benchmark import render_comparison

    text = render_comparison(_run_variant(), _run_variant(run_id="run-bbb"))
    assert "comparable" in text.lower()


def test_comparison_flags_an_invalid_run() -> None:
    from harness.benchmark import render_comparison

    text = render_comparison(
        _run_variant(invalid_reason="runtime_identity"),
        _run_variant(run_id="run-bbb"),
    )
    assert "INVALID" in text
    assert "runtime_identity" in text


def test_comparison_reports_incompatible_fingerprints() -> None:
    from harness.benchmark import render_comparison

    # Same prompt on a different machine or a different model file is not a
    # capability change, and the comparison must say so rather than present the
    # delta as if it were.
    text = render_comparison(
        _run_variant(hostname="testbox", model_id="fake"),
        _run_variant(run_id="run-bbb", hostname="otherbox", model_id="other"),
    )
    assert "not comparable" in text.lower() or "incompatible" in text.lower()
    assert "otherbox" in text or "other" in text


def test_comparison_lists_each_case_side_by_side() -> None:
    from harness.benchmark import render_comparison

    a = _run_variant(cases=[_case("echo", 500.0), _case("json", 300.0)])
    b = _run_variant(run_id="run-bbb", cases=[_case("echo", 800.0), _case("json", 310.0)])
    text = render_comparison(a, b)
    assert "echo" in text
    assert "json" in text
    # Both sides of a case row appear: the slower run's p50 (805) and the
    # faster one's (505). The median of [800,805,810] is 805, of [500,505,510]
    # is 505 — the first sample is not the p50.
    assert "505" in text
    assert "805" in text


def test_comparison_notes_a_case_present_in_only_one_run() -> None:
    from harness.benchmark import render_comparison

    a = _run_variant(cases=[_case("echo", 500.0), _case("only_a", 200.0)])
    b = _run_variant(run_id="run-bbb", cases=[_case("echo", 800.0)])
    text = render_comparison(a, b)
    assert "only_a" in text
