"""The two outputs of a run: a JSON artefact and something a person reads.

They answer different questions. The artefact is what a later run is compared
against, so it holds every sample including the warmups and every field of the
fingerprint. The summary is what somebody reads to decide whether the run is
worth comparing at all — which is why the first thing it prints is whether the
run was invalidated and whether the serving process was ever verified.

The summary reports percentiles and never a mean. A mean over 500 ms, 510 ms
and 24830 ms is 8613 ms, a figure that describes none of the three samples and
that moves with the warmup policy rather than with the runtime.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from harness.benchmark.models import BenchmarkRun, CaseResult
from harness.benchmark.stats import Distribution
from harness.telemetry.redact import redact

_HEADER = (
    f"{'case':<28}{'reps':>5}{'fail':>6}{'pass':>7}"
    f"{'p50 ms':>11}{'p90 ms':>11}{'p95 ms':>11}{'tok/s p50':>11}{'cache p50':>11}"
)


def artifact_path(run: BenchmarkRun, directory: str | Path) -> Path:
    """Where this run's artefact belongs. Named by run id, nothing else.

    The id already carries the timestamp, so a directory listing is in
    chronological order and two runs a second apart cannot overwrite one
    another.
    """
    return Path(directory) / f"{run.run_id}.json"


def write_run(run: BenchmarkRun, directory: str | Path) -> Path:
    """Write the artefact, creating the directory if needed.

    No redaction happens here: the run was assembled from an already-redacted
    effective configuration and a command line whose credential values were
    replaced when it was recorded. Scrubbing again at write time would hide a
    leak upstream instead of preventing it.
    """
    path = artifact_path(run, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_run(path: str | Path) -> BenchmarkRun:
    """Load an artefact, refusing anything that is not one."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a benchmark artefact must be a JSON object")
    return BenchmarkRun.model_validate(raw)


def invalidate_artifact(path: str | Path, reason: str, *, detail: str | None = None,
                        at: datetime | None = None) -> BenchmarkRun:
    """Record that a stored run should not be compared, keeping its samples.

    Rewriting the same file rather than moving or deleting it: the invalidated
    run is the evidence for the invalidation, and anything that removes it
    turns "this measurement was wrong" into a claim nobody can check.
    """
    path = Path(path)
    updated = read_run(path).invalidated(reason, detail=detail, at=at or datetime.now(UTC))
    path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return updated


def render_summary(run: BenchmarkRun) -> str:
    """A plain-text report. No colours, no rich — this is library output."""
    lines: list[str] = []

    for entry in run.invalidations:
        suffix = f" — {redact(entry.detail)}" if entry.detail else ""
        lines.append(f"INVALID ({entry.reason}){suffix}")

    lines.append(f"benchmark run {run.run_id}   suite={run.suite}")
    lines.append(f"started  {run.started_at.isoformat()}{_duration(run)}")
    lines.extend(_fingerprint_lines(run))

    if run.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {redact(warning)}" for warning in run.warnings)

    lines.append("")
    lines.append(_HEADER)
    lines.extend(_case_line(case) for case in run.cases)
    return "\n".join(lines)


# -- internals -------------------------------------------------------------


def _duration(run: BenchmarkRun) -> str:
    if run.finished_at is None:
        return ""
    seconds = (run.finished_at - run.started_at).total_seconds()
    return f"   finished {run.finished_at.isoformat()}   duration {seconds:.1f} s"


def _fingerprint_lines(run: BenchmarkRun) -> list[str]:
    host = run.fingerprint.host
    runtime = run.fingerprint.runtime
    model = run.fingerprint.model

    where = " / ".join(part for part in (host.hostname, host.cpu_model, host.kernel) if part)
    if host.gpus:
        where += f" / {', '.join(host.gpus)}"

    if runtime.identity_verified:
        identity = f"identity verified (pid {runtime.pid})"
    else:
        detail = redact(runtime.identity_detail) if runtime.identity_detail else ""
        identity = f"identity UNVERIFIED — {detail}" if detail else "identity UNVERIFIED"

    flags = (" ".join(runtime.launch_argv) if runtime.launch_argv
             else "not recorded (this harness did not start the server)")

    return [
        f"host     {where or 'unknown'}",
        f"runtime  {runtime.base_url} [{runtime.ownership or 'unknown'}] — {identity}",
        f"flags    {redact(flags)}",
        f"model    {model.model_id or 'unknown'} — n_ctx {model.n_ctx or 'unknown'}"
        + (f", {model.quantization}" if model.quantization else ""),
    ]


def _case_line(case: CaseResult) -> str:
    label = f"{case.case_id} (+{case.warmup} warmup)"
    return (
        f"{label:<28}{case.repetitions:>5}{case.failures:>6}"
        f"{_rate(case.success_rate):>7}"
        f"{_at(case.wall_clock_ms, 'p50'):>11}{_at(case.wall_clock_ms, 'p90'):>11}"
        f"{_at(case.wall_clock_ms, 'p95'):>11}"
        f"{_at(case.generation_tokens_per_s, 'p50'):>11}"
        f"{_ratio(case.cache_hit_ratio):>11}"
    )


def _at(dist: Distribution | None, field: str) -> str:
    """A percentile, or a dash where nothing was measured."""
    if dist is None:
        return "-"
    return f"{getattr(dist, field):.1f}"


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _ratio(dist: Distribution | None) -> str:
    return "-" if dist is None else f"{dist.p50 * 100:.0f}%"


__all__ = [
    "artifact_path",
    "invalidate_artifact",
    "read_run",
    "render_summary",
    "write_run",
]
