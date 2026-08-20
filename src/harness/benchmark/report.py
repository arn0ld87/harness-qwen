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
from harness.benchmark.prefix_invariant import PrefixInvariantReport
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


def render_cache_report(report: PrefixInvariantReport) -> str:
    """A plain-text report of a prefix-invariant run.

    The first line states the verdict — valid, or why not. Then one row per
    step with the prefix hash (short) and the cache accounting side by side,
    so a reader can see the hash hold while the cache hits, and the one step
    where either breaks. Aggregates close the report: the warm cache-hit ratio,
    the reprocessed tokens, and the output those tokens could have produced.
    """
    lines: list[str] = []

    if report.violations:
        for v in report.violations:
            segs = ", ".join(v.segments) or "?"
            lines.append(f"VIOLATION at step {v.at_step} ({v.kind.value}): "
                         f"prefix moved on {segs} — undeclared change")
    if report.anomalies:
        lines.append(f"ANOMALY at step{'s' if len(report.anomalies) > 1 else ''} "
                     f"{', '.join(str(i) for i in report.anomalies)}: "
                     "prefix stable but cache missed")
    verdict = "valid" if report.valid else "INVALID"
    lines.append(f"prefix invariant {verdict}   run {report.run_id}")
    lines.append(f"started  {report.started_at.isoformat()}{_prefix_duration(report)}")

    if report.fingerprint is not None:
        lines.append(f"host     {report.fingerprint.host.hostname or 'unknown'}")
        lines.append(f"model    {report.fingerprint.model.model_id or 'unknown'}")

    lines.append("")
    lines.append(_PREFIX_HEADER)
    lines.extend(_prefix_step_row(step) for step in report.steps)

    lines.append("")
    pct = report.cache_hit_ratio * 100
    lines.append(f"cache hit ratio (warm)   {pct:.0f}%")
    lines.append(f"reprocessed tokens       {report.total_reprocessed_tokens}")
    lines.append(f"discarded output tokens  {report.discarded_output_tokens}")

    if report.invalidations:
        lines.append("")
        lines.append("invalidations:")
        for record in report.invalidations:
            reasons = ", ".join(r.value for r in record.reasons)
            applied = "applied" if record.applied else "declared, not applied"
            lines.append(f"  - {reasons} — {applied}")

    return "\n".join(lines)


def render_comparison(a: BenchmarkRun, b: BenchmarkRun) -> str:
    """Two runs side by side, and whether the delta between them means anything.

    A number from one run is only an answer to "did it change?" when the other
    run measured the same thing on the same machine — otherwise the delta is
    hardware, flags or model file moving, not capability. The comparison states
    that up front: it names what differs in the fingerprint before showing a
    single latency, so a reader is not left to infer comparability from two
    numbers that look similar for unrelated reasons.
    """
    lines: list[str] = []
    lines.append(f"comparison   {a.run_id}  vs  {b.run_id}")
    lines.append(f"  suite      {a.suite}  /  {b.suite}")

    for label, run in (("A", a), ("B", b)):
        if not run.valid:
            reasons = ", ".join(entry.reason for entry in run.invalidations)
            lines.append(f"  run {label}     INVALID ({reasons})")

    comparable, mismatches = _comparable(a, b)
    if comparable:
        lines.append("  fingerprint comparable — host, model and runtime match")
    else:
        lines.append("  fingerprint NOT comparable — the delta is not a capability change:")
        lines.extend(f"    - {redact(reason)}" for reason in mismatches)

    lines.append("")
    lines.append(_COMPARE_HEADER)
    lines.extend(_comparison_rows(a, b))
    return "\n".join(lines)


# -- internals -------------------------------------------------------------


def _duration(run: BenchmarkRun) -> str:
    if run.finished_at is None:
        return ""
    seconds = (run.finished_at - run.started_at).total_seconds()
    return f"   finished {run.finished_at.isoformat()}   duration {seconds:.1f} s"


# -- prefix-invariant report internals --------------------------------------


_PREFIX_HEADER = (
    f"{'#':>2}{'kind':<22}{'prefix hash':<11}"
    f"{'prompt':>8}{'cached':>8}{'reproc':>8}{'hit':>7}"
)


def _prefix_duration(report: PrefixInvariantReport) -> str:
    if report.finished_at is None:
        return ""
    seconds = (report.finished_at - report.started_at).total_seconds()
    return f"   finished {report.finished_at.isoformat()}   duration {seconds:.1f} s"


def _prefix_step_row(step) -> str:
    """One row: index, kind, short hash, prompt/cached/reprocessed tokens, hit%."""
    hit = step.cache_hit_ratio * 100
    return (
        f"{step.index:>2}{step.kind.value:<22}{step.prefix_hash[:8]:<11}"
        f"{step.prompt_tokens:>8}{step.cached_tokens:>8}"
        f"{step.reprocessed_tokens:>8}{hit:>6.0f}%"
    )


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


# -- comparison internals --------------------------------------------------


_COMPARE_HEADER = (
    f"{'case':<28}{'A p50 ms':>11}{'B p50 ms':>11}"
    f"{'A pass':>9}{'B pass':>9}{'A tok/s p50':>13}{'B tok/s p50':>13}"
)


def _comparable(a: BenchmarkRun, b: BenchmarkRun) -> tuple[bool, list[str]]:
    """Same host, model and runtime — the three things that move a result on
    their own. Anything else (flags, n_ctx) is recorded in the fingerprint too,
    but these three are the floor: with one of them different, no per-case
    number is a capability delta."""
    mismatches: list[str] = []
    ha, hb = a.fingerprint.host, b.fingerprint.host
    if ha.hostname != hb.hostname:
        mismatches.append(f"host: {ha.hostname} vs {hb.hostname}")
    if ha.cpu_model != hb.cpu_model:
        mismatches.append(f"cpu: {ha.cpu_model} vs {hb.cpu_model}")
    if ha.gpus != hb.gpus:
        mismatches.append(f"gpus: {ha.gpus} vs {hb.gpus}")
    ma, mb = a.fingerprint.model, b.fingerprint.model
    if ma.model_id != mb.model_id:
        mismatches.append(f"model: {ma.model_id} vs {mb.model_id}")
    if ma.quantization != mb.quantization:
        mismatches.append(f"quantization: {ma.quantization} vs {mb.quantization}")
    if a.fingerprint.runtime.base_url != b.fingerprint.runtime.base_url:
        mismatches.append(
            f"runtime: {a.fingerprint.runtime.base_url} vs "
            f"{b.fingerprint.runtime.base_url}"
        )
    return not mismatches, mismatches


def _comparison_rows(a: BenchmarkRun, b: BenchmarkRun) -> list[str]:
    by_id_a = {case.case_id: case for case in a.cases}
    by_id_b = {case.case_id: case for case in b.cases}
    # Stable order: A's cases first (in file order), then anything only in B.
    order = [case.case_id for case in a.cases]
    order += [cid for cid in (by_id_b.keys() - by_id_a.keys())]

    rows: list[str] = []
    for cid in order:
        ca = by_id_a.get(cid)
        cb = by_id_b.get(cid)
        label = cid
        if ca is None:
            label = f"{cid} (only B)"
        elif cb is None:
            label = f"{cid} (only A)"
        rows.append(
            f"{label:<28}"
            f"{_at(ca.wall_clock_ms if ca else None, 'p50'):>11}"
            f"{_at(cb.wall_clock_ms if cb else None, 'p50'):>11}"
            f"{_rate(ca.success_rate if ca else None):>9}"
            f"{_rate(cb.success_rate if cb else None):>9}"
            f"{_at(ca.generation_tokens_per_s if ca else None, 'p50'):>13}"
            f"{_at(cb.generation_tokens_per_s if cb else None, 'p50'):>13}"
        )
    return rows


__all__ = [
    "artifact_path",
    "invalidate_artifact",
    "read_run",
    "render_cache_report",
    "render_comparison",
    "render_summary",
    "write_run",
]
