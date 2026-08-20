#!/usr/bin/env python3
"""Enforce the coverage floors recorded in ``pyproject.toml`` (issue #9).

``--cov-fail-under`` only knows one number for the whole tree, which is the
wrong shape for this repository: the agent loop, the security classifier and
the memory store carry the risk, while the CLI and the llama.cpp client are
mostly I/O that unit tests should not be pretending to cover. A single number
either lets a critical module rot or blocks work over a peripheral one.

So the floors are per package, they live in configuration rather than code,
and they start just below where the suite already stands — a gate that is red
on the day it ships teaches everyone to skip it.

Usage:
    uv run pytest --cov=harness --cov-report=json
    uv run python scripts/coverage_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "src/harness/"


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Minimum statement coverage: overall, per package, per file.

    File floors exist because a package average hides its riskiest member.
    ``tools`` sits near 44% mostly because ``filesystem.py`` and ``git.py``
    have no unit tests yet, which says nothing about ``shell.py`` — the
    bubblewrap boundary, and the file where an untested path is a security
    claim nobody checked.
    """

    total: float = 0.0
    modules: dict[str, float] = field(default_factory=dict)
    files: dict[str, float] = field(default_factory=dict)


def load_thresholds(pyproject: Path) -> Thresholds:
    """Read the floors, refusing a configuration that enforces nothing.

    Fail-closed on purpose: a dropped or misspelled ``[tool.coverage_gate]``
    would otherwise leave the gate reporting success while checking nothing,
    which is worse than having no gate — it looks like protection.
    """
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle).get("tool", {}).get("coverage_gate", {})
    modules = {name: float(value) for name, value in config.get("modules", {}).items()}
    files = {name: float(value) for name, value in config.get("files", {}).items()}
    if not modules:
        sys.exit(
            f"{pyproject}: [tool.coverage_gate.modules] is missing or empty, so "
            "the gate would enforce nothing. Restore the floors rather than "
            "running without them."
        )
    return Thresholds(
        total=float(config.get("total", 0.0)), modules=modules, files=files
    )


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(
            f"coverage report not found at {path}. Run "
            "'uv run pytest --cov=harness --cov-report=json' first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _module_of(path: str) -> str | None:
    """Package a measured file belongs to, or None for the top level."""
    normalised = path.replace("\\", "/")
    index = normalised.find(PACKAGE_ROOT)
    if index < 0:
        return None
    tail = normalised[index + len(PACKAGE_ROOT) :]
    head, slash, _ = tail.partition("/")
    return head if slash else None


def check(report: dict[str, Any], thresholds: Thresholds) -> list[str]:
    """Return one message per breached floor; empty means the gate passes.

    Every breach is reported, not just the first: a run that names one gap at
    a time turns a single fix into a sequence of pushes.
    """
    failures: list[str] = []

    total = float(report.get("totals", {}).get("percent_covered", 0.0))
    if total < thresholds.total:
        failures.append(
            f"total coverage {total:.1f}% is below the required {thresholds.total:.1f}%"
        )

    # Pooled per package: statements, not an average of per-file percentages,
    # which would weigh a ten-line module like a three-hundred-line one.
    covered: dict[str, int] = {name: 0 for name in thresholds.modules}
    statements: dict[str, int] = {name: 0 for name in thresholds.modules}
    for path, data in report.get("files", {}).items():
        module = _module_of(path)
        if module is None or module not in thresholds.modules:
            continue
        summary = data.get("summary", {})
        covered[module] += int(summary.get("covered_lines", 0))
        statements[module] += int(summary.get("num_statements", 0))

    measured = {
        path.replace("\\", "/"): data.get("summary", {})
        for path, data in report.get("files", {}).items()
    }
    for wanted, floor in sorted(thresholds.files.items()):
        summary = next(
            (s for path, s in measured.items() if path.endswith(wanted)), None
        )
        if summary is None or int(summary.get("num_statements", 0)) == 0:
            failures.append(
                f"file {wanted!r} has a {floor:.1f}% floor but was not measured "
                "— was it renamed, or left out of --cov?"
            )
            continue
        statements_here = int(summary.get("num_statements", 0))
        covered_here = int(summary.get("covered_lines", 0))
        percent = 100.0 * covered_here / statements_here
        if percent < floor:
            failures.append(
                f"file {wanted!r} at {percent:.1f}% is below its {floor:.1f}% "
                f"floor ({covered_here}/{statements_here} statements)"
            )

    for module, floor in sorted(thresholds.modules.items()):
        if statements[module] == 0:
            failures.append(
                f"module {module!r} has a {floor:.1f}% floor but no measured "
                "files — was it renamed, or left out of --cov?"
            )
            continue
        percent = 100.0 * covered[module] / statements[module]
        if percent < floor:
            failures.append(
                f"module {module!r} at {percent:.1f}% is below its "
                f"{floor:.1f}% floor ({covered[module]}/{statements[module]} statements)"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "coverage.json",
        help="coverage.json written by 'pytest --cov-report=json'",
    )
    parser.add_argument(
        "--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml",
        help="file holding the [tool.coverage_gate] floors",
    )
    args = parser.parse_args(argv)

    thresholds = load_thresholds(args.pyproject)
    failures = check(load_report(args.report), thresholds)
    if failures:
        print("Coverage gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nFloors live in [tool.coverage_gate] in pyproject.toml.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Coverage gate passed ({len(thresholds.modules)} module floors, "
        f"{len(thresholds.files)} file floors)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
