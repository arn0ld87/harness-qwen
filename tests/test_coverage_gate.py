"""The coverage gate itself (issue #9).

A gate that miscounts is worse than no gate: it either blocks work for no
reason or waves through the module it was meant to protect. These tests pin
the arithmetic and the failure reporting against hand-written reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from coverage_gate import (  # noqa: E402
    Thresholds,
    check,
    load_report,
    load_thresholds,
)

REPO = Path(__file__).resolve().parents[1]


def _report(files: dict[str, tuple[int, int]], total: float) -> dict[str, object]:
    """Build a coverage.json-shaped report from {path: (covered, missing)}."""
    return {
        "totals": {"percent_covered": total},
        "files": {
            path: {
                "summary": {
                    "covered_lines": covered,
                    "missing_lines": missing,
                    "num_statements": covered + missing,
                    "percent_covered": 100.0 * covered / (covered + missing),
                }
            }
            for path, (covered, missing) in files.items()
        },
    }


def test_passing_report_reports_nothing() -> None:
    report = _report(
        {
            "src/harness/agent/loop.py": (90, 10),
            "src/harness/security/rules.py": (50, 50),
        },
        total=70.0,
    )
    thresholds = Thresholds(total=65.0, modules={"agent": 80.0, "security": 40.0})
    assert check(report, thresholds) == []


def test_module_below_its_floor_is_named_with_numbers() -> None:
    report = _report({"src/harness/security/rules.py": (30, 70)}, total=90.0)
    thresholds = Thresholds(total=50.0, modules={"security": 40.0})

    failures = check(report, thresholds)

    assert len(failures) == 1
    assert "security" in failures[0]
    assert "30.0%" in failures[0]
    assert "40.0%" in failures[0]


def test_module_coverage_is_pooled_across_its_files() -> None:
    """Per-file percentages do not average — statements do.

    Two files at 50% and 100% are not 75% of the module unless they happen to
    be the same size, and a gate that assumed so would drift as files grow.
    """
    report = _report(
        {
            "src/harness/agent/loop.py": (10, 90),   # 10%
            "src/harness/agent/retry.py": (10, 0),   # 100%
        },
        total=100.0,
    )
    thresholds = Thresholds(total=0.0, modules={"agent": 20.0})

    failures = check(report, thresholds)

    assert len(failures) == 1
    assert "18.2%" in failures[0]  # 20 of 110 statements, not 55%


def test_total_below_floor_is_reported() -> None:
    report = _report({"src/harness/agent/loop.py": (90, 10)}, total=61.4)
    failures = check(report, Thresholds(total=65.0, modules={}))
    assert len(failures) == 1
    assert "total" in failures[0]
    assert "61.4%" in failures[0]


def test_every_failure_is_reported_not_just_the_first() -> None:
    """One CI run should surface every gap, not one per push."""
    report = _report(
        {
            "src/harness/agent/loop.py": (10, 90),
            "src/harness/memory/store.py": (10, 90),
        },
        total=10.0,
    )
    thresholds = Thresholds(total=50.0, modules={"agent": 80.0, "memory": 80.0})

    failures = check(report, thresholds)

    assert len(failures) == 3


def test_a_configured_module_with_no_measured_files_fails_loudly() -> None:
    """A renamed or unmeasured package must not silently pass its gate."""
    report = _report({"src/harness/agent/loop.py": (90, 10)}, total=90.0)
    thresholds = Thresholds(total=0.0, modules={"retrieval": 50.0})

    failures = check(report, thresholds)

    assert len(failures) == 1
    assert "retrieval" in failures[0]
    assert "no measured files" in failures[0]


def test_thresholds_come_from_pyproject() -> None:
    """The gate is configuration, so raising it needs no code change."""
    thresholds = load_thresholds(REPO / "pyproject.toml")
    assert thresholds.total > 0
    # The five modules the issue calls critical.
    assert {"agent", "security", "context", "memory", "protocol"} <= set(
        thresholds.modules
    )


def test_load_report_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_report(tmp_path / "absent.json")


def test_load_report_reads_what_coverage_writes(tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report({"src/harness/x.py": (1, 1)}, 50.0)))
    assert load_report(path)["totals"]["percent_covered"] == 50.0


def test_missing_configuration_is_refused_not_ignored(tmp_path: Path) -> None:
    """A gate that enforces nothing must say so, not report success.

    A dropped or renamed ``[tool.coverage_gate]`` section would otherwise
    print "passed" while checking no floors at all — the failure mode the rest
    of this repo treats as fail-closed.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_thresholds(pyproject)


def test_configuration_without_module_floors_is_refused(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.coverage_gate]\ntotal = 65.0\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        load_thresholds(pyproject)


def test_file_floor_catches_what_a_package_floor_hides() -> None:
    """One risky file can rot while its package average stays comfortable."""
    report = _report(
        {
            "src/harness/tools/shell.py": (40, 60),      # 40%
            "src/harness/tools/compression.py": (95, 5),
        },
        total=90.0,
    )
    thresholds = Thresholds(
        total=0.0,
        modules={"tools": 40.0},
        files={"src/harness/tools/shell.py": 85.0},
    )

    failures = check(report, thresholds)

    assert len(failures) == 1
    assert "shell.py" in failures[0]
    assert "40.0%" in failures[0]


def test_configured_file_that_is_not_measured_fails_loudly() -> None:
    report = _report({"src/harness/tools/compression.py": (95, 5)}, total=95.0)
    thresholds = Thresholds(
        total=0.0, modules={}, files={"src/harness/tools/shell.py": 85.0}
    )

    failures = check(report, thresholds)

    assert len(failures) == 1
    assert "not measured" in failures[0]


def test_file_floors_come_from_pyproject() -> None:
    thresholds = load_thresholds(REPO / "pyproject.toml")
    assert "src/harness/tools/shell.py" in thresholds.files
