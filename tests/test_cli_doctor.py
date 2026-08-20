"""`harness doctor` as a command (issue #16).

The readiness logic is tested in `test_diagnostics.py`; what matters here is
the contract a script depends on: a machine-readable report and an exit code
that means something.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from harness.cli import app

runner = CliRunner()


def test_json_output_is_parseable_and_carries_the_exit_code(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["doctor", "--json", "--no-save"], env={"HARNESS_WORKSPACE": str(tmp_path)}
    )

    payload = json.loads(result.stdout)
    assert isinstance(payload["checks"], list)
    assert payload["exit_code"] == result.exit_code


def test_a_failing_check_exits_non_zero(tmp_path: Path) -> None:
    """Attach configured with nothing listening: a run would not work."""
    result = runner.invoke(
        app,
        ["doctor", "--json", "--no-save"],
        env={
            "HARNESS_WORKSPACE": str(tmp_path),
            "HARNESS_RUNTIME_ATTACH": "true",
            "HARNESS_RUNTIME_PORT": "1",
        },
    )

    assert result.exit_code == 1
    assert any(
        check["severity"] == "fail" for check in json.loads(result.stdout)["checks"]
    )


def test_a_broken_configuration_is_reported_not_traced(tmp_path: Path) -> None:
    """A stack trace for a typo in a config file is a bad answer."""
    result = runner.invoke(
        app,
        ["doctor", "--json", "--no-save"],
        env={"HARNESS_WORKSPACE": str(tmp_path), "HARNESS_RUNTIME_PORT": "not-a-port"},
    )

    assert result.exit_code == 1
    assert "HARNESS_RUNTIME_PORT" in result.stdout
    assert "Traceback" not in result.stdout


def test_doctor_writes_nothing_with_no_save(tmp_path: Path) -> None:
    before = sorted(path.name for path in tmp_path.iterdir())

    runner.invoke(
        app, ["doctor", "--json", "--no-save"], env={"HARNESS_WORKSPACE": str(tmp_path)}
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == before
