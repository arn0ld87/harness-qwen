"""``harness config show`` and ``harness memory inspect`` (issue #15).

Both commands exist to answer a question that otherwise needs sqlite3 by hand
and four config sources read in the right order. Two properties are therefore
tested harder than the formatting: nothing is printed in the clear that the
redactor would have removed from the journal, and inspecting a database never
writes to it — not even to migrate it into a shape this harness prefers.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.core import RunRuntimeState, StepStatus, TaskState
from harness.memory import MemoryStore

SECRET = "sk-live-0123456789abcdef0123"


@pytest.fixture
def runner() -> CliRunner:
    # A wide terminal so rich does not wrap a tmp_path in the middle of an
    # assertion; the commands are being tested, not the line breaker.
    return CliRunner(env={"COLUMNS": "200"})


def _write_config(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _populate(database: Path) -> None:
    """Two runs: one still open with an uncertain step, one that failed."""
    now = datetime.now(UTC)
    with MemoryStore(database) as store:
        alpha = TaskState(
            run_id="run-alpha",
            goal="wire up the inspect command",
            workspace="/workspace",
            findings=["config layer already resolves provenance"],
            step_index=2,
            created_at=now,
            updated_at=now,
        )
        store.save_task_state(alpha)
        store.save_runtime_state(
            RunRuntimeState(run_id="run-alpha", tool_calls=3, retries_used=1)
        )
        first = store.append_step(
            "run-alpha", step_index=0, action="read the config",
            tool="read_file", arguments={"path": "harness.json"},
        )
        store.complete_step(first, status=StepStatus.DONE, exit_code=0,
                            duration_ms=12.5)
        store.append_step(
            "run-alpha", step_index=1, action="publish the artefact",
            tool="run_command", status=StepStatus.UNCERTAIN,
            arguments={"command": f"curl -H 'Authorization: Bearer {SECRET}'"},
            note="interrupted before its outcome was recorded",
        )

        beta = TaskState(
            run_id="run-beta", goal="run the test suite", workspace="/workspace",
            created_at=now, updated_at=now,
        )
        store.start_run(beta)
        failed = store.append_step(
            "run-beta", step_index=0, action="run pytest", tool="run_command",
        )
        store.complete_step(failed, status=StepStatus.FAILED, exit_code=1,
                            error_kind="execution_failed")

        store.put_fact("cache-cost", "a cold prefix costs 25 s", "architecture")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- config show ----------------------------------------------------------


def test_config_show_names_the_layer_and_the_source(
    runner: CliRunner, tmp_path: Path
) -> None:
    config = _write_config(tmp_path / "harness.json", {"runtime": {"port": 9999}})

    result = runner.invoke(
        app, ["config", "show", "--config", str(config)],
        env={"HARNESS_BUDGET_MAX_STEPS": "7"},
    )

    assert result.exit_code == 0, result.output
    assert "runtime.port" in result.output
    assert "9999" in result.output
    assert "file" in result.output
    assert "HARNESS_BUDGET_MAX_STEPS" in result.output
    assert "env" in result.output


def test_config_show_json_carries_value_origin_and_source(
    runner: CliRunner, tmp_path: Path
) -> None:
    config = _write_config(tmp_path / "harness.json", {"runtime": {"port": 9999}})

    result = runner.invoke(app, ["config", "show", "--config", str(config), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    port = payload["settings"]["runtime.port"]
    assert port["value"] == 9999
    assert port["origin"] == "file"
    assert port["source"] == str(config)
    assert payload["settings"]["budget.max_steps"]["origin"] == "default"


def test_config_show_redacts_declared_and_stray_secrets(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``api_key`` is redacted by name; a key smuggled into a free-form field
    by its text. Either one printed in full is the leak this command would
    otherwise institutionalise."""
    config = _write_config(
        tmp_path / "harness.json",
        {
            "runtime": {"api_key": SECRET},
            "model": {"extra_flags": [f"--api-key={SECRET}"]},
        },
    )

    human = runner.invoke(app, ["config", "show", "--config", str(config)])
    machine = runner.invoke(app, ["config", "show", "--config", str(config), "--json"])

    assert human.exit_code == 0, human.output
    assert machine.exit_code == 0, machine.output
    assert SECRET not in human.output
    assert SECRET not in machine.stdout
    assert "redacted" in human.output
    assert "redacted" in json.dumps(json.loads(machine.stdout))


def test_config_show_explains_an_invalid_setting(
    runner: CliRunner, tmp_path: Path
) -> None:
    config = _write_config(tmp_path / "harness.json", {"runtime": {"port": 99999}})

    result = runner.invoke(app, ["config", "show", "--config", str(config)])

    assert result.exit_code == 1
    assert "runtime.port" in result.output
    assert str(config) in result.output


# -- memory inspect -------------------------------------------------------


def test_memory_inspect_lists_runs_steps_and_facts(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    result = runner.invoke(app, ["memory", "inspect", "--database", str(database)])

    assert result.exit_code == 0, result.output
    assert "run-alpha" in result.output
    assert "run-beta" in result.output
    assert "cache-cost" in result.output


def test_memory_inspect_shows_state_and_uncertain_steps_for_one_run(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    result = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--run", "run-alpha"]
    )

    assert result.exit_code == 0, result.output
    assert "run-beta" not in result.output
    assert "config layer already resolves provenance" in result.output  # TaskState
    assert "Tool calls" in result.output                                 # RuntimeState
    assert "uncertain" in result.output.lower()
    assert "publish the artefact" in result.output


def test_memory_inspect_filters_by_status(runner: CliRunner, tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    result = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--status", "failed",
              "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [entry["run"]["run_id"] for entry in payload["runs"]] == ["run-beta"]
    assert [step["status"] for step in payload["runs"][0]["steps"]] == ["failed"]


def test_memory_inspect_json_carries_task_and_runtime_state(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    result = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--run", "run-alpha",
              "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    entry = payload["runs"][0]
    assert payload["schema_version"] == 2
    assert entry["task_state"]["step_index"] == 2
    assert entry["runtime_state"]["tool_calls"] == 3
    assert entry["step_status_counts"]["uncertain"] == 1
    assert entry["unfinished"] == 1
    assert [fact["key"] for fact in payload["facts"]] == ["cache-cost"]


def test_memory_inspect_redacts_secrets_in_step_arguments(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    human = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--run", "run-alpha"]
    )
    machine = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--json"]
    )

    assert SECRET not in human.output
    assert SECRET not in machine.stdout


def test_memory_inspect_finds_facts_by_keyword(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    hit = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--fact", "prefix",
              "--json"],
    )
    miss = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--fact", "kubernetes",
              "--json"],
    )

    assert [f["key"] for f in json.loads(hit.stdout)["facts"]] == ["cache-cost"]
    assert json.loads(miss.stdout)["facts"] == []


def test_memory_inspect_leaves_the_database_untouched(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)
    before = _digest(database)

    result = runner.invoke(app, ["memory", "inspect", "--database", str(database)])

    assert result.exit_code == 0, result.output
    assert _digest(database) == before


def test_memory_inspect_refuses_an_older_schema_without_migrating(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Opening the store normally would upgrade it. Looking at a database is
    not consent to rewrite it, and the upgrade would happen unannounced."""
    database = tmp_path / "memory.sqlite"
    _populate(database)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    result = runner.invoke(app, ["memory", "inspect", "--database", str(database)])

    assert result.exit_code == 1
    assert "schema version" in result.output
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    connection.close()


def test_memory_inspect_does_not_create_a_missing_database(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "nowhere" / "memory.sqlite"

    result = runner.invoke(app, ["memory", "inspect", "--database", str(database)])

    assert result.exit_code == 1
    assert str(database) in result.output
    assert not database.exists()


def test_memory_inspect_reports_a_file_that_is_not_a_database(
    runner: CliRunner, tmp_path: Path
) -> None:
    database = tmp_path / "memory.sqlite"
    database.write_text("this is not a database", encoding="utf-8")

    result = runner.invoke(app, ["memory", "inspect", "--database", str(database)])

    assert result.exit_code == 1
    assert "SQLite" in result.output


def test_memory_inspect_reports_damaged_state_and_keeps_going(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A store nobody can read is exactly when inspect is needed, so one
    unreadable row degrades that run instead of the whole report."""
    database = tmp_path / "memory.sqlite"
    _populate(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE task_state SET state_json = ? WHERE run_id = ?",
        ('{"run_id": "run-alpha"}', "run-alpha"),
    )
    connection.commit()
    connection.close()

    result = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    entries = {entry["run"]["run_id"]: entry for entry in payload["runs"]}
    assert entries["run-alpha"]["task_state"] is None
    assert entries["run-alpha"]["errors"]
    assert entries["run-beta"]["errors"] == []


def test_memory_inspect_names_an_unknown_run(runner: CliRunner, tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)

    result = runner.invoke(
        app, ["memory", "inspect", "--database", str(database), "--run", "run-gamma"]
    )

    assert result.exit_code == 1
    assert "run-gamma" in result.output


def test_memory_inspect_falls_back_to_the_configured_database(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.sqlite"
    _populate(database)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["memory", "inspect"], env={"HARNESS_DATABASE": str(database)}
    )

    assert result.exit_code == 0, result.output
    assert "run-alpha" in result.output


def test_a_read_only_store_cannot_be_written_to(tmp_path: Path) -> None:
    """The guarantee is SQLite's, not this code's discipline."""
    database = tmp_path / "memory.sqlite"
    _populate(database)

    with MemoryStore(database, read_only=True) as store:
        assert store.get_run("run-alpha") is not None
        with pytest.raises(sqlite3.OperationalError):
            store.put_fact("new", "value")
