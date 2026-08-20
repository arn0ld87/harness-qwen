"""The upgrade path between schema versions (issue #25).

A resume point that does not survive an upgrade is worse than no resume point:
the run believes it can continue and continues from a lie. These tests pin the
three properties the store depends on — an older database moves forward with
its rows intact, a migration that fails leaves the database exactly as it was,
and a database from a newer harness is refused rather than read.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness.memory import MemoryStore
from harness.memory import migrations as migrations_module
from harness.memory.migrations import (
    SCHEMA_VERSION,
    MigrationError,
    SchemaVersionError,
    check_schema,
    ensure_schema,
    object_exists,
    schema_version,
)

FIXTURE_V1 = Path(__file__).parent / "fixtures" / "memory" / "schema_v1.sql"


def _v1_database(path: Path) -> Path:
    """A version 1 database carrying one run, one step, one fact."""
    conn = sqlite3.connect(path)
    conn.executescript(FIXTURE_V1.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO runs (run_id, goal, workspace, created_at, updated_at, "
        "steps_taken) VALUES ('run-v1', 'survive the upgrade', '/w', "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1)"
    )
    conn.execute(
        "INSERT INTO steps (run_id, step_index, action, status, started_at) "
        "VALUES ('run-v1', 0, 'tool_call', 'done', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO task_state (run_id, step_index, state_json, updated_at) "
        "VALUES ('run-v1', 1, '{\"run_id\": \"run-v1\"}', "
        "'2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO facts (key, value, category, created_at, updated_at) "
        "VALUES ('prefix', 'the prefix is sacred', 'architecture', "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    return path


def test_a_version_one_database_migrates_with_its_rows_intact(tmp_path: Path) -> None:
    database = _v1_database(tmp_path / "memory.sqlite")

    with MemoryStore(database) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert store.has_table("runtime_state")
        assert [row["run_id"] for row in store.list_runs()] == ["run-v1"]
        assert len(store.get_steps("run-v1")) == 1
        assert [fact.key for fact in store.get_facts()] == ["prefix"]


def test_migrating_twice_is_a_no_op(tmp_path: Path) -> None:
    """The version gate, not the migration body, is what makes this safe."""
    database = _v1_database(tmp_path / "memory.sqlite")

    with MemoryStore(database):
        pass
    with MemoryStore(database) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert [row["run_id"] for row in store.list_runs()] == ["run-v1"]


def test_a_failing_migration_leaves_the_database_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half a migration is the one outcome nobody can recover from by hand."""
    database = _v1_database(tmp_path / "memory.sqlite")

    def _half_way(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE runtime_state (run_id TEXT PRIMARY KEY)")
        raise sqlite3.OperationalError("disk went away mid-upgrade")

    monkeypatch.setitem(migrations_module.MIGRATIONS, 1, _half_way)

    conn = sqlite3.connect(database)
    try:
        with pytest.raises(MigrationError) as caught:
            ensure_schema(conn, "fixture")
        assert "1 to 2" in str(caught.value)
        assert schema_version(conn) == 1
        assert not object_exists(conn, "table", "runtime_state")
    finally:
        conn.close()


def test_a_newer_schema_version_is_refused_by_both_doors(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    with MemoryStore(database):
        pass
    conn = sqlite3.connect(database)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()

    try:
        with pytest.raises(SchemaVersionError, match="newer than the supported"):
            ensure_schema(conn, "fixture")
        with pytest.raises(SchemaVersionError, match="not the supported"):
            check_schema(conn, "fixture")
    finally:
        conn.close()

    with pytest.raises(SchemaVersionError):
        MemoryStore(database)


def test_a_version_with_no_registered_migration_is_refused(tmp_path: Path) -> None:
    """A gap in the chain stops the upgrade instead of skipping a step."""
    database = _v1_database(tmp_path / "memory.sqlite")
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
        with pytest.raises(SchemaVersionError, match="predates schema versioning"):
            ensure_schema(conn, "fixture")
    finally:
        conn.close()


def test_a_fresh_database_is_created_atomically(tmp_path: Path) -> None:
    """Either the whole schema and its version land, or neither does."""
    database = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(database)
    try:
        assert ensure_schema(conn, "fresh") == SCHEMA_VERSION
        assert schema_version(conn) == SCHEMA_VERSION
        assert object_exists(conn, "table", "runs")
    finally:
        conn.close()
