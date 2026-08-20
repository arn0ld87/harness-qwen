"""SQLite-backed working memory: runs, step history and the resume point.

A cold prompt costs ~25 s on this machine, which is the entire reason this
subsystem exists — restarting a killed run from step zero is measurably more
expensive than writing every step to disk before it executes.

Everything goes through the stdlib ``sqlite3`` module. No ORM: the schema is
four tables plus an optional index, and an ORM would only hide the transaction
boundaries that make resume correct.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.core import PlanStep, RunResult, RunRuntimeState, StepStatus, TaskState
from harness.memory.facts import Fact, FactStore, fts5_available
from harness.memory.migrations import (
    SCHEMA_VERSION,
    StoreError,
    UnknownRunError,
    configure,
    ensure_schema,
    object_exists,
)


class StepRecord(BaseModel):
    """One attempted step, as recorded before it ran and updated after."""

    id: int
    run_id: str
    step_index: int
    role: str | None = None
    action: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    status: StepStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None
    exit_code: int | None = None
    error_kind: str | None = None
    note: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    """Durable store for run state, step history and persistent facts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread is off because the agent loop may hand a blocking
        # call to a worker thread; the lock is what keeps that safe.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        configure(self._conn)
        self.schema_version = ensure_schema(self._conn, str(self.path))
        self.facts = FactStore(self._conn, self._lock, fts5=fts5_available())

    @property
    def fts5(self) -> bool:
        return self.facts.fts5

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def journal_mode(self) -> str:
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    def has_table(self, name: str) -> bool:
        return object_exists(self._conn, "table", name)

    # -- runs --------------------------------------------------------------

    def _upsert_run(self, state: TaskState, *, overwrite_goal: bool) -> None:
        goal_clause = (
            "goal = excluded.goal, workspace = excluded.workspace,"
            if overwrite_goal
            else ""
        )
        self._conn.execute(
            f"""
            INSERT INTO runs (run_id, goal, workspace, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                {goal_clause}
                updated_at = excluded.updated_at
            """,
            (
                state.run_id,
                state.goal,
                state.workspace,
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
            ),
        )

    def start_run(self, state: TaskState) -> None:
        """Register a run. Idempotent, so a resumed run reuses its row."""
        with self._lock, self._conn:
            self._upsert_run(state, overwrite_goal=True)

    def finish_run(self, result: RunResult) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE runs SET
                    updated_at = ?, stop_reason = ?, answer = ?, verified = ?,
                    steps_taken = ?, tool_calls = ?, retries = ?,
                    prompt_tokens = ?, completion_tokens = ?, cached_tokens = ?,
                    elapsed_s = ?
                WHERE run_id = ?
                """,
                (
                    _now(), str(result.stop_reason), result.answer,
                    int(result.verified), result.steps_taken, result.tool_calls,
                    result.retries, result.total_prompt_tokens,
                    result.total_completion_tokens, result.total_cached_tokens,
                    result.elapsed_s, result.run_id,
                ),
            )
            if cur.rowcount == 0:
                raise UnknownRunError(f"no run with id {result.run_id!r}")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM runs WHERE run_id = ?", (run_id,)
            ).rowcount > 0

    # -- task state --------------------------------------------------------

    def save_task_state(self, state: TaskState) -> None:
        """Persist the resume point.

        Writes the ``runs`` row in the same transaction, so a caller that only
        ever saves state still ends up with a referentially complete database
        and a partially written pair is impossible.
        """
        with self._lock, self._conn:
            self._upsert_run(state, overwrite_goal=False)
            self._conn.execute(
                """
                INSERT INTO task_state (run_id, step_index, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    step_index = excluded.step_index,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state.run_id,
                    state.step_index,
                    state.model_dump_json(),
                    state.updated_at.isoformat(),
                ),
            )

    def load_task_state(self, run_id: str) -> TaskState | None:
        row = self._conn.execute(
            "SELECT state_json FROM task_state WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return TaskState.model_validate_json(row["state_json"])
        except ValidationError as exc:
            raise StoreError(
                f"stored task state for run {run_id!r} does not match the current "
                f"TaskState model: {exc}"
            ) from exc

    def resumable_runs(self) -> list[str]:
        """Runs that have a saved state but no recorded stop reason."""
        rows = self._conn.execute(
            "SELECT t.run_id FROM task_state t JOIN runs r USING (run_id) "
            "WHERE r.stop_reason IS NULL ORDER BY t.updated_at DESC"
        ).fetchall()
        return [r["run_id"] for r in rows]

    def save_runtime_state(self, state: RunRuntimeState) -> None:
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    """
                    INSERT INTO runtime_state (run_id, runtime_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        runtime_json = excluded.runtime_json,
                        updated_at = excluded.updated_at
                    """,
                    (state.run_id, state.model_dump_json(), _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise UnknownRunError(
                    f"cannot save runtime state for unknown run {state.run_id!r}"
                ) from exc

    def load_runtime_state(self, run_id: str) -> RunRuntimeState | None:
        row = self._conn.execute(
            "SELECT runtime_json FROM runtime_state WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return RunRuntimeState.model_validate_json(row["runtime_json"])
        except ValidationError as exc:
            raise StoreError(
                f"stored runtime state for run {run_id!r} is invalid: {exc}"
            ) from exc

    # -- steps -------------------------------------------------------------

    def append_step(
        self,
        run_id: str,
        *,
        step_index: int,
        action: str,
        role: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        status: StepStatus = StepStatus.RUNNING,
        note: str | None = None,
    ) -> int:
        """Record a step before it executes and return its row id.

        Written ahead of execution on purpose: a step missing from the history
        is one that never started, and a step left at RUNNING is one that was
        killed mid-flight. Both are recoverable states. A step recorded only on
        success would be neither.
        """
        with self._lock, self._conn:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO steps (run_id, step_index, role, action, tool,
                                       arguments, status, started_at, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id, step_index, role, action, tool,
                        json.dumps(arguments) if arguments is not None else None,
                        str(status), _now(), note,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise UnknownRunError(
                    f"cannot append a step to unknown run {run_id!r}"
                ) from exc
            return int(cur.lastrowid or 0)

    def complete_step(
        self,
        step_id: int,
        *,
        status: StepStatus,
        exit_code: int | None = None,
        error_kind: str | None = None,
        duration_ms: float | None = None,
        note: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE steps SET status = ?, finished_at = ?, exit_code = ?,
                                 error_kind = ?, duration_ms = ?,
                                 note = COALESCE(?, note)
                WHERE id = ?
                """,
                (str(status), _now(), exit_code, error_kind, duration_ms,
                 note, step_id),
            )
            if cur.rowcount == 0:
                raise StoreError(f"no step with id {step_id}")

    def save_tool_checkpoint(
        self,
        *,
        step_id: int,
        status: StepStatus,
        task_state: TaskState,
        runtime_state: RunRuntimeState,
        exit_code: int | None = None,
        error_kind: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Commit tool completion and both resume states atomically."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE steps SET status = ?, finished_at = ?, exit_code = ?,
                                 error_kind = ?, duration_ms = ?
                WHERE id = ? AND run_id = ?
                """,
                (
                    str(status),
                    _now(),
                    exit_code,
                    error_kind,
                    duration_ms,
                    step_id,
                    task_state.run_id,
                ),
            )
            if cur.rowcount == 0:
                raise StoreError(
                    f"no step {step_id} for run {task_state.run_id!r}"
                )
            self._upsert_run(task_state, overwrite_goal=False)
            self._conn.execute(
                """
                INSERT INTO task_state (run_id, step_index, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    step_index = excluded.step_index,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    task_state.run_id,
                    task_state.step_index,
                    task_state.model_dump_json(),
                    task_state.updated_at.isoformat(),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO runtime_state (run_id, runtime_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    runtime_json = excluded.runtime_json,
                    updated_at = excluded.updated_at
                """,
                (runtime_state.run_id, runtime_state.model_dump_json(), _now()),
            )

    def get_steps(self, run_id: str) -> list[StepRecord]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index, id", (run_id,)
        ).fetchall()
        return [_to_step(r) for r in rows]

    def unfinished_steps(self, run_id: str) -> list[StepRecord]:
        """Steps recorded but never completed — what a killed run left behind."""
        return [s for s in self.get_steps(run_id) if s.finished_at is None]

    # -- facts -------------------------------------------------------------

    def put_fact(self, key: str, value: str, category: str = "general") -> None:
        self.facts.put(key, value, category)

    def get_fact(self, key: str) -> Fact | None:
        return self.facts.get(key)

    def get_facts(self, category: str | None = None) -> list[Fact]:
        return self.facts.list(category)

    def delete_fact(self, key: str) -> bool:
        return self.facts.delete(key)

    def search_facts(
        self, query: str, *, category: str | None = None, limit: int = 20
    ) -> list[Fact]:
        return self.facts.search(query, category=category, limit=limit)


def _to_step(row: sqlite3.Row) -> StepRecord:
    data = dict(row)
    raw = data.get("arguments")
    data["arguments"] = json.loads(raw) if raw else None
    return StepRecord.model_validate(data)


def plan_steps_from(records: Iterable[StepRecord]) -> list[PlanStep]:
    """Project recorded steps onto the plan representation used by the loop."""
    return [
        PlanStep(id=r.step_index, task=r.action, status=r.status, note=r.note)
        for r in records
    ]


__all__ = [
    "SCHEMA_VERSION",
    "MemoryStore",
    "StepRecord",
    "plan_steps_from",
]
