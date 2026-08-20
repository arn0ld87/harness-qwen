"""Read-only projection of the memory database for ``memory inspect`` (#15).

Two decisions shape this module. The store is opened read-only, so looking at
a database cannot migrate, repair or otherwise rewrite it — the command runs
against a file another process may be writing, and a diagnostic that changes
its subject is worse than no diagnostic.

And a row that no longer parses is reported as a finding for that run rather
than raised: a store nobody can read is precisely when someone reaches for
this command, so one broken resume point must not take the report down with
it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.core import StepStatus
from harness.memory.migrations import StoreError, UnknownRunError
from harness.memory.store import MemoryStore
from harness.telemetry.redact import redact_data

DEFAULT_RUN_LIMIT = 20


def open_for_inspection(path: str | Path) -> MemoryStore:
    """Open ``path`` read-only, or explain why it cannot be read.

    A missing file is a message, not an empty report: writing one would create
    the database the caller was asking about.
    """
    database = Path(path)
    if not database.exists():
        raise StoreError(
            f"no memory database at {database} — no run has written one yet"
        )
    try:
        return MemoryStore(database, read_only=True)
    except sqlite3.DatabaseError as exc:
        raise StoreError(
            f"{database} is not a readable SQLite database: {exc}"
        ) from exc


def inspect_memory(
    store: MemoryStore,
    *,
    run_id: str | None = None,
    status: StepStatus | None = None,
    limit: int = DEFAULT_RUN_LIMIT,
    include_facts: bool = True,
    fact_query: str | None = None,
) -> dict[str, Any]:
    """Everything the store knows, filtered, redacted and JSON-ready.

    One payload feeds both the table and ``--json``, so the automation and the
    human are never looking at different subsets of the same database.
    """
    if run_id is not None:
        row = store.get_run(run_id)
        if row is None:
            raise UnknownRunError(f"no run {run_id!r} in {store.path}")
        rows = [row]
    else:
        rows = store.list_runs(limit=limit)

    runs = []
    for row in rows:
        entry = _inspect_run(store, row, status=status)
        # A status filter is a question about steps; a run with no answer to
        # it is noise rather than a result.
        if status is not None and not entry["steps"]:
            continue
        runs.append(entry)

    payload: dict[str, Any] = {
        "database": str(store.path),
        "schema_version": store.schema_version,
        "runs": runs,
    }
    if include_facts:
        payload["facts"] = [
            fact.model_dump(mode="json")
            for fact in (
                store.search_facts(fact_query) if fact_query else store.get_facts()
            )
        ]
    # Step arguments are whatever a tool was called with, so a run that
    # curled an authenticated endpoint stored the header. The rule that keeps
    # secrets out of the journal holds for the command that prints it back.
    return redact_data(payload)


def damaged_runs(payload: dict[str, Any]) -> list[str]:
    """Run ids whose state or history could not be read."""
    return [entry["run"]["run_id"] for entry in payload["runs"] if entry["errors"]]


def _inspect_run(
    store: MemoryStore, row: dict[str, Any], *, status: StepStatus | None
) -> dict[str, Any]:
    run_id = str(row["run_id"])
    errors: list[str] = []
    task = _guard(errors, "task_state", lambda: store.load_task_state(run_id))
    runtime = _guard(errors, "runtime_state", lambda: store.load_runtime_state(run_id))
    steps = _guard(errors, "steps", lambda: store.get_steps(run_id)) or []

    counts: dict[str, int] = {}
    for step in steps:
        counts[str(step.status)] = counts.get(str(step.status), 0) + 1

    return {
        "run": dict(row),
        "task_state": task.model_dump(mode="json") if task is not None else None,
        "runtime_state": (
            runtime.model_dump(mode="json") if runtime is not None else None
        ),
        # Counted over the whole history, listed after the filter: the shape of
        # a run does not change because someone asked about its failures.
        "step_status_counts": counts,
        "unfinished": sum(1 for step in steps if step.finished_at is None),
        "steps": [
            step.model_dump(mode="json")
            for step in steps
            if status is None or step.status is status
        ],
        "errors": errors,
    }


def _guard[T](errors: list[str], label: str, read: Callable[[], T]) -> T | None:
    """Perform one read, recording rather than raising what it cannot parse.

    ``ValueError`` covers both ways a stored blob goes bad: pydantic's
    ``ValidationError`` when the JSON no longer matches the model, and
    ``JSONDecodeError`` when it is not JSON at all.
    """
    try:
        return read()
    except (StoreError, sqlite3.DatabaseError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
        return None


__all__ = [
    "DEFAULT_RUN_LIMIT",
    "damaged_runs",
    "inspect_memory",
    "open_for_inspection",
]
