"""Schema version detection and the upgrade path between versions.

Separated from the store because this is the only code allowed to decide that
a database on disk may be used. A schema written by an older harness is
upgraded; one written by a newer harness is refused. Silently reading either
would corrupt the resume point, which is the one thing this subsystem exists
to protect.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Upgrades from the keyed version to the next one. Empty while SCHEMA_VERSION
# is 1; the mechanism ships now so that version 2 becomes a data migration
# rather than a data loss.
def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE runtime_state (
            run_id       TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            runtime_json TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
        """
    )


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_1_to_2}


class StoreError(RuntimeError):
    """The store could not satisfy a request."""


class SchemaVersionError(StoreError):
    """The database on disk was written by a different schema version."""


class UnknownRunError(StoreError):
    """An operation referenced a run_id that has no row in ``runs``."""


def check_schema(conn: sqlite3.Connection, label: str = "database") -> int:
    """Assert the database already is at :data:`SCHEMA_VERSION` and return it.

    The read-only counterpart of :func:`ensure_schema`, for callers that only
    want to look: reading a database is not consent to rewrite it, and an
    inspection that silently migrated would perform the one irreversible act
    its user was trying to avoid.

    Raises:
        SchemaVersionError: The database is at another version, or carries no
            harness schema at all.
    """
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version == SCHEMA_VERSION:
        return version
    if version == 0 and not object_exists(conn, "table", "runs"):
        raise SchemaVersionError(
            f"{label}: no harness schema in this database — it is empty or "
            "belongs to something else"
        )
    raise SchemaVersionError(
        f"{label}: schema version {version} is not the supported "
        f"{SCHEMA_VERSION}; opening it read-only will not migrate it"
    )


def object_exists(conn: sqlite3.Connection, kind: str, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", (kind, name)
    ).fetchone()
    return row is not None


def configure(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL keeps a reader (a status command) from blocking the run that is
    # writing its step history. It is a no-op for :memory: databases.
    conn.execute("PRAGMA journal_mode = WAL")
    # NORMAL loses at most the last transactions on a power cut, never on a
    # process kill — and a killed process is the failure that resume exists
    # for. FULL would cost an fsync per step to defend against a threat this
    # harness does not have.
    conn.execute("PRAGMA synchronous = NORMAL")


def ensure_schema(conn: sqlite3.Connection, label: str = "database") -> int:
    """Bring ``conn`` to :data:`SCHEMA_VERSION` and return the version reached.

    Raises:
        SchemaVersionError: The database is newer than this harness, carries
            tables but no recorded version, or sits at a version with no
            registered upgrade.
    """
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])

    if version == 0:
        if object_exists(conn, "table", "runs"):
            raise SchemaVersionError(
                f"{label}: tables present but no schema version recorded; this "
                "database predates schema versioning and must be migrated or "
                "removed by hand"
            )
        script = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(f"{script}\nPRAGMA user_version = {SCHEMA_VERSION};")
        conn.commit()
        return SCHEMA_VERSION

    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{label}: schema version {version} is newer than the supported "
            f"{SCHEMA_VERSION}; refusing to read it"
        )

    while version < SCHEMA_VERSION:
        upgrade = MIGRATIONS.get(version)
        if upgrade is None:
            raise SchemaVersionError(
                f"{label}: no migration from schema version {version} to "
                f"{version + 1}"
            )
        upgrade(conn)
        version += 1
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    return version
