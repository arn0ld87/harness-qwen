"""Persistent memory: explicitly written facts, searched by keyword.

Facts are architecture notes, conventions, decisions and known issues. They
are written on purpose and never harvested from transcripts, so the store
stays small enough that FTS5 ranking and a LIKE scan return the same answers —
which is what makes the fallback acceptable instead of a silent downgrade.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from harness.memory.migrations import object_exists

# FTS5 lives here and not in schema.sql because SQL has no conditional DDL and
# this index is optional by design.
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    key, value, category, content='facts', content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, key, value, category)
    VALUES (new.id, new.key, new.value, new.category);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
    VALUES ('delete', old.id, old.key, old.value, old.category);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
    VALUES ('delete', old.id, old.key, old.value, old.category);
    INSERT INTO facts_fts(rowid, key, value, category)
    VALUES (new.id, new.key, new.value, new.category);
END;
"""
FTS_TRIGGERS = ("facts_fts_ai", "facts_fts_ad", "facts_fts_au")


class Fact(BaseModel):
    """One entry of persistent memory."""

    key: str
    value: str
    category: str
    created_at: datetime
    updated_at: datetime


def fts5_available() -> bool:
    """Whether this SQLite build can create FTS5 tables.

    Probed against a throwaway in-memory database because the compile-options
    pragma does not reliably list statically linked extensions.
    """
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        return True
    except sqlite3.Error:
        return False
    finally:
        probe.close()


def like_escape(term: str) -> str:
    for char in ("\\", "%", "_"):
        term = term.replace(char, "\\" + char)
    return term


def split_terms(query: str) -> list[str]:
    """Split a user query into plain search terms.

    Punctuation is dropped rather than escaped, so an FTS5 operator typed by
    accident cannot become a syntax error the caller has to handle.
    """
    out: list[str] = []
    current: list[str] = []
    for char in query:
        if char.isalnum() or char in "_-":
            current.append(char)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


class FactStore:
    """Fact operations over an already-open, already-migrated connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        *,
        fts5: bool,
        read_only: bool = False,
    ) -> None:
        self._conn = conn
        self._lock = lock
        if read_only:
            # Building or repairing the index is a write. A reader takes the
            # index as it finds it and falls back to the LIKE scan, which
            # returns the same answers on a store this small.
            self.fts5 = fts5 and object_exists(conn, "table", "facts_fts")
            return
        self.fts5 = fts5
        self._ensure_fts()

    def _ensure_fts(self) -> None:
        conn = self._conn
        if not self.fts5:
            # A database created by an FTS5-capable build carries triggers that
            # write into a table this build cannot open. Dropping them keeps
            # fact writes working; the index is rebuilt on the next open that
            # does have FTS5.
            if object_exists(conn, "table", "facts_fts"):
                for trigger in FTS_TRIGGERS:
                    conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                conn.commit()
            return
        stale = not object_exists(conn, "table", "facts_fts") or any(
            not object_exists(conn, "trigger", t) for t in FTS_TRIGGERS
        )
        conn.executescript(FTS_SQL)
        if stale:
            conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
        conn.commit()

    def put(self, key: str, value: str, category: str = "general") -> None:
        """Insert or replace a fact. ``created_at`` survives an update."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO facts (key, value, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
                """,
                (key, value, category, now, now),
            )

    def get(self, key: str) -> Fact | None:
        row = self._conn.execute("SELECT * FROM facts WHERE key = ?", (key,)).fetchone()
        return Fact.model_validate(dict(row)) if row else None

    def list(self, category: str | None = None) -> list[Fact]:
        if category is None:
            rows = self._conn.execute("SELECT * FROM facts ORDER BY key").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE category = ? ORDER BY key", (category,)
            ).fetchall()
        return [Fact.model_validate(dict(r)) for r in rows]

    def delete(self, key: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM facts WHERE key = ?", (key,)
            ).rowcount > 0

    def search(
        self, query: str, *, category: str | None = None, limit: int = 20
    ) -> list[Fact]:
        """Find facts by keyword, ranked when FTS5 is available."""
        terms = split_terms(query)
        if not terms:
            return []
        if self.fts5:
            try:
                return self._search_fts(terms, category, limit)
            except sqlite3.OperationalError:
                # A build that advertises FTS5 but fails on this query is not
                # worth diagnosing at runtime; the LIKE scan still answers.
                pass
        return self._search_like(terms, category, limit)

    def _search_fts(
        self, terms: list[str], category: str | None, limit: int
    ) -> list[Fact]:
        match = " ".join(f'"{t}"' for t in terms)
        sql = (
            "SELECT f.* FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
            "WHERE facts_fts MATCH ?"
        )
        params: list[Any] = [match]
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " ORDER BY bm25(facts_fts) LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [Fact.model_validate(dict(r)) for r in rows]

    def _search_like(
        self, terms: list[str], category: str | None, limit: int
    ) -> list[Fact]:
        clauses: list[str] = []
        params: list[Any] = []
        for term in terms:
            pattern = f"%{like_escape(term)}%"
            clauses.append(
                r"(key LIKE ? ESCAPE '\' OR value LIKE ? ESCAPE '\' "
                r"OR category LIKE ? ESCAPE '\')"
            )
            params.extend([pattern, pattern, pattern])
        sql = f"SELECT * FROM facts WHERE {' AND '.join(clauses)}"
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY updated_at DESC, key LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [Fact.model_validate(dict(r)) for r in rows]
