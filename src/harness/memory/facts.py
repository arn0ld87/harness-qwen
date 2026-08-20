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


class FactMatch(BaseModel):
    """A fact together with why it ranked where it did.

    ``score`` is comparable only inside one result set: FTS5 yields a
    negated bm25 value, the fallback a term-occurrence count. ``strategy``
    is carried alongside so a caller can tell the two apart instead of
    reading a number that silently changed meaning.
    """

    fact: Fact
    score: float
    strategy: str


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


def _fact_from(row: sqlite3.Row) -> Fact:
    """Project a row onto :class:`Fact`, ignoring any ranking column joined in."""
    return Fact(
        key=row["key"],
        value=row["value"],
        category=row["category"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _like_score(row: sqlite3.Row, terms: list[str]) -> float:
    """Rank a fallback hit by how often the terms actually occur in it.

    Substring occurrences rather than token hits, because that is what the
    LIKE filter matched — a score computed on a different notion of "match"
    than the filter would reorder rows for reasons the filter never saw. The
    key weighs double: a term in the key names the fact, in the value it only
    appears in it.
    """
    key = row["key"].lower()
    value = row["value"].lower()
    category = row["category"].lower()
    return float(
        sum(
            2 * key.count(term.lower()) + value.count(term.lower())
            + category.count(term.lower())
            for term in terms
        )
    )


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
        return [
            match.fact
            for match in self.search_ranked(query, category=category, limit=limit)
        ]

    def search_ranked(
        self, query: str, *, category: str | None = None, limit: int = 20
    ) -> list[FactMatch]:
        """:meth:`search`, keeping the ranking signal instead of discarding it.

        The one place that decides how facts are searched, so retrieval
        (``harness.retrieval``) inherits this strategy rather than growing a
        second one that could drift from it.
        """
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
    ) -> list[FactMatch]:
        match = " ".join(f'"{t}"' for t in terms)
        sql = (
            "SELECT f.*, bm25(facts_fts) AS rank_score FROM facts_fts "
            "JOIN facts f ON f.id = facts_fts.rowid WHERE facts_fts MATCH ?"
        )
        params: list[Any] = [match]
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        # Ties break on key, never on rowid: two equally good facts must come
        # back in the same order on every call, or retrieval stops being
        # reproducible in a test.
        sql += " ORDER BY rank_score, f.key LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            FactMatch(
                fact=_fact_from(row),
                # bm25 counts down towards better; negate so every strategy
                # agrees that a higher score is a better match.
                score=-float(row["rank_score"]),
                strategy="fts5",
            )
            for row in rows
        ]

    def _search_like(
        self, terms: list[str], category: str | None, limit: int
    ) -> list[FactMatch]:
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
        # No LIMIT in SQL: scoring happens in Python, so cutting the rows
        # first would discard the best match whenever it is not also the most
        # recent one. Facts are written by hand and never harvested, so the
        # candidate set stays small enough to sort here.
        sql += " ORDER BY updated_at DESC, key"
        rows = self._conn.execute(sql, params).fetchall()
        scored = [
            FactMatch(fact=_fact_from(row), score=_like_score(row, terms), strategy="like")
            for row in rows
        ]
        # Stable sort, so the recency order above survives as the tie-break.
        scored.sort(key=lambda match: -match.score)
        return scored[:limit]
