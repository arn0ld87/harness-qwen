"""Retrieval over the persistent facts already in the memory store.

No new table and no new index: ``facts`` plus the optional ``facts_fts``
mirror in ``harness.memory.facts`` is the documented FTS5 index, and a
second copy of the same content would have to be kept in sync with it — a
retriever that can answer from stale data is worse than one that cannot
answer at all. That also keeps the schema version where it is.

The retriever takes an open :class:`~harness.memory.store.MemoryStore`, never
a path. It therefore has no way to open a database the caller did not already
open, which is how "no unchecked absolute paths, no foreign workspace data"
is enforced here — by owning no path handling rather than by validating paths
correctly.
"""

from __future__ import annotations

from harness.memory.facts import FactMatch
from harness.memory.store import MemoryStore
from harness.retrieval.base import (
    DEFAULT_LIMIT,
    RetrievalCapability,
    RetrievalCapabilityError,
    RetrievalHit,
    check_limit,
)

SOURCE = "facts"


class SqliteFtsRetriever:
    """Rank persistent facts by keyword, via FTS5 where the build has it.

    Falls back to the store's LIKE scan otherwise. The fallback is a real
    answer, not a stub: facts are written by hand and never harvested, so the
    corpus stays small enough that a scan returns what bm25 would have. Pass
    ``require_fts5`` when that is not good enough for the caller.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        require_fts5: bool = False,
        category: str | None = None,
    ) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError(
                "SqliteFtsRetriever needs an open MemoryStore, not "
                f"{type(store).__name__} — it opens no databases of its own"
            )
        self._store = store
        self._category = category
        if require_fts5 and not store.fts5:
            raise RetrievalCapabilityError(
                "FTS5 is unavailable in this SQLite build (or absent from this "
                "database), and ranked retrieval was required"
            )

    def capability(self) -> RetrievalCapability:
        fts5 = self._store.fts5
        return RetrievalCapability(
            backend="sqlite-facts",
            strategy="fts5" if fts5 else "like",
            ranked=fts5,
            detail=""
            if fts5
            else "FTS5 unavailable; falling back to a scored LIKE scan",
        )

    def retrieve(self, query: str, *, limit: int = DEFAULT_LIMIT) -> list[RetrievalHit]:
        check_limit(limit)
        matches = self._store.facts.search_ranked(
            query, category=self._category, limit=limit
        )
        return [_to_hit(match, rank) for rank, match in enumerate(matches, start=1)]


def _to_hit(match: FactMatch, rank: int) -> RetrievalHit:
    return RetrievalHit(
        source=SOURCE,
        id=match.fact.key,
        text=match.fact.value,
        score=match.score,
        rank=rank,
        strategy=match.strategy,
        metadata={
            "category": match.fact.category,
            "updated_at": match.fact.updated_at.isoformat(),
        },
    )
