"""What a retriever is, and what a caller gets back from one.

The interface is deliberately small: one query, one limit, a list of hits.
Anything richer (filters expressed as a query language, hybrid scoring,
embeddings) would have to be paid for by a benchmark first — CONTEXT.md
section 8 rules out a vector database without demonstrated need, and an
interface shaped for one invites it in through the back door.

It is synchronous although the rest of the harness is asyncio. The consumer
is ``harness.context.compressor.RetrieveFn`` (``query -> str``), which the
compression ladder calls inline while pricing a rung; and SQLite has no async
driver here, so an async signature would only wrap a blocking call in
ceremony.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

DEFAULT_LIMIT = 5
"""Hits per retrieval. Small on purpose: the result lands in the append zone
and every token in it is generation time this machine does not get back
(CONTEXT.md section 1)."""


class RetrievalError(RuntimeError):
    """A retrieval could not be served."""


class RetrievalCapabilityError(RetrievalError):
    """The backend cannot offer a capability the caller demanded.

    Raised at construction, not at query time: a caller that requires ranked
    retrieval wants to find out before it has built a run around it.
    """


class RetrievalCapability(BaseModel):
    """What a backend can actually do, as opposed to what it is named after.

    ``SqliteFtsRetriever`` keeps its name on a build without FTS5 because it
    still answers; ``ranked`` is what says whether those answers came from a
    ranking function or from a scan with a counted-occurrence tie-break.
    """

    backend: str
    strategy: str
    ranked: bool
    detail: str = ""


class RetrievalHit(BaseModel):
    """One retrieved item, carrying where it came from and why it ranked here.

    ``source`` plus ``id`` is the addressable identity of the item, so a hit
    can be traced back to the row that produced it rather than arriving as
    anonymous text. ``score`` is comparable only within one result set — see
    ``strategy``.
    """

    source: str
    id: str
    text: str
    score: float
    rank: int
    strategy: str
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Query, limit, structured hits — the whole contract."""

    def capability(self) -> RetrievalCapability:
        """What this backend can do right now, on this machine."""
        ...

    def retrieve(self, query: str, *, limit: int = DEFAULT_LIMIT) -> list[RetrievalHit]:
        """Best ``limit`` hits for ``query``, best first.

        A query with no usable terms returns no hits rather than raising: an
        empty question honestly has no answer. A ``limit`` below 1 does raise,
        because an empty list would be indistinguishable from "nothing
        matched".
        """
        ...


def check_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    return limit


def render_hits(hits: Sequence[RetrievalHit]) -> str:
    """Render hits for the append zone: one labelled block per hit.

    Terse by design — this text is appended to a prompt whose every token is
    paid for at generation time. The label keeps the provenance the hit was
    carrying, so the model can cite where a claim came from instead of
    treating retrieved text as its own knowledge.
    """
    return "\n\n".join(
        f"{hit.source}:{hit.id} (rank {hit.rank}, score {hit.score:.3f})\n{hit.text}"
        for hit in hits
    )


def as_retrieve_fn(
    retriever: Retriever, *, limit: int = DEFAULT_LIMIT
) -> Callable[[str], str]:
    """Adapt a retriever to the ``RetrieveFn`` the compression ladder injects.

    Returns rendered text rather than hits because that is the shape the
    ladder appends, and appending is the only thing retrieval is allowed to
    do to a prompt: the result belongs at the tail (~0.8 s) and never in the
    cached prefix (~25 s to rebuild).

    No match renders to an empty string, which the ladder reads as "this rung
    changed nothing" and skips.
    """
    check_limit(limit)

    def retrieve(query: str) -> str:
        return render_hits(retriever.retrieve(query, limit=limit))

    return retrieve
