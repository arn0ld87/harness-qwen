"""The retrieval tool: the model queries the fact store through the registry.

This is the tool counterpart to the compression ladder's ``RetrieveAgain``
rung (issue #18): same retriever, same ``render_hits`` labels, but model-driven
rather than triggered by an append-budget rung. Both only ever append — the
result lands in the append zone (~0.8 s) and the cached prefix (~25 s to
rebuild) never moves.

The function is a coroutine on purpose. SQLite has no async driver here, so an
async signature elsewhere only wraps a blocking call in ceremony; and running
this in ``asyncio.to_thread`` would put a SQLite connection the event-loop
thread already touches (the compression ladder calls the same retriever
inline) into a worker thread. A coroutine dispatches in the event loop
directly — the same thread the ladder uses — which ``check_same_thread = False``
plus the store's lock keep safe, and which keeps retrieval off the hot path
for tool dispatch without introducing a second thread.
"""

from __future__ import annotations

from harness.core import ToolResult
from harness.retrieval.base import DEFAULT_LIMIT, Retriever, render_hits

EMPTY_RESULT = "No matching facts found."


async def retrieve_facts(
    retriever: Retriever, *, query: str, limit: int = DEFAULT_LIMIT
) -> ToolResult:
    """Run one retrieval and render the hits for the append zone.

    ``limit`` defaults to ``DEFAULT_LIMIT`` because the schema marks it
    optional: a model that omits it gets the small, generation-cheap bound
    rather than the whole corpus. ``render_hits`` keeps the ``source:id``
    labels so a claim the model makes can cite where a fact came from instead
    of presenting retrieved text as its own knowledge. No match is an honest
    empty answer, not an error.
    """
    hits = retriever.retrieve(query, limit=limit)
    return ToolResult(
        tool="retrieve_facts",
        ok=True,
        content=render_hits(hits) or EMPTY_RESULT,
    )


__all__ = ["retrieve_facts"]