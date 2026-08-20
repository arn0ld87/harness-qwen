"""Retrieval: a small interface over the facts already in the memory store."""

from harness.retrieval.base import (
    DEFAULT_LIMIT,
    RetrievalCapability,
    RetrievalCapabilityError,
    RetrievalError,
    RetrievalHit,
    Retriever,
    as_retrieve_fn,
    check_limit,
    render_hits,
)
from harness.retrieval.sqlite import SOURCE, SqliteFtsRetriever

__all__ = [
    "DEFAULT_LIMIT",
    "SOURCE",
    "RetrievalCapability",
    "RetrievalCapabilityError",
    "RetrievalError",
    "RetrievalHit",
    "Retriever",
    "SqliteFtsRetriever",
    "as_retrieve_fn",
    "check_limit",
    "render_hits",
]
