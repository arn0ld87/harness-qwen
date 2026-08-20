"""Retrieval as an agent tool (issue #18).

The model queries the persistent fact store through the ``ToolRegistry``;
results land in the append zone carrying their ``source:id`` labels, and the
cached prefix never moves. These are the acceptance criteria from the issue,
each pinned by a test that fails without the piece of wiring it exercises.

The compression-ladder invariants the issue also names — no retrieval during
``ContextOverflow`` recovery, stale retrieval outputs marked on repeat — are
already covered by ``test_context_compressor.py`` (``RetrieveAgain`` is skipped
at overflow, ``DropSupersededToolOutputs`` supersedes earlier tool outputs by
name). Activating the rung in ``build_loop`` makes those tests speak for the
real integration rather than for an isolated ladder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.core import SideEffect, StopReason
from harness.memory import MemoryStore
from harness.models import FakeProvider
from harness.retrieval import SqliteFtsRetriever
from harness.session import build_loop, build_registry
from harness.tools.builtin import RETRIEVE_FACTS


def _seed(store: MemoryStore) -> None:
    store.put_fact("cache-notes", "cache cache cache warm", "context")
    store.put_fact("prefix-notes", "the prefix is sacred, cache once", "context")
    store.put_fact("sandbox-notes", "bubblewrap confines the workspace", "security")


RETRIEVE_CALL = (
    '{"action":"tool","tool":"retrieve_facts",'
    '"arguments":{"query":"cache"},"reason":"look up cache notes"}'
)


def _answer(content: str) -> str:
    return json.dumps(
        {"action": "answer", "content": content, "evidence": []}
    )


# --- registration ----------------------------------------------------------


def test_retrieve_facts_registered_when_a_retriever_is_given(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite") as store:
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        assert "retrieve_facts" in registry.names


def test_retrieve_facts_absent_without_a_retriever(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)

    assert "retrieve_facts" not in registry.names


def test_retrieve_facts_stays_available_in_a_read_only_run(tmp_path: Path) -> None:
    """Retrieval is read-only, so a look-only run keeps it while losing the
    mutating tools. A run that can only look is exactly the run that benefits
    from querying memory."""
    with MemoryStore(tmp_path / "m.sqlite") as store:
        registry = build_registry(
            tmp_path, read_only=True, retriever=SqliteFtsRetriever(store)
        )

        assert "retrieve_facts" in registry.names
        assert "write_file" not in registry.names
        assert "run_command" not in registry.names


def test_retrieve_facts_spec_is_read_only_and_repeatable() -> None:
    """SideEffect.NONE: an interrupted call may simply run again on resume,
    and the no-progress / supersede logic treats it as a safe repeat."""
    assert RETRIEVE_FACTS.side_effect is SideEffect.NONE
    assert RETRIEVE_FACTS.risk.name == "ALLOW"


# --- tool result shape -----------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_facts_returns_hits_labelled_with_source_and_id(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "m.sqlite") as store:
        _seed(store)
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        result = await registry.invoke("retrieve_facts", {"query": "cache"})

        assert result.ok
        assert result.error_kind is None
        # source:id labels — the addressable identity the issue requires for
        # audit and telemetry, preserved verbatim from render_hits.
        assert "facts:cache-notes" in result.content
        assert "facts:prefix-notes" in result.content
        assert "cache cache cache warm" in result.content


@pytest.mark.asyncio
async def test_retrieve_facts_honours_the_limit(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite") as store:
        _seed(store)
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        result = await registry.invoke(
            "retrieve_facts", {"query": "cache", "limit": 1}
        )

        assert result.ok
        assert "facts:cache-notes" in result.content
        assert "facts:prefix-notes" not in result.content


@pytest.mark.asyncio
async def test_retrieve_facts_reports_no_match_honestly(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite") as store:
        _seed(store)
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        result = await registry.invoke(
            "retrieve_facts", {"query": "nonexistentterm"}
        )

        assert result.ok
        assert "No matching facts found." in result.content


@pytest.mark.asyncio
async def test_retrieve_facts_requires_a_query(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite") as store:
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        result = await registry.invoke("retrieve_facts", {"limit": 5})

        assert not result.ok
        assert result.error_kind == "invalid_arguments"


@pytest.mark.asyncio
async def test_retrieve_facts_rejects_a_limit_below_one(tmp_path: Path) -> None:
    """The schema bound (minimum: 1) is enforced before the retriever sees the
    call, so a ValueError never escapes as an execution_failed result."""
    with MemoryStore(tmp_path / "m.sqlite") as store:
        registry = build_registry(tmp_path, retriever=SqliteFtsRetriever(store))

        result = await registry.invoke(
            "retrieve_facts", {"query": "cache", "limit": 0}
        )

        assert not result.ok
        assert result.error_kind == "invalid_arguments"


# --- end to end: retrieve -> tool result -> answer -------------------------


@pytest.mark.asyncio
async def test_e2e_retrieve_then_answer(tmp_path: Path) -> None:
    """The path the issue is about: the model calls the retrieval tool, its
    result reaches the append zone with source labels, and the next step
    answers from what it retrieved."""
    from harness.config import HarnessConfig

    config = HarnessConfig.model_validate(
        {"workspace": str(tmp_path), "database": str(tmp_path / "m.sqlite")}
    )
    store = MemoryStore(tmp_path / "m.sqlite")
    _seed(store)
    provider = FakeProvider([RETRIEVE_CALL, _answer("the cache is warm")])
    loop = build_loop(
        config, goal="explain the cache", provider=provider, memory=store
    )
    try:
        result = await loop.run()

        assert result.stop_reason is StopReason.ANSWERED
        assert result.tool_calls == 1
        retrieved = [
            m
            for m in loop.assembler.append_messages
            if "facts:cache-notes" in m.content
        ]
        assert retrieved, "retrieval result must reach the append zone"
    finally:
        loop.memory.close()
        loop.journal.close()


# --- prefix stability across retrieval calls --------------------------------


@pytest.mark.asyncio
async def test_prefix_hash_is_stable_across_retrieval_calls(tmp_path: Path) -> None:
    """The retrieval tool is registered once at setup, so it sits in the cached
    prefix from the first call. Subsequent retrieval calls append results but
    never touch the prefix — the hash the FakeProvider records on every call
    must not move, which is the invariant the whole cache design rests on."""
    from harness.config import HarnessConfig

    config = HarnessConfig.model_validate(
        {"workspace": str(tmp_path), "database": str(tmp_path / "m.sqlite")}
    )
    store = MemoryStore(tmp_path / "m.sqlite")
    _seed(store)
    provider = FakeProvider([RETRIEVE_CALL, RETRIEVE_CALL, _answer("done")])
    loop = build_loop(
        config, goal="explain the cache", provider=provider, memory=store
    )
    try:
        await loop.run()

        assert len(provider.prefix_hashes) >= 3
        assert provider.prefix_stable
    finally:
        loop.memory.close()
        loop.journal.close()