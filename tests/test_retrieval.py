"""Retrieval over persistent facts, on both the FTS5 and the fallback path.

Every ranking test runs twice: once against the FTS5 index and once against
the LIKE scan, because the fallback is not a degraded mode nobody exercises —
it is what a stock SQLite build without FTS5 gets, and it has to answer the
same questions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.context import PromptAssembler, TokenBudget
from harness.context.compressor import CompressionState, RetrieveAgain, RetrieveFn
from harness.memory import MemoryStore
from harness.memory.facts import fts5_available
from harness.retrieval import (
    DEFAULT_LIMIT,
    RetrievalCapabilityError,
    RetrievalHit,
    Retriever,
    SqliteFtsRetriever,
    as_retrieve_fn,
    render_hits,
)

FTS5 = "fts5"
LIKE = "like"
STRATEGIES = [
    pytest.param(
        FTS5,
        marks=pytest.mark.skipif(
            not fts5_available(), reason="this SQLite build has no FTS5"
        ),
    ),
    pytest.param(LIKE),
]


@pytest.fixture
def strategy(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _store(tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    """Open a store pinned to one search strategy.

    ``fts5_available`` is patched rather than the store's attribute so the
    whole FactStore setup runs the way it would on a build without FTS5 —
    including dropping the triggers that would otherwise write into an index
    this configuration must not use.
    """
    if strategy == LIKE:
        monkeypatch.setattr("harness.memory.store.fts5_available", lambda: False)
    return MemoryStore(tmp_path / "memory.sqlite")


def _seed(store: MemoryStore) -> None:
    store.put_fact("cache-notes", "cache cache cache warm", "context")
    store.put_fact("prefix-notes", "the prefix is sacred, cache once", "context")
    store.put_fact("sandbox-notes", "bubblewrap confines the workspace", "security")


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_retriever_satisfies_the_interface(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        retriever = SqliteFtsRetriever(store)

        assert isinstance(retriever, Retriever)
        assert retriever.capability().strategy == strategy


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_denser_match_outranks_the_passing_mention(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        hits = SqliteFtsRetriever(store).retrieve("cache")

        assert [hit.id for hit in hits] == ["cache-notes", "prefix-notes"]
        assert hits[0].score > hits[1].score
        assert [hit.rank for hit in hits] == [1, 2]


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_hit_carries_source_id_and_ranking_metadata(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        hit = SqliteFtsRetriever(store).retrieve("bubblewrap")[0]

        assert isinstance(hit, RetrievalHit)
        assert hit.source == "facts"
        assert hit.id == "sandbox-notes"
        assert hit.text == "bubblewrap confines the workspace"
        assert hit.strategy == strategy
        assert hit.metadata["category"] == "security"
        assert hit.metadata["updated_at"]


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_ties_break_on_key_so_repeated_queries_agree(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        store.put_fact("b-note", "shared term", "context")
        store.put_fact("a-note", "shared term", "context")
        retriever = SqliteFtsRetriever(store)

        first = retriever.retrieve("shared")
        second = retriever.retrieve("shared")

        assert [hit.id for hit in first] == ["a-note", "b-note"]
        assert first == second


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
@pytest.mark.parametrize("query", ["", "   ", "!!!", "%_"])
def test_query_without_usable_terms_returns_nothing(
    tmp_path: Path, strategy: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        assert SqliteFtsRetriever(store).retrieve(query) == []


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
@pytest.mark.parametrize("query", ['"cache', "cache*", "(cache)", "cache%", "cache!!"])
def test_punctuation_around_a_term_is_dropped(
    tmp_path: Path, strategy: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither FTS5 syntax nor LIKE wildcards may reach the engine as syntax."""
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        hits = SqliteFtsRetriever(store).retrieve(query)

        assert [hit.id for hit in hits][:1] == ["cache-notes"]


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_operator_words_are_searched_for_not_obeyed(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NEAR(x)`` raw would be an FTS5 syntax error, and ``OR`` would widen
    the query. Both come back as ordinary terms that simply do not occur, so
    the caller gets an empty result instead of an exception."""
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        assert SqliteFtsRetriever(store).retrieve("cache OR NEAR(x)") == []


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_limit_caps_the_result_set(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)
        retriever = SqliteFtsRetriever(store)

        assert len(retriever.retrieve("cache", limit=1)) == 1
        assert [h.id for h in retriever.retrieve("cache", limit=1)] == ["cache-notes"]


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
@pytest.mark.parametrize("limit", [0, -1])
def test_limit_below_one_is_rejected(
    tmp_path: Path, strategy: str, limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent empty result would look exactly like "nothing matched"."""
    with (
        _store(tmp_path, strategy, monkeypatch) as store,
        pytest.raises(ValueError, match="limit"),
    ):
        SqliteFtsRetriever(store).retrieve("cache", limit=limit)


@pytest.mark.parametrize("strategy", STRATEGIES, indirect=True)
def test_category_scopes_the_search(
    tmp_path: Path, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, strategy, monkeypatch) as store:
        _seed(store)

        scoped = SqliteFtsRetriever(store, category="security")

        assert [hit.id for hit in scoped.retrieve("workspace")] == ["sandbox-notes"]
        assert scoped.retrieve("cache") == []


def test_missing_fts5_still_answers_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, LIKE, monkeypatch) as store:
        _seed(store)
        capability = SqliteFtsRetriever(store).capability()

        assert capability.strategy == LIKE
        assert capability.ranked is False
        assert "FTS5" in capability.detail
        assert SqliteFtsRetriever(store).retrieve("cache")


def test_missing_fts5_raises_when_the_caller_requires_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        _store(tmp_path, LIKE, monkeypatch) as store,
        pytest.raises(RetrievalCapabilityError, match="FTS5"),
    ):
        SqliteFtsRetriever(store, require_fts5=True)


@pytest.mark.skipif(not fts5_available(), reason="this SQLite build has no FTS5")
def test_present_fts5_satisfies_the_requirement(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        capability = SqliteFtsRetriever(store, require_fts5=True).capability()

        assert capability.strategy == FTS5
        assert capability.ranked is True


def test_a_path_is_not_a_store(tmp_path: Path) -> None:
    """The retriever owns no path handling, so it can open no foreign database."""
    with pytest.raises(TypeError, match="MemoryStore"):
        SqliteFtsRetriever(str(tmp_path / "somewhere.sqlite"))  # type: ignore[arg-type]


def test_read_only_store_is_searchable(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    with MemoryStore(database) as writable:
        _seed(writable)

    with MemoryStore(database, read_only=True) as reader:
        hits = SqliteFtsRetriever(reader).retrieve("cache")

        assert [hit.id for hit in hits][:1] == ["cache-notes"]


def test_render_hits_is_empty_for_no_results() -> None:
    assert render_hits([]) == ""


def test_render_hits_labels_every_hit_with_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _store(tmp_path, LIKE, monkeypatch) as store:
        _seed(store)

        rendered = render_hits(SqliteFtsRetriever(store).retrieve("cache"))

        assert "facts:cache-notes" in rendered
        assert "cache cache cache warm" in rendered
        assert "facts:prefix-notes" in rendered


def test_retrieve_fn_appends_to_the_append_zone_only(tmp_path: Path) -> None:
    """The prefix is cached; a retrieval that touched it would cost ~25 s."""
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        _seed(store)
        assembler = PromptAssembler(system="system", task="task")
        prefix_before = assembler.prefix_text()
        retrieve: RetrieveFn = as_retrieve_fn(SqliteFtsRetriever(store))

        RetrieveAgain().apply(
            CompressionState(
                assembler=assembler,
                budget=TokenBudget(),
                retrieve=retrieve,
                retrieval_query="cache",
            )
        )

        assert assembler.prefix_text() == prefix_before
        assert "cache cache cache warm" in assembler.append_messages[-1].content


def test_retrieve_fn_returns_empty_text_when_nothing_matches(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        _seed(store)

        assert as_retrieve_fn(SqliteFtsRetriever(store))("nonexistentterm") == ""


def test_retrieve_fn_honours_its_limit(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        _seed(store)

        rendered = as_retrieve_fn(SqliteFtsRetriever(store), limit=1)("cache")

        assert "facts:cache-notes" in rendered
        assert "facts:prefix-notes" not in rendered
        assert DEFAULT_LIMIT > 1
