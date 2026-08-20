"""Benchmark case definitions and the shipped capability suite (issue #19).

A suite file is data, and data that is wrong in a way nobody notices produces
a run whose numbers describe a different question than the one asked. So the
loader rejects rather than defaults: an unknown key, a missing schema for a
schema-constrained case, a repetition count of zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.benchmark import (
    BenchmarkCase,
    BenchmarkSuite,
    Expectation,
    ExpectationKind,
    OutputMode,
    load_suite,
)
from harness.core import ModelResponse, ToolCall

CAPABILITY_SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "model-capabilities.json"


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal(**case: object) -> dict:
    return {
        "schema_version": 1,
        "name": "minimal",
        "cases": [{"id": "one", "prompt": "hi", **case}],
    }


# -- the suite shipped with the repository ---------------------------------


def test_the_capability_suite_ships_and_loads() -> None:
    suite = load_suite(CAPABILITY_SUITE)
    assert suite.cases
    assert len({case.id for case in suite.cases}) == len(suite.cases)


def test_the_capability_suite_covers_both_structured_output_paths() -> None:
    # DISCOVERY.md section 6: json_schema and native tool calls are two
    # independent hard sampling constraints. Which one the model complies
    # with more reliably is the question this suite exists to answer.
    modes = {case.output_mode for case in load_suite(CAPABILITY_SUITE).cases}
    assert OutputMode.JSON_SCHEMA in modes
    assert OutputMode.TOOLS in modes


def test_every_shipped_case_states_a_repetition_count_above_one() -> None:
    # A single sample has no median and no percentile; it is an anecdote.
    assert all(case.repetitions > 1 for case in load_suite(CAPABILITY_SUITE).cases)


# -- loading and validation ------------------------------------------------


def test_suite_defaults_fill_in_cases_that_do_not_state_a_value(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "schema_version": 1,
        "name": "defaults",
        "defaults": {"repetitions": 7, "warmup": 2, "max_tokens": 64},
        "cases": [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y", "repetitions": 3}],
    })
    suite = load_suite(path)
    assert (suite.cases[0].repetitions, suite.cases[0].warmup) == (7, 2)
    assert suite.cases[0].max_tokens == 64
    # An explicit value on the case wins over the suite default.
    assert suite.cases[1].repetitions == 3


def test_a_misspelled_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, _minimal(repititions=5))
    with pytest.raises(ValidationError):
        load_suite(path)


def test_duplicate_case_ids_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "schema_version": 1,
        "name": "dupes",
        "cases": [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}],
    })
    with pytest.raises(ValidationError):
        load_suite(path)


def test_a_schema_constrained_case_without_a_schema_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, _minimal(output_mode="json_schema"))
    with pytest.raises(ValidationError):
        load_suite(path)


def test_a_tool_case_without_tool_definitions_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, _minimal(output_mode="tools"))
    with pytest.raises(ValidationError):
        load_suite(path)


def test_a_missing_suite_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_suite(tmp_path / "absent.json")
    assert "absent.json" in str(excinfo.value)


def test_zero_repetitions_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, _minimal(repetitions=0))
    with pytest.raises(ValidationError):
        load_suite(path)


def test_a_case_may_run_without_warmup_but_says_so(tmp_path: Path) -> None:
    suite = load_suite(_write(tmp_path, _minimal(warmup=0)))
    assert suite.cases[0].warmup == 0


# -- expectations ----------------------------------------------------------


def test_json_object_expectation_accepts_a_parsable_object() -> None:
    expectation = Expectation(kind=ExpectationKind.JSON_OBJECT, required_keys=["answer"])
    ok, detail = expectation.check(ModelResponse(content='{"answer": 4}'))
    assert ok and detail is None


def test_json_object_expectation_reports_the_missing_key() -> None:
    expectation = Expectation(kind=ExpectationKind.JSON_OBJECT, required_keys=["answer"])
    ok, detail = expectation.check(ModelResponse(content='{"other": 4}'))
    assert not ok
    assert "answer" in (detail or "")


def test_json_object_expectation_rejects_prose() -> None:
    expectation = Expectation(kind=ExpectationKind.JSON_OBJECT)
    ok, _ = expectation.check(ModelResponse(content="The answer is four."))
    assert not ok


def test_tool_call_expectation_checks_the_tool_name() -> None:
    expectation = Expectation(kind=ExpectationKind.TOOL_CALL, value="read_file")
    response = ModelResponse(
        content="", tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "x"})]
    )
    assert expectation.check(response)[0]
    other = ModelResponse(
        content="", tool_calls=[ToolCall(id="1", name="write_file", arguments={})]
    )
    assert not expectation.check(other)[0]


def test_tool_call_expectation_fails_when_the_model_answered_in_prose() -> None:
    expectation = Expectation(kind=ExpectationKind.TOOL_CALL)
    assert not expectation.check(ModelResponse(content="I would read the file."))[0]


def test_contains_expectation_is_case_insensitive() -> None:
    expectation = Expectation(kind=ExpectationKind.CONTAINS, value="Berlin")
    assert expectation.check(ModelResponse(content="the capital is berlin"))[0]


def test_regex_expectation_matches_anywhere_in_the_content() -> None:
    expectation = Expectation(kind=ExpectationKind.REGEX, value=r"\b42\b")
    assert expectation.check(ModelResponse(content="it is 42, obviously"))[0]
    assert not expectation.check(ModelResponse(content="it is 43"))[0]


def test_the_default_expectation_judges_nothing() -> None:
    # A latency case is not a capability case; forcing a verdict on it would
    # invent a pass rate for a question nobody asked.
    ok, detail = Expectation().check(ModelResponse(content=""))
    assert ok and detail is None


def test_an_expectation_that_needs_a_value_refuses_to_be_built_without_one() -> None:
    with pytest.raises(ValidationError):
        Expectation(kind=ExpectationKind.CONTAINS)


# -- request construction --------------------------------------------------


def test_a_case_renders_its_system_prompt_ahead_of_the_user_turn() -> None:
    case = BenchmarkCase(id="a", prompt="question", system="rules")
    request = case.to_request()
    assert [m.role for m in request.messages] == ["system", "user"]
    assert request.messages[0].content == "rules"


def test_a_schema_case_puts_the_schema_on_the_request_not_in_the_prompt() -> None:
    # The schema constrains sampling; injecting it into the prompt would both
    # change the measurement and move the cached prefix.
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    case = BenchmarkCase(
        id="a", prompt="q", output_mode=OutputMode.JSON_SCHEMA, json_schema=schema
    )
    request = case.to_request()
    assert request.json_schema == schema
    assert "properties" not in request.messages[-1].content


def test_a_suite_can_be_filtered_to_named_cases() -> None:
    suite = BenchmarkSuite(
        name="s",
        cases=[BenchmarkCase(id="a", prompt="x"), BenchmarkCase(id="b", prompt="y")],
    )
    assert [c.id for c in suite.select(["b"]).cases] == ["b"]


def test_filtering_to_an_unknown_case_is_an_error_not_an_empty_run() -> None:
    suite = BenchmarkSuite(name="s", cases=[BenchmarkCase(id="a", prompt="x")])
    with pytest.raises(KeyError):
        suite.select(["nope"])
