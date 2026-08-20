"""What a benchmark asks the model, and how the answer is judged.

A case is data, kept in a JSON file rather than in code, because the question
a benchmark asks changes far more often than the machinery that asks it — and
because a run artefact is only comparable to an earlier one if the question
behind both can be read back verbatim.

The loader refuses rather than defaults. An unknown key is a misspelling that
would otherwise be dropped silently, and a run configured by a dropped key
measures a different question than the one written down.
"""

from __future__ import annotations

import enum
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.core import GenerationRequest, Message, ModelResponse

SUITE_SCHEMA_VERSION = 1


class _Strict(BaseModel):
    """Reject unknown keys, for the reason ``config`` does: a typo that is
    ignored is indistinguishable from a setting that took effect."""

    model_config = ConfigDict(extra="forbid")


class OutputMode(enum.StrEnum):
    """Which output path the case exercises.

    ``JSON_SCHEMA`` and ``TOOLS`` are the two independent hard sampling
    constraints the runtime offers (DISCOVERY.md section 6). Neither injects
    anything into the prompt, so the two can be compared on the same prompt —
    which is the comparison the capability suite exists to make.
    """

    TEXT = "text"
    JSON_SCHEMA = "json_schema"
    TOOLS = "tools"


class ExpectationKind(enum.StrEnum):
    """How a response is judged. ``NONE`` judges nothing."""

    NONE = "none"
    JSON_OBJECT = "json_object"
    TOOL_CALL = "tool_call"
    CONTAINS = "contains"
    REGEX = "regex"


_NEEDS_VALUE = (ExpectationKind.CONTAINS, ExpectationKind.REGEX)


class Expectation(_Strict):
    """A deterministic verdict on one response.

    Deliberately not a model-graded rubric. A benchmark whose scoring depends
    on the same class of system it is measuring cannot separate a capability
    change from a grader change, and this suite is meant to be re-runnable
    against a different build months later with the same meaning.
    """

    kind: ExpectationKind = ExpectationKind.NONE
    value: str | None = None
    """Substring, regex or tool name, depending on ``kind``."""
    required_keys: list[str] = Field(default_factory=list)
    """For ``JSON_OBJECT``: keys the parsed object must carry."""

    @model_validator(mode="after")
    def _value_present_where_it_is_needed(self) -> Expectation:
        if self.kind in _NEEDS_VALUE and not self.value:
            raise ValueError(f"expectation kind {self.kind!r} needs a 'value' to compare against")
        return self

    def check(self, response: ModelResponse) -> tuple[bool, str | None]:
        """Return ``(met, reason_if_not)``.

        The reason is recorded on the sample rather than raised: a case that
        the model failed is a result, not an error in the benchmark.
        """
        if self.kind is ExpectationKind.NONE:
            return True, None
        if self.kind is ExpectationKind.JSON_OBJECT:
            return self._check_json(response.content)
        if self.kind is ExpectationKind.TOOL_CALL:
            return self._check_tool_call(response)
        if self.kind is ExpectationKind.CONTAINS:
            found = (self.value or "").lower() in response.content.lower()
            return found, None if found else f"content does not contain {self.value!r}"
        found = re.search(self.value or "", response.content) is not None
        return found, None if found else f"content does not match {self.value!r}"

    def _check_json(self, content: str) -> tuple[bool, str | None]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            return False, f"content is not valid JSON: {exc.msg}"
        if not isinstance(parsed, dict):
            return False, f"content is JSON but not an object ({type(parsed).__name__})"
        missing = [key for key in self.required_keys if key not in parsed]
        if missing:
            return False, f"object is missing required keys: {', '.join(missing)}"
        return True, None

    def _check_tool_call(self, response: ModelResponse) -> tuple[bool, str | None]:
        if not response.tool_calls:
            return False, "no tool call was emitted"
        if self.value is None:
            return True, None
        names = [call.name for call in response.tool_calls]
        if self.value in names:
            return True, None
        return False, f"expected a call to {self.value!r}, got {', '.join(names) or 'none'}"


class BenchmarkCase(_Strict):
    """One question, asked ``repetitions`` times after ``warmup`` throwaways."""

    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    system: str | None = None
    prompt: str

    output_mode: OutputMode = OutputMode.TEXT
    json_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None

    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    """Set it to make a case reproducible; leave it unset to measure the
    variance the model actually shows in use."""
    stop: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=600.0, gt=0)

    repetitions: int = Field(default=5, ge=1)
    warmup: int = Field(default=1, ge=0)
    """Calls made and recorded before measurement starts. One is the minimum
    that matters here: the first call after a prefix change reprocesses the
    whole prompt (24.83 s for 4816 tokens), every later one does not."""

    expect: Expectation = Field(default_factory=Expectation)

    @model_validator(mode="after")
    def _constraint_matches_the_mode(self) -> BenchmarkCase:
        """A mode without its constraint would silently become a text case.

        That is the worst possible failure for this suite: the run completes,
        the numbers look normal, and the question asked was not the one in the
        file.
        """
        if self.output_mode is OutputMode.JSON_SCHEMA and self.json_schema is None:
            raise ValueError(f"case {self.id!r}: output_mode 'json_schema' needs a 'json_schema'")
        if self.output_mode is OutputMode.TOOLS and not self.tools:
            raise ValueError(f"case {self.id!r}: output_mode 'tools' needs 'tools' definitions")
        return self

    def to_request(self) -> GenerationRequest:
        """Build the model call. The constraint rides on the request.

        Neither the schema nor the tool definitions are written into the
        prompt text: llama-server applies them as sampling constraints, and
        pasting them into the prompt would change both the measurement and
        the cached prefix.
        """
        messages: list[Message] = []
        if self.system is not None:
            messages.append(Message(role="system", content=self.system))
        messages.append(Message(role="user", content=self.prompt))

        return GenerationRequest(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=list(self.stop),
            timeout_s=self.timeout_s,
            json_schema=self.json_schema if self.output_mode is OutputMode.JSON_SCHEMA else None,
            tools=self.tools if self.output_mode is OutputMode.TOOLS else None,
        )


class CaseDefaults(_Strict):
    """Suite-wide values a case may omit.

    Only the knobs a whole suite tends to share. Prompts and expectations are
    deliberately absent: a default question would make a case readable only
    together with the file header.
    """

    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    timeout_s: float | None = Field(default=None, gt=0)
    repetitions: int | None = Field(default=None, ge=1)
    warmup: int | None = Field(default=None, ge=0)
    system: str | None = None

    def apply(self, case: dict[str, Any]) -> dict[str, Any]:
        """Fill in only what the case did not state itself."""
        merged = dict(case)
        for field, value in self.model_dump(exclude_none=True).items():
            merged.setdefault(field, value)
        return merged


class BenchmarkSuite(_Strict):
    """A named set of cases, as read from a suite file."""

    schema_version: int = SUITE_SCHEMA_VERSION
    name: str
    description: str = ""
    defaults: CaseDefaults = Field(default_factory=CaseDefaults)
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _case_ids_are_unique(self) -> BenchmarkSuite:
        """Two cases with one id produce two results nobody can tell apart."""
        seen = [case.id for case in self.cases]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids in suite {self.name!r}: {', '.join(duplicates)}")
        return self

    def select(self, ids: list[str]) -> BenchmarkSuite:
        """Narrow the suite, refusing an id it does not contain.

        Refusing rather than returning fewer cases: a mistyped ``--case``
        would otherwise produce an empty, successful run.
        """
        known = {case.id: case for case in self.cases}
        missing = [name for name in ids if name not in known]
        if missing:
            raise KeyError(f"unknown case ids: {', '.join(missing)}")
        return self.model_copy(update={"cases": [known[name] for name in ids]})


def load_suite(path: str | Path) -> BenchmarkSuite:
    """Read and validate a suite file, applying its defaults to each case."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"benchmark suite not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a benchmark suite must be a JSON object")

    defaults = CaseDefaults.model_validate(raw.get("defaults", {}))
    payload = dict(raw)
    payload["cases"] = [defaults.apply(case) if isinstance(case, dict) else case
                        for case in raw.get("cases", [])]
    return BenchmarkSuite.model_validate(payload)


__all__ = [
    "SUITE_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkSuite",
    "CaseDefaults",
    "Expectation",
    "ExpectationKind",
    "OutputMode",
    "load_suite",
]
