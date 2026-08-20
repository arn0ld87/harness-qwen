"""Codec for grammar-constrained JSON completion.

Unlike :class:`~harness.protocol.native.NativeToolCallCodec`, nothing here can
rely on the chat template to present the tools or on the runtime to hand back
a structured ``tool_calls`` list: the model emits plain text and the runtime's
job is only to keep that text inside the JSON Schema from
:func:`request_kwargs`. Grammar-constrained sampling makes the *shape*
reliable — braces balance, required keys exist, enums hold — but it cannot
stop a 3B-active model from wrapping the object in prose, fencing it, or using
the wrong quote character before it settles into the grammar. ``parse`` is
where that residue gets cleaned up, and it must always resolve to either an
``Action`` or a specific, actionable :class:`~harness.core.ParseError` — never
an exception, because the retry policy has nothing to put back into the
prompt from a stack trace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from harness.core import Action, ModelResponse, ParseError, ToolSpec
from harness.protocol.codec import PromptFragment, strip_reasoning_markup
from harness.protocol.schema import ActionValidator, build_action_schema, json_type_name

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(text: str) -> str:
    """Drop a comma that precedes a closing brace or bracket.

    Safe as an unconditional pass: valid JSON never has a comma immediately
    before ``}``/``]``, so this only ever touches the malformed case.
    """
    return _TRAILING_COMMA.sub(r"\1", text)


def _quotes_to_double(text: str) -> str:
    """Best-effort repair for Python-style single-quoted strings.

    Only fires when the candidate has no double quote anywhere — the moment a
    real ``"`` is present, a blind ``'`` -> ``"`` swap risks mangling an
    apostrophe sitting inside a legitimately double-quoted string (``"it's
    fine"``), which is a worse outcome than leaving a single-quoted candidate
    to fail parsing with a clear error. When there is no double quote to
    protect, the whole object is quoted consistently with ``'`` and the swap
    is unambiguous.
    """
    if '"' in text:
        return text
    return text.replace("'", '"')


def _scan_balanced(text: str, start: int) -> int | None:
    """Index just past the JSON value opened at ``start``, or ``None``.

    A minimal string-aware bracket counter: a brace or bracket character only
    changes the nesting depth when it sits outside a string literal, so
    ``{"a": "}"}`` closes on the real ``}`` and not the one quoted inside the
    value. Depth is tracked with one counter across both bracket types
    because the scanner's only job is finding a plausible span to hand to the
    real parser — ``json.loads`` is what actually rejects a mismatched
    ``{...]``.
    """
    depth = 0
    in_string = False
    quote = ""
    escape = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
        else:
            if ch in "\"'":
                in_string = True
                quote = ch
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def _candidate_spans(text: str) -> list[str]:
    """Every top-level balanced ``{...}``/``[...]`` substring, left to right.

    This is what makes fenced blocks need no special handling: the scanner
    only looks for brace/bracket characters, so ```` ```json ```` markers,
    leading prose ("Here's my answer:") and trailing commentary are simply
    skipped as ordinary text on the way to the first opening brace. A stray
    brace in prose that happens to balance (rare, but possible) still yields
    a candidate — later candidates are given a chance too, and the one that
    actually looks like an action wins.
    """
    spans: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            end = _scan_balanced(text, i)
            if end is not None:
                spans.append(text[i:end])
                i = end
                continue
        i += 1
    return spans


def _parse_json_lenient(span: str) -> tuple[Any, str | None]:
    """Parse one candidate span, retrying once with repairs on failure."""
    text = span.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        repaired = _strip_trailing_commas(_quotes_to_double(text))
        if repaired == text:
            return None, f"the JSON did not parse: {exc}"
        try:
            return json.loads(repaired), None
        except json.JSONDecodeError as exc2:
            return None, f"the JSON did not parse even after repair: {exc2}"


def _extract_payload(content: str) -> tuple[dict[str, Any] | None, str]:
    """Find the action object among the candidate spans in ``content``.

    Preference order: a dict that already has an ``"action"`` key wins
    outright (the strongest signal of being the real payload, even past a
    stray balanced brace in leading prose); otherwise the first dict-shaped
    candidate is handed to :class:`ActionValidator`, whose own precheck
    already phrases "no action field" precisely — duplicating that message
    here would only let the two drift apart. A one-element array wrapping an
    object is unwrapped before either check, since a small model reaching for
    ``[...]`` around a single action is a formatting tic, not a different
    answer.
    """
    spans = _candidate_spans(content)
    if not spans:
        return None, (
            "the response contained no JSON object; respond with exactly one "
            'JSON object, e.g. {"action": "answer", "content": "..."}'
        )

    best_error: str | None = None
    fallback: dict[str, Any] | None = None

    for span in spans:
        value, error = _parse_json_lenient(span)
        if error is not None:
            if best_error is None:
                best_error = error
            continue

        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], dict):
                value = value[0]
            else:
                if best_error is None:
                    best_error = (
                        f"the response was a JSON array of {len(value)} items; "
                        "send exactly one action object, not a list"
                    )
                continue

        if not isinstance(value, dict):
            if best_error is None:
                best_error = (
                    f"the action must be a JSON object, not a {json_type_name(value)}"
                )
            continue

        if "action" in value:
            return value, ""
        if fallback is None:
            fallback = value

    if fallback is not None:
        return fallback, ""
    return None, best_error or "the response did not contain a valid JSON action object"


def _param_summary(tool: ToolSpec) -> str:
    """Compact ``(name: type, name?: type)`` listing from the tool's schema.

    Deliberately not the full JSON Schema — that is what :func:`request_kwargs`
    sends the runtime for sampling. Repeating it in the prompt text would pay
    for the same information twice; the model only needs enough here to
    choose the right tool and shape plausible arguments.
    """
    params = tool.parameters
    if not isinstance(params, dict) or params.get("type") != "object":
        return "()"
    properties = params.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "()"
    required = set(params.get("required") or [])
    parts: list[str] = []
    for name, spec in properties.items():
        type_name = spec.get("type", "any") if isinstance(spec, dict) else "any"
        marker = "" if name in required else "?"
        parts.append(f"{name}{marker}: {type_name}")
    return "(" + ", ".join(parts) + ")"


class ConstrainedJsonCodec:
    """Renders a tool listing into the prompt, constrains sampling to the
    action schema, and repairs the model's JSON on the way back out."""

    def __init__(self, tools: Sequence[ToolSpec] = ()) -> None:
        self.tools: tuple[ToolSpec, ...] = tuple(tools)
        # Built once: the codec is bound to one tool set for its lifetime,
        # same as NativeToolCallCodec — a run that registers a new tool gets a
        # new codec, not a live-mutated one.
        self._validator = ActionValidator(self.tools)

    def render_tools(self, tools: Sequence[ToolSpec] | None = None) -> PromptFragment:
        """The action envelope plus a compact tool listing.

        Unlike the native codec this is never empty: the grammar constrains
        *shape*, not tool choice by name, so the model still has to be told
        which tools exist and what they take.
        """
        specs = self.tools if tools is None else tuple(tools)
        lines = [
            "Respond with exactly one JSON object and nothing else — no prose, "
            "no code fence, no explanation outside the object.",
            "",
            'To call a tool: {"action": "tool", "tool": "<name>", '
            '"arguments": {...}, "reason": "<why>"}',
            'To give the final answer: {"action": "answer", "content": "<answer>", '
            '"evidence": ["<ref>", ...]}',
        ]
        if specs:
            lines.append("")
            lines.append("Tools:")
            for tool in sorted(specs, key=lambda s: s.name):
                lines.append(f"- {tool.name}{_param_summary(tool)}: {tool.description}")
        return "\n".join(lines)

    def request_kwargs(self, tools: Sequence[ToolSpec] | None = None) -> dict[str, Any]:
        specs = self.tools if tools is None else tuple(tools)
        return {"json_schema": build_action_schema(specs)}

    def parse(self, response: ModelResponse) -> Action | ParseError:
        # response.tool_calls is not read here: this codec's contract is a
        # JSON object in content. Populated tool_calls would mean the request
        # was built for the wrong codec, not a case for this one to guess at.
        content = strip_reasoning_markup(response.content)

        if not content.strip():
            if response.finish_reason == "length":
                return ParseError(
                    raw=response.content,
                    reason=(
                        "the response was cut off at the token limit before "
                        "producing any content; be shorter and give the JSON "
                        "action first"
                    ),
                )
            return ParseError(
                raw=response.content,
                reason=(
                    "the response was empty; respond with one JSON object like "
                    '{"action": "answer", "content": "..."}'
                ),
            )

        payload, reason = _extract_payload(content)
        if payload is None:
            if response.finish_reason == "length":
                reason = (
                    f"{reason}; the response was also cut off at the token limit, "
                    "so a shorter answer is more likely to complete"
                )
            return ParseError(raw=response.content, reason=reason)

        result = self._validator.validate(payload)
        if isinstance(result, str):
            return ParseError(raw=response.content, reason=result)
        return result
