"""JSON Schema and validator for the action union.

The schema names the tools that actually exist. A generic
``"tool": {"type": "string"}`` would let constrained sampling produce a
syntactically perfect call to a tool nobody registered — a whole class of
failure the grammar can prevent for free.

The same tool list drives a Pydantic validator, so a response that arrived
without the grammar applied is rejected by the identical rule set.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

from pydantic import Field, TypeAdapter, ValidationError, create_model

from harness.core import Action, AnswerAction, ToolAction, ToolSpec

ACTION_KINDS: tuple[str, ...] = ("tool", "answer")

_REASON_DESCRIPTION = "Why this tool call is the next step."
_CONTENT_DESCRIPTION = "The final answer to the task."
_EVIDENCE_DESCRIPTION = "References a verifier can check: paths, commands, exit codes."

# Model class names leak into Pydantic error locations through the discriminated
# union; they are noise to the model reading the feedback.
_INTERNAL_LOCS = frozenset({"RegisteredToolAction", "ToolAction", "AnswerAction"})

_JSON_TYPE_NAMES = {
    dict: "object", list: "array", str: "string", bool: "boolean",
    int: "number", float: "number", type(None): "null",
}


def json_type_name(value: object) -> str:
    """Name a decoded value the way the model's own output format names it."""
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _arguments_schema(tool: ToolSpec) -> dict[str, Any]:
    """The tool's own parameter schema, deep-copied.

    The result is handed to the runtime and may be rewritten there; sharing the
    dict would mutate the registry's ToolSpec as a side effect.
    """
    params = tool.parameters
    if not isinstance(params, dict) or params.get("type") != "object":
        return {"type": "object"}
    return copy.deepcopy(params)


def _tool_branch(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"const": "tool"},
            "tool": {"const": tool.name},
            "arguments": _arguments_schema(tool),
            "reason": {"type": "string", "description": _REASON_DESCRIPTION},
        },
        "required": ["action", "tool", "arguments"],
        "additionalProperties": False,
    }


def _answer_branch() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"const": "answer"},
            "content": {"type": "string", "description": _CONTENT_DESCRIPTION},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": _EVIDENCE_DESCRIPTION,
            },
        },
        # evidence stays optional here: an answer without it is the Verifier's
        # problem to downgrade, not a sampling error to refuse mid-generation.
        "required": ["action", "content"],
        "additionalProperties": False,
    }


def build_action_schema(tools: Sequence[ToolSpec]) -> dict[str, Any]:
    """JSON Schema for the action union, one branch per registered tool.

    With no tools the schema admits answers only — which is correct: a tool
    action that cannot name a tool is not representable.
    """
    branches = [_tool_branch(tool) for tool in tools]
    branches.append(_answer_branch())
    return {"title": "AgentAction", "oneOf": branches}


def _build_adapter(tool_names: tuple[str, ...]) -> TypeAdapter[Action]:
    tool_model: type[ToolAction] = ToolAction
    if tool_names:
        tool_model = create_model(
            "RegisteredToolAction",
            __base__=ToolAction,
            tool=(Literal[tool_names], ...),  # type: ignore[valid-type]
        )
    union = Annotated[tool_model | AnswerAction, Field(discriminator="action")]
    return cast(TypeAdapter[Action], TypeAdapter(union))


class ActionValidator:
    """Validates a decoded JSON payload against the registered tools.

    Returns a reason string instead of raising, because every caller has to put
    that reason back into the prompt anyway.
    """

    def __init__(self, tools: Sequence[ToolSpec] = ()) -> None:
        self.tool_names: tuple[str, ...] = tuple(dict.fromkeys(tool.name for tool in tools))
        self._adapter = _build_adapter(self.tool_names)

    def validate(self, payload: object) -> Action | str:
        if not isinstance(payload, dict):
            return (
                f"the action must be a JSON object, not a {json_type_name(payload)}"
            )
        # Copied because _precheck normalises recoverable argument shapes in
        # place, and the caller's decoded JSON is not ours to rewrite.
        candidate = dict(payload)
        pre = self._precheck(candidate)
        if pre is not None:
            return pre
        try:
            return self._adapter.validate_python(candidate)
        except ValidationError as exc:
            return f"the action object did not validate: {describe_validation_error(exc)}"

    def _precheck(self, payload: dict[str, Any]) -> str | None:
        """Catch the frequent shape errors before Pydantic phrases them badly."""
        if "action" not in payload:
            keys = ", ".join(repr(k) for k in list(payload)[:8]) or "none"
            return (
                'the object has no "action" field; it must be "tool" or "answer" '
                f"(keys found: {keys})"
            )
        kind = payload["action"]
        if kind not in ACTION_KINDS:
            return f'"action" was {kind!r}; the only actions are "tool" and "answer"'

        if kind == "answer":
            if "content" not in payload:
                return 'an answer action needs a "content" field holding the final answer'
            return None

        if "tool" not in payload:
            return f'a tool action needs a "tool" field naming one of: {self._available()}'
        name = payload["tool"]
        if not isinstance(name, str) or not name.strip():
            return f'"tool" must be a tool name; available tools: {self._available()}'
        if self.tool_names and name not in self.tool_names:
            return f"unknown tool {name!r}; available tools: {self._available()}"

        arguments = payload.get("arguments")
        if arguments is None:
            payload["arguments"] = {}
            return None
        if isinstance(arguments, str):
            # Some templates stringify the argument object; recover it rather
            # than burning a retry on a formatting detail.
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                return '"arguments" was a string that is not valid JSON; send a JSON object'
            if not isinstance(decoded, dict):
                return '"arguments" must be a JSON object mapping parameter names to values'
            payload["arguments"] = decoded
            return None
        if not isinstance(arguments, dict):
            return (
                f'"arguments" must be a JSON object, not a {json_type_name(arguments)}'
            )
        return None

    def _available(self) -> str:
        return ", ".join(self.tool_names) if self.tool_names else "none registered"


def describe_validation_error(exc: ValidationError) -> str:
    """Flatten a Pydantic error into one line the model can act on."""
    parts: list[str] = []
    for error in exc.errors():
        loc = [str(item) for item in error["loc"] if str(item) not in _INTERNAL_LOCS]
        field = ".".join(loc)
        parts.append(f"{field}: {error['msg']}" if field else error["msg"])
    return "; ".join(parts)
