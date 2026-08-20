"""Tool registry: name to callable, with the guardrails around the call.

Every failure mode a tool can hit becomes a ``ToolResult`` with ``ok=False`` and
an ``error_kind``, never an exception escaping into the agent loop. A wrong tool
name in particular is a recoverable mistake — the model gets the list of tools
that do exist and can correct itself on the next step, which is far cheaper than
a retry against an unchanged prompt.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from harness.core import Risk, ToolResult, ToolSpec

ToolFn = Callable[..., ToolResult | Awaitable[ToolResult]]

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    spec: ToolSpec
    fn: ToolFn


class ToolRegistry:
    """Holds the tool set for a run and is the only way to invoke it."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec=spec, fn=fn)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names]

    def openai_tools(self) -> list[dict[str, Any]]:
        """Tool declarations for the stable prefix; order is deterministic.

        Sorting by name matters: the schema block sits in the cached prefix, and
        a reordered set of otherwise identical tools would invalidate it.
        """
        return [spec.to_openai_tool() for spec in self.specs()]

    async def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate, dispatch and time one tool call. Never raises."""
        started = time.perf_counter()
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(
                tool=name,
                ok=False,
                error_kind="not_found",
                content=(
                    f"Unknown tool {name!r}. Available tools: "
                    f"{', '.join(self.names) or '(none registered)'}"
                ),
                duration_ms=_elapsed_ms(started),
            )

        if entry.spec.risk is Risk.DENY:
            return ToolResult(
                tool=name,
                ok=False,
                error_kind="denied",
                content=f"Tool {name!r} is registered as denied and was not executed.",
                duration_ms=_elapsed_ms(started),
            )

        args = dict(arguments or {})
        errors = validate_arguments(entry.spec.parameters, args)
        if errors:
            return ToolResult(
                tool=name,
                ok=False,
                error_kind="invalid_arguments",
                content="Invalid arguments for {}: {}".format(name, "; ".join(errors)),
                duration_ms=_elapsed_ms(started),
            )

        try:
            result = await asyncio.wait_for(
                _call(entry.fn, args), timeout=entry.spec.timeout_s
            )
        except TimeoutError:
            return ToolResult(
                tool=name,
                ok=False,
                error_kind="timeout",
                content=f"Tool {name!r} exceeded its {entry.spec.timeout_s:g}s timeout.",
                duration_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            return ToolResult(
                tool=name,
                ok=False,
                error_kind=_error_kind(exc),
                content=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(started),
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                tool=name,
                ok=False,
                error_kind="execution_failed",
                content=f"Tool {name!r} returned {type(result).__name__}, expected ToolResult.",
                duration_ms=_elapsed_ms(started),
            )

        # The tool names itself for direct callers; the registry is the
        # authority once it is dispatched through here.
        return result.model_copy(
            update={
                "tool": name,
                "duration_ms": result.duration_ms or _elapsed_ms(started),
            }
        )


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Check ``arguments`` against the JSON Schema subset tool specs use.

    A dedicated validator dependency would cover far more of the specification
    than a tool schema ever uses; this covers object/property/required/enum and
    the scalar bounds, and reports every violation at once so one round trip is
    enough for the model to fix the call.
    """
    errors: list[str] = []
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])

    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument {key!r}")

    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                allowed = ", ".join(sorted(properties)) or "none"
                errors.append(
                    f"unexpected argument {key!r} (allowed: {allowed})"
                )

    for key, value in arguments.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            errors.extend(f"{key}: {problem}" for problem in _check_value(subschema, value))

    return errors


def _check_value(schema: dict[str, Any], value: Any) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    types = [expected] if isinstance(expected, str) else list(expected or [])
    if types and not any(_matches_type(kind, value) for kind in types):
        errors.append(f"expected {' or '.join(types)}, got {type(value).__name__}")
        return errors

    choices = schema.get("enum")
    if choices is not None and value not in choices:
        errors.append(f"must be one of {choices}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if low is not None and value < low:
            errors.append(f"must be >= {low}")
        if high is not None and value > high:
            errors.append(f"must be <= {high}")

    if isinstance(value, str):
        min_len, max_len = schema.get("minLength"), schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"must be at least {min_len} characters")
        if max_len is not None and len(value) > max_len:
            errors.append(f"must be at most {max_len} characters")

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(f"[{index}] {problem}" for problem in _check_value(item_schema, item))

    if isinstance(value, dict) and schema.get("type") == "object":
        errors.extend(validate_arguments(schema, value))

    return errors


def _matches_type(kind: str, value: Any) -> bool:
    allowed = _JSON_TYPES.get(kind)
    if allowed is None:
        return True
    # JSON has no bool/int union; Python does, and ``True`` is not an integer
    # argument by any reading the model intended.
    if kind in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


async def _call(fn: ToolFn, arguments: dict[str, Any]) -> Any:
    """Await async tools, run sync ones off the event loop."""
    if inspect.iscoroutinefunction(_unwrap(fn)):
        return await fn(**arguments)
    result = await asyncio.to_thread(fn, **arguments)
    if inspect.isawaitable(result):
        return await result
    return result


def _unwrap(fn: ToolFn) -> Any:
    """See through functools.partial so coroutine detection still works."""
    inner = fn
    while hasattr(inner, "func"):
        inner = inner.func
    return inner


def _error_kind(exc: BaseException) -> str:
    kind = getattr(exc, "kind", None)
    if isinstance(kind, str):
        return kind
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    return "execution_failed"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
