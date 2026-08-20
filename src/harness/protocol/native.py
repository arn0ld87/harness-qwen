"""Codec for the runtime's native (``--jinja``) tool calling.

The chat template renders the tool schemas itself and the runtime constrains the
call with a lazy grammar, so this codec contributes no prompt text and reads the
already-parsed ``tool_calls`` off the response. Everything it can get wrong is
therefore a shape problem, not a text-parsing problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.core import Action, AnswerAction, ModelResponse, ParseError, ToolAction, ToolSpec
from harness.protocol.codec import PromptFragment, strip_reasoning_markup


class NativeToolCallCodec:
    """Maps ``ModelResponse.tool_calls`` onto the action vocabulary."""

    def __init__(self, tools: Sequence[ToolSpec] = ()) -> None:
        self.tools: tuple[ToolSpec, ...] = tuple(tools)

    def render_tools(self, tools: Sequence[ToolSpec] | None = None) -> PromptFragment:
        """Empty: the chat template already presents the tools.

        Repeating them in the prefix would pay for the same tokens twice and let
        the two presentations drift apart.
        """
        return ""

    def request_kwargs(self, tools: Sequence[ToolSpec] | None = None) -> dict[str, Any]:
        specs = self.tools if tools is None else tuple(tools)
        if not specs:
            # An empty ``tools`` array is not the same as no tool calling; some
            # templates emit a tool preamble for it regardless.
            return {}
        return {"tools": [spec.to_openai_tool() for spec in specs]}

    def parse(self, response: ModelResponse) -> Action | ParseError:
        content = strip_reasoning_markup(response.content)

        if response.tool_calls:
            # Roles are sequential and one step executes one action, so extra
            # parallel calls are dropped rather than queued behind an unobserved
            # result.
            call = response.tool_calls[0]
            name = call.name.strip()
            if not name:
                return ParseError(
                    raw=response.content,
                    reason=(
                        f"tool call {call.id!r} carried an empty tool name; "
                        "name the tool you want to run"
                    ),
                )
            return ToolAction(tool=name, arguments=call.arguments, reason=content)

        if content:
            return AnswerAction(content=content)

        if response.finish_reason == "length":
            return ParseError(
                raw=response.content,
                reason=(
                    "the response was cut off at the token limit before it produced "
                    "a tool call or an answer; be shorter"
                ),
            )
        return ParseError(
            raw=response.content,
            reason=(
                "the response contained neither a tool call nor any content; "
                "either call a tool or give the final answer"
            ),
        )
