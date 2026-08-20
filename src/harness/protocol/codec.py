"""The action protocol interface shared by both codecs.

The agent loop never learns which encoding produced an action. That is the
point: the runtime supports native tool calls and schema-constrained JSON, and
only a benchmark can say which one this model obeys more reliably, so swapping
one for the other must cost nothing above this line.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from harness.core import Action, ModelResponse, ParseError, ToolSpec

type PromptFragment = str
"""Prompt text a codec contributes. It lands in the stable prefix, so identical
input must render byte-identical output."""

# Templates that do not split reasoning into ``ModelResponse.reasoning`` leave
# it inline in the content. The back-reference on the tag name keeps a <think>
# block from being closed by a stray </reasoning> further down.
_CLOSED_THINK = re.compile(r"<(think|thinking|reasoning)\s*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<(think|thinking|reasoning)\s*>.*\Z", re.DOTALL | re.IGNORECASE)


@runtime_checkable
class ActionCodec(Protocol):
    """Turns tool specs into request fields and a response into an Action."""

    def render_tools(self, tools: Sequence[ToolSpec]) -> PromptFragment:
        """Prompt fragment describing the tools, or "" when the template does it."""
        ...

    def request_kwargs(self, tools: Sequence[ToolSpec]) -> dict[str, Any]:
        """Fields merged into a :class:`~harness.core.GenerationRequest`."""
        ...

    def parse(self, response: ModelResponse) -> Action | ParseError:
        """Decode one response. Never raises: an unusable reply is a ParseError.

        The retry policy needs a value it can put back into the prompt, and an
        exception thrown from inside the loop is not that.
        """
        ...


def strip_reasoning_markup(text: str) -> str:
    """Drop inline thinking blocks from model content.

    Reasoning must never reach an AnswerAction or be mistaken for the action
    payload — a discarded plan often contains a JSON draft that the model then
    revised.
    """
    cleaned = _CLOSED_THINK.sub("", text)
    # An unterminated block means generation stopped inside the reasoning; the
    # remainder is not content.
    cleaned = _OPEN_THINK.sub("", cleaned)
    return cleaned.strip()
