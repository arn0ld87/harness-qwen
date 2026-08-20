from harness.core import CommandEvidence, ModelResponse, ToolCall, ToolSpec
from harness.protocol.constrained import ConstrainedJsonCodec
from harness.protocol.native import NativeToolCallCodec
from harness.protocol.schema import build_action_schema


def _tool() -> ToolSpec:
    return ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def test_constrained_codec_repairs_fenced_single_quoted_action() -> None:
    codec = ConstrainedJsonCodec([_tool()])

    action = codec.parse(
        ModelResponse(
            content=(
                "```json\n{'action':'tool','tool':'read_file',"
                "'arguments':{'path':'README.md'},}\n```"
            )
        )
    )

    assert action.action == "tool"
    assert action.arguments == {"path": "README.md"}


def test_native_codec_uses_first_tool_call() -> None:
    action = NativeToolCallCodec([_tool()]).parse(
        ModelResponse(
            content="inspect",
            tool_calls=[
                ToolCall(id="1", name="read_file", arguments={"path": "README.md"}),
                ToolCall(id="2", name="read_file", arguments={"path": "API.md"}),
            ],
        )
    )

    assert action.action == "tool"
    assert action.arguments == {"path": "README.md"}


def test_answer_schema_requires_typed_evidence_objects() -> None:
    schema = build_action_schema([_tool()])
    answer_branch = schema["oneOf"][-1]
    evidence_items = answer_branch["properties"]["evidence"]["items"]

    assert "oneOf" in evidence_items
    assert CommandEvidence(step_id=1, kind="test").model_dump() == {
        "kind": "test",
        "step_id": 1,
    }
