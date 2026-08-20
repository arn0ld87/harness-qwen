from pathlib import Path

import pytest

from harness.agent.verifier import ExecutedStep, Verifier
from harness.core import AnswerAction, CommandEvidence, ToolResult


def _command_step(
    command: str,
    *,
    exit_code: int,
    output: str = "",
    step_id: int = 17,
    run_id: str = "run-1",
    tool: str = "run_command",
) -> ExecutedStep:
    return ExecutedStep(
        id=step_id,
        run_id=run_id,
        step_index=1,
        tool=tool,
        arguments={"command": command},
        result=ToolResult(
            tool="run_command",
            ok=exit_code == 0,
            exit_code=exit_code,
            content=output,
        ),
    )


@pytest.mark.parametrize(
    ("step", "verified"),
    [
        (_command_step("echo pytest", exit_code=0), False),
        (_command_step("pytest", exit_code=1), False),
        (_command_step("pytest", exit_code=0), True),
        (_command_step("echo unrelated", exit_code=0, output="pytest"), False),
    ],
)
def test_test_claim_requires_successful_test_command(
    tmp_path: Path,
    step: ExecutedStep,
    verified: bool,
) -> None:
    outcome = Verifier().verify(
        AnswerAction(
            content="All tests pass",
            evidence=[CommandEvidence(step_id=17, kind="test")],
        ),
        history=[step],
        run_id="run-1",
        workspace=tmp_path,
    )

    assert outcome.verified is verified


@pytest.mark.parametrize(
    "step",
    [
        _command_step("pytest", exit_code=0, step_id=18),
        _command_step("pytest", exit_code=0, run_id="other-run"),
        _command_step("pytest", exit_code=0, tool="read_file"),
    ],
)
def test_test_claim_rejects_evidence_from_wrong_step_run_or_tool(
    tmp_path: Path,
    step: ExecutedStep,
) -> None:
    outcome = Verifier().verify(
        AnswerAction(
            content="All tests pass",
            evidence=[CommandEvidence(step_id=17, kind="test")],
        ),
        history=[step],
        run_id="run-1",
        workspace=tmp_path,
    )

    assert outcome.verified is False


@pytest.mark.parametrize(
    ("content", "command", "kind"),
    [
        ("Lint passes", "uv run ruff check .", "lint"),
        ("Typecheck passes", "uv run mypy src", "typecheck"),
        ("Build succeeds", "uv build", "build"),
    ],
)
def test_command_claim_requires_matching_command_kind(
    tmp_path: Path,
    content: str,
    command: str,
    kind: str,
) -> None:
    outcome = Verifier().verify(
        AnswerAction(
            content=content,
            evidence=[CommandEvidence(step_id=17, kind=kind)],
        ),
        history=[_command_step(command, exit_code=0)],
        run_id="run-1",
        workspace=tmp_path,
    )

    assert outcome.verified is True
