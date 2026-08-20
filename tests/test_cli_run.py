"""`harness run` end to end (issue #13).

Every case uses `FakeProvider` with scripted responses: the loop, the protocol
and the tool layer are all exercised, the 35B model is not. What is being
tested here is the process contract — what a caller sees and what the exit
code commits to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from harness.cli import app
from harness.cli_run import (
    EXIT_BUDGET,
    EXIT_CANCELLED,
    EXIT_CONFIG,
    EXIT_NO_PROGRESS,
    EXIT_OK,
    EXIT_UNVERIFIED,
    exit_code_for,
)
from harness.core import RunResult, StopReason

runner = CliRunner()

LIST_CALL = (
    '{"action":"tool","tool":"list_files",'
    '"arguments":{"path":"."},"reason":"look around"}'
)


def _answer(content: str, evidence: str = "") -> str:
    return json.dumps(
        {
            "action": "answer",
            "content": content,
            "evidence": json.loads(evidence) if evidence else [],
        }
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path


def _invoke(workspace: Path, *args: str, script: list[str]) -> Result:
    """Run the CLI with a scripted provider instead of a real one."""
    import harness.session as session
    from harness.models import FakeProvider

    original = session.build_provider
    session.build_provider = lambda _config: FakeProvider(list(script))  # type: ignore[assignment]
    try:
        return runner.invoke(
            app,
            ["run", *args],
            env={
                "HARNESS_WORKSPACE": str(workspace),
                "HARNESS_DATABASE": str(workspace / "memory.sqlite"),
            },
        )
    finally:
        session.build_provider = original  # type: ignore[assignment]


# -- the exit code contract ------------------------------------------------


def test_an_unverified_answer_is_not_success() -> None:
    """The harness could not confirm the claim, so the caller must not
    treat it as confirmed. That assurance is the point of the project."""
    result = RunResult(run_id="r", stop_reason=StopReason.ANSWERED, verified=False)
    assert exit_code_for(result) == EXIT_UNVERIFIED


def test_each_stop_reason_has_its_own_code() -> None:
    """Different responses: a bigger budget, a different task, a human look.

    One generic failure code would hide all three.
    """
    codes = {
        exit_code_for(RunResult(run_id="r", stop_reason=reason))
        for reason in (
            StopReason.BUDGET_EXHAUSTED,
            StopReason.NO_PROGRESS,
            StopReason.CANCELLED,
            StopReason.UNRECOVERABLE_ERROR,
        )
    }
    assert len(codes) == 4


def test_a_verified_answer_is_zero() -> None:
    result = RunResult(run_id="r", stop_reason=StopReason.ANSWERED, verified=True)
    assert exit_code_for(result) == EXIT_OK


# -- running ---------------------------------------------------------------


def test_a_claim_without_evidence_exits_unverified(workspace: Path) -> None:
    """Only claims are checked. "done" asserts nothing; "the tests pass" does.

    The verifier downgrades a checkable claim with nothing to check it
    against — an answer that makes no such claim is not a failure to verify.
    """
    result = _invoke(
        workspace, "run the tests", script=[_answer("The tests pass.")]
    )

    assert result.exit_code == EXIT_UNVERIFIED
    assert "Unverified" in result.stdout


def test_an_answer_that_claims_nothing_is_not_a_verification_failure(
    workspace: Path,
) -> None:
    result = _invoke(workspace, "say hello", script=[_answer("Hello.")])

    assert result.exit_code == EXIT_OK


def test_a_run_that_uses_a_tool_reports_it(workspace: Path) -> None:
    result = _invoke(
        workspace, "look around", script=[LIST_CALL, _answer("there is a README")]
    )

    assert "Tool calls" in result.stdout
    assert "1" in result.stdout


def test_budget_exhaustion_has_its_own_exit_code(workspace: Path) -> None:
    result = _invoke(
        workspace,
        "keep looking",
        "--max-tool-calls",
        "1",
        script=[LIST_CALL, LIST_CALL, LIST_CALL],
    )

    assert result.exit_code == EXIT_BUDGET


def test_no_progress_has_its_own_exit_code(workspace: Path) -> None:
    """The same call three times over is a stuck run, not a slow one."""
    result = _invoke(
        workspace, "spin", script=[LIST_CALL, LIST_CALL, LIST_CALL, LIST_CALL]
    )

    assert result.exit_code == EXIT_NO_PROGRESS


def test_json_output_is_parseable(workspace: Path) -> None:
    result = _invoke(workspace, "say hello", "--json", script=[_answer("done")])

    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == "answered"
    assert payload["run_id"]


def test_no_chain_of_thought_is_printed_or_stored(workspace: Path) -> None:
    """`reasoning` never leaves the model layer, in either output or journal."""
    result = _invoke(
        workspace, "think", "--json", script=[_answer("done")]
    )

    assert "reasoning" not in result.stdout
    journal_files = list((workspace / "runs").rglob("*.jsonl"))
    for path in journal_files:
        assert "reasoning" not in path.read_text(encoding="utf-8")


# -- configuration ---------------------------------------------------------


def test_a_budget_flag_overrides_the_configuration(workspace: Path) -> None:
    result = _invoke(
        workspace, "spin", "--max-steps", "1", "--json", script=[LIST_CALL, LIST_CALL]
    )

    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == "budget_exhausted"


def test_a_broken_configuration_exits_two(workspace: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "anything"],
        env={"HARNESS_WORKSPACE": str(workspace), "HARNESS_RUNTIME_PORT": "nope"},
    )

    assert result.exit_code == EXIT_CONFIG
    assert "Traceback" not in result.stdout


def test_read_only_offers_no_tool_that_changes_anything(workspace: Path) -> None:
    """The cheapest way to guarantee a look-only run is not to offer the tools."""
    from harness.config import HarnessConfig
    from harness.session import build_loop

    config = HarnessConfig.model_validate(
        {"workspace": str(workspace), "database": str(workspace / "m.sqlite")}
    )
    from harness.models import FakeProvider

    loop = build_loop(
        config, goal="look", provider=FakeProvider(["x"]), read_only=True
    )
    try:
        assert "write_file" not in loop.tools.names
        assert "run_command" not in loop.tools.names
        assert "read_file" in loop.tools.names
    finally:
        loop.memory.close()
        loop.journal.close()


# -- resume ----------------------------------------------------------------


def test_resuming_an_unknown_run_is_refused(workspace: Path) -> None:
    result = _invoke(
        workspace, "carry on", "--resume", "run-does-not-exist", script=[_answer("x")]
    )

    assert result.exit_code == EXIT_CONFIG
    assert "resume" in result.stdout


def test_a_run_can_be_resumed_by_id(workspace: Path) -> None:
    """The whole reason the store exists: a cold prompt costs 25 s."""
    first = _invoke(
        workspace,
        "look around",
        "--run-id",
        "run-fixed",
        "--max-tool-calls",
        "1",
        script=[LIST_CALL, LIST_CALL],
    )
    assert first.exit_code == EXIT_BUDGET

    second = _invoke(
        workspace,
        "look around",
        "--resume",
        "run-fixed",
        "--json",
        script=[_answer("there is a README")],
    )

    payload = json.loads(second.stdout)
    assert payload["run_id"] == "run-fixed"
    # The tool call from the first attempt is still counted.
    assert payload["tool_calls"] >= 1


def test_resuming_with_a_different_goal_is_refused(workspace: Path) -> None:
    """Continuing would append to a history that belongs to another task."""
    _invoke(
        workspace,
        "the original goal",
        "--run-id",
        "run-fixed",
        "--max-tool-calls",
        "1",
        script=[LIST_CALL, LIST_CALL],
    )

    result = _invoke(
        workspace,
        "a completely different goal",
        "--resume",
        "run-fixed",
        script=[_answer("x")],
    )

    assert result.exit_code == EXIT_CONFIG


def test_cancelled_runs_report_as_cancelled() -> None:
    result = RunResult(run_id="r", stop_reason=StopReason.CANCELLED)
    assert exit_code_for(result) == EXIT_CANCELLED


# -- approval --------------------------------------------------------------


def test_without_approval_a_confirmable_command_is_denied(workspace: Path) -> None:
    """Fail-closed, and the reason it matters in practice.

    `sed -i` is classified as needing confirmation. An unattended run has
    nobody to ask, so it is denied — correct, and also why a run asked to edit
    a file that way spends its budget being refused rather than working.
    """
    from harness.tools.shell import run_command

    result = run_command(workspace, "sed -i 's/a/b/' README.md")

    assert result.ok is False
    assert result.error_kind == "denied"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "# demo\n"


def test_approval_is_explicit_and_leaves_deny_alone(workspace: Path) -> None:
    """The opt-out approves confirmable commands, never denied ones."""
    from harness.tools.shell import run_command

    approvals: list[str] = []

    def approve(command: str, _reason: str) -> bool:
        approvals.append(command)
        return True

    confirmed = run_command(
        workspace, "sed -i 's/demo/DEMO/' README.md", confirm_callback=approve
    )
    denied = run_command(workspace, "rm -rf /", confirm_callback=approve)

    assert confirmed.ok is True
    assert approvals == ["sed -i 's/demo/DEMO/' README.md"]
    assert denied.ok is False
    assert denied.error_kind == "denied"


# -- against the real model ------------------------------------------------


@pytest.mark.local_llm
def test_a_run_against_the_real_server_completes(workspace: Path) -> None:
    """The wiring works against a real llama-server.

    Deliberately asserts nothing about the model's answer: what it decides to
    do varies between samples, and a test that required a particular decision
    would measure the model rather than this code. Whether the model is any
    good at the task is a benchmark question (#21), not a unit test.
    """
    result = runner.invoke(
        app,
        ["run", "Say hello.", "--max-steps", "2", "--json"],
        env={
            "HARNESS_WORKSPACE": str(workspace),
            "HARNESS_DATABASE": str(workspace / "memory.sqlite"),
            "HARNESS_RUNTIME_ATTACH": "true",
        },
    )

    payload = json.loads(result.stdout)
    assert payload["run_id"]
    assert payload["stop_reason"] in {
        "answered",
        "budget_exhausted",
        "no_progress",
    }
    assert payload["total_prompt_tokens"] > 0
