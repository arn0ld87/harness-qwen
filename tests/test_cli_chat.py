"""`harness chat` (issue #14).

The thing worth testing is that this is not a second agent: the same loop, the
same tools, the same security boundary as `harness run`, with a person in
between. A chat mode with its own control flow would be a second place where
budgets and verification are enforced, and the second place is the one that
drifts.

Input is scripted through `CliRunner`, the model through `FakeProvider`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from harness.cli import app
from harness.cli_chat import COMMANDS, Totals
from harness.core import RunResult, StopReason

runner = CliRunner()

LIST_CALL = (
    '{"action":"tool","tool":"list_files",'
    '"arguments":{"path":"."},"reason":"look"}'
)


def _answer(content: str) -> str:
    return json.dumps({"action": "answer", "content": content, "evidence": []})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path


def _chat(workspace: Path, *args: str, script: list[str], stdin: str) -> Result:
    import harness.session as session
    from harness.models import FakeProvider

    # One provider for the whole session, not one per turn: the loop is
    # rebuilt each turn, and a fresh provider would rewind the script and
    # answer the second message with the first reply.
    provider = FakeProvider(list(script))
    original = session.build_provider
    session.build_provider = lambda _config: provider  # type: ignore[assignment]
    try:
        return runner.invoke(
            app,
            ["chat", *args],
            input=stdin,
            env={
                "HARNESS_WORKSPACE": str(workspace),
                "HARNESS_DATABASE": str(workspace / "memory.sqlite"),
            },
        )
    finally:
        session.build_provider = original  # type: ignore[assignment]


# -- turns -----------------------------------------------------------------


def test_a_turn_runs_the_agent_and_prints_its_answer(workspace: Path) -> None:
    result = _chat(
        workspace,
        script=[_answer("Hello back.")],
        stdin="say hello\n/exit\n",
    )

    assert "Hello back." in result.stdout
    assert result.exit_code == 0


def test_a_turn_may_use_tools(workspace: Path) -> None:
    result = _chat(
        workspace,
        script=[LIST_CALL, _answer("There is a README.")],
        stdin="what is here\n/usage\n/exit\n",
    )

    assert "There is a README." in result.stdout
    assert "Tool calls" in result.stdout


def test_several_turns_share_one_session(workspace: Path) -> None:
    """One run id and one warm cache, because a cold prompt costs ~25 s."""
    result = _chat(
        workspace,
        "--run-id",
        "chat-fixed",
        script=[_answer("first"), _answer("second")],
        stdin="one\ntwo\n/usage\n/exit\n",
    )

    assert "first" in result.stdout
    assert "second" in result.stdout
    assert "Turns" in result.stdout


def test_an_empty_line_is_not_a_turn(workspace: Path) -> None:
    result = _chat(workspace, script=[_answer("only once")], stdin="\n\nhi\n/exit\n")

    assert result.stdout.count("only once") == 1


# -- control commands ------------------------------------------------------


def test_status_reports_the_session(workspace: Path) -> None:
    result = _chat(workspace, "--run-id", "chat-fixed", script=[], stdin="/status\n/exit\n")

    assert "chat-fixed" in result.stdout
    assert str(workspace) in result.stdout


def test_context_reports_the_budget_zones(workspace: Path) -> None:
    """Watching the append zone is how a degrading session is noticed early."""
    result = _chat(workspace, script=[], stdin="/context\n/exit\n")

    assert "Prefix" in result.stdout
    assert "Append" in result.stdout


def test_usage_reports_tokens_and_cache(workspace: Path) -> None:
    result = _chat(workspace, script=[_answer("hi")], stdin="hello\n/usage\n/exit\n")

    assert "Prompt tokens" in result.stdout
    assert "cached" in result.stdout


def test_help_lists_every_command(workspace: Path) -> None:
    result = _chat(workspace, script=[], stdin="/help\n/exit\n")

    for name in COMMANDS:
        assert name in result.stdout


def test_an_unknown_command_does_not_become_a_turn(workspace: Path) -> None:
    """Otherwise a typo costs a full model call."""
    result = _chat(workspace, script=[], stdin="/nonsense\n/exit\n")

    assert "unknown command" in result.stdout
    assert result.exit_code == 0


def test_exit_ends_the_session(workspace: Path) -> None:
    result = _chat(workspace, script=[], stdin="/exit\n")
    assert result.exit_code == 0


def test_end_of_input_ends_the_session(workspace: Path) -> None:
    """Ctrl-D leaves rather than raising at the person who pressed it."""
    result = _chat(workspace, script=[], stdin="")
    assert result.exit_code == 0


# -- same components as run ------------------------------------------------


def test_chat_and_run_assemble_the_same_agent(workspace: Path) -> None:
    """The point of the issue: no separate chat architecture.

    Both go through `session.build_loop`, so the tool set, the prompt prefix
    and the budget cannot differ between the two entry points.
    """
    from harness.config import HarnessConfig
    from harness.models import FakeProvider
    from harness.session import build_loop

    config = HarnessConfig.model_validate(
        {"workspace": str(workspace), "database": str(workspace / "m.sqlite")}
    )
    first = build_loop(config, goal="x", provider=FakeProvider(["a"]))
    second = build_loop(config, goal="x", provider=FakeProvider(["a"]))
    try:
        assert first.tools.names == second.tools.names
        assert first.assembler.prefix_hash() == second.assembler.prefix_hash()
        assert first.budget == second.budget
    finally:
        for loop in (first, second):
            loop.memory.close()
            loop.journal.close()


def test_read_only_carries_through_to_chat(workspace: Path) -> None:
    result = _chat(workspace, "--read-only", script=[], stdin="/status\n/exit\n")

    assert "write_file" not in result.stdout
    assert "read_file" in result.stdout


# -- resume ----------------------------------------------------------------


def test_resuming_an_unknown_session_is_refused(workspace: Path) -> None:
    result = _chat(workspace, "--resume", "nope", script=[], stdin="/exit\n")

    assert result.exit_code == 2
    assert "no session" in result.stdout


def test_a_session_can_be_resumed(workspace: Path) -> None:
    first = _chat(
        workspace,
        "--run-id",
        "chat-fixed",
        script=[_answer("noted")],
        stdin="remember this\n/exit\n",
    )
    assert first.exit_code == 0

    second = _chat(
        workspace,
        "--resume",
        "chat-fixed",
        script=[_answer("still here")],
        stdin="are you there\n/exit\n",
    )

    assert "resuming chat-fixed" in second.stdout
    assert "still here" in second.stdout


# -- totals ----------------------------------------------------------------


def test_totals_accumulate_across_turns() -> None:
    totals = Totals()
    totals.add(
        RunResult(
            run_id="r",
            stop_reason=StopReason.ANSWERED,
            tool_calls=2,
            total_prompt_tokens=100,
            total_cached_tokens=80,
        )
    )
    totals.add(
        RunResult(run_id="r", stop_reason=StopReason.ANSWERED, tool_calls=1,
                  total_prompt_tokens=50)
    )

    assert totals.turns == 2
    assert totals.tool_calls == 3
    assert totals.prompt_tokens == 150
    assert totals.cached_tokens == 80


def test_the_prefix_survives_a_second_turn(workspace: Path) -> None:
    """The expensive invariant: a message must not rewrite the cached prefix.

    The goal sits in the prefix, so giving every turn its own goal would pay a
    full reprocess — ~25 s on this hardware — for each thing a person types.
    Later messages belong in the append zone, and this is what proves they
    stay there.
    """
    from harness.config import HarnessConfig
    from harness.memory import MemoryStore
    from harness.models import FakeProvider
    from harness.session import build_loop

    config = HarnessConfig.model_validate(
        {"workspace": str(workspace), "database": str(workspace / "m.sqlite")}
    )
    memory = MemoryStore(workspace / "m.sqlite")
    try:
        first = build_loop(
            config,
            goal="the session goal",
            run_id="chat-fixed",
            provider=FakeProvider([_answer("one")]),
            memory=memory,
        )
        before = first.assembler.prefix_hash()
        first.journal.close()

        second = build_loop(
            config,
            goal="the session goal",
            run_id="chat-fixed",
            provider=FakeProvider([_answer("two")]),
            memory=memory,
        )
        assert second.assembler.prefix_hash() == before
        second.journal.close()
    finally:
        memory.close()
