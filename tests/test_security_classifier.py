from pathlib import Path

import pytest

from harness.core import Risk
from harness.security import classify_command
from harness.tools.shell import run_command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat file.txt", Risk.ALLOW),
        ("cat ../secret", Risk.DENY),
        ("cat /etc/passwd", Risk.DENY),
        ("cat ~/.ssh/id_rsa", Risk.DENY),
        ("cat $HOME/.ssh/id_rsa", Risk.DENY),
        ("grep foo file.txt", Risk.ALLOW),
        ("grep foo /etc/passwd", Risk.DENY),
        ("find .", Risk.ALLOW),
        ("find /home", Risk.DENY),
    ],
)
def test_read_commands_are_confined_to_workspace(
    tmp_path: Path,
    command: str,
    expected: Risk,
) -> None:
    (tmp_path / "file.txt").write_text("foo\n", encoding="utf-8")

    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is expected


@pytest.mark.parametrize(
    "command",
    [
        "cat $(realpath ../secret)",
        "sh -c 'cat ../secret'",
        "bash -c 'cat /etc/passwd'",
        "env X=1 cat ~/.ssh/id_rsa",
        "env --chdir=/etc cat passwd",
        "cat </etc/passwd",
    ],
)
def test_shell_escape_constructions_are_never_allowed(
    tmp_path: Path,
    command: str,
) -> None:
    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is not Risk.ALLOW


def test_symlink_escape_is_denied(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)

    risk, _ = classify_command("cat escape", workspace=tmp_path)

    assert risk is Risk.DENY


def test_run_command_reads_workspace_file_inside_sandbox(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("visible\n", encoding="utf-8")

    result = run_command(tmp_path, "cat file.txt")

    assert result.ok is True
    assert "visible" in result.content


def test_approved_shell_wrapper_still_cannot_read_etc_passwd(tmp_path: Path) -> None:
    result = run_command(
        tmp_path,
        "sh -c 'cat /etc/passwd'",
        confirm_callback=lambda _command, _reason: True,
    )

    assert result.ok is False
    assert result.exit_code != 0
