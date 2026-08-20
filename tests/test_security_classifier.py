from pathlib import Path

import pytest

from harness.core import Risk
from harness.security import classifier, classify_command
from harness.security.shellsplit import split_segments
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


# ---------------------------------------------------------------------------
# Severity: deny beats confirm beats allow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A denial anywhere in the chain wins, whichever end it sits at.
        ("ls && shutdown", Risk.DENY),
        ("shutdown && ls", Risk.DENY),
        ("ls; curl example.com; shutdown", Risk.DENY),
        ("shutdown; curl example.com; ls", Risk.DENY),
        # Without a denial, the confirm-worthy segment still wins over allow.
        ("ls; curl example.com", Risk.CONFIRM),
        ("curl example.com; ls", Risk.CONFIRM),
        # Only an all-allow chain stays allowed.
        ("ls; pwd; git status", Risk.ALLOW),
    ],
)
def test_the_most_severe_segment_decides_the_verdict(
    command: str, expected: Risk, tmp_path: Path
) -> None:
    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is expected


def test_the_reported_reason_belongs_to_the_most_severe_segment(
    tmp_path: Path,
) -> None:
    _risk, reason = classify_command("ls; curl example.com; ls", workspace=tmp_path)

    assert reason == "network access"


# ---------------------------------------------------------------------------
# Anything unrecognised lands in CONFIRM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("", "empty command"),
        ("   ", "empty command"),
        (";", "command contains no executable segment"),
        ("; ;", "command contains no executable segment"),
        ("frobnicate --all", "unrecognised command: frobnicate"),
        ("./build.sh", "unrecognised command: build.sh"),
        ("sudo", "no command after prefixes"),
        ("env FOO=1", "no command after prefixes"),
    ],
)
def test_what_the_policy_cannot_classify_is_confirmed_not_allowed(
    command: str, reason: str, tmp_path: Path
) -> None:
    """An unknown command is not a safe command — the module docstring's rule."""
    risk, actual = classify_command(command, workspace=tmp_path)

    assert risk is Risk.CONFIRM
    assert actual == reason


@pytest.mark.parametrize("command", ["ls '", 'ls "', "ls 'a"])
def test_a_command_that_cannot_be_tokenised_is_confirmed(
    command: str, tmp_path: Path
) -> None:
    """Failing to parse must fail closed: no tokens is no evidence of safety."""
    risk, reason = classify_command(command, workspace=tmp_path)

    assert risk is Risk.CONFIRM
    assert reason.startswith("command could not be parsed")


# ---------------------------------------------------------------------------
# Whole-string rules, checked before any splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [":(){ :|:& };:", "b() { b|b& }; b"])
def test_a_fork_bomb_is_denied_before_the_string_is_split(
    command: str, tmp_path: Path
) -> None:
    """Splitting on ";" and "|" is what the syntax abuses, so the pattern runs
    against the raw string first."""
    risk, reason = classify_command(command, workspace=tmp_path)

    assert (risk, reason) == (Risk.DENY, "fork bomb")


@pytest.mark.parametrize(
    "command",
    [
        "psql -c 'DROP DATABASE prod'",
        'mysql -e "drop   database prod"',
        "echo 'Drop Database prod' | psql",
    ],
)
def test_dropping_a_database_is_denied_wherever_it_appears(
    command: str, tmp_path: Path
) -> None:
    risk, reason = classify_command(command, workspace=tmp_path)

    assert (risk, reason) == (Risk.DENY, "drops a database")


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "sudo shutdown",
        "sudo -n shutdown",
        "env FOO=1 shutdown",
        "FOO=1 BAR=2 shutdown",
        "timeout 5 shutdown",
        "nice -n 10 shutdown",
        "sudo env FOO=1 timeout 30s shutdown",
    ],
)
def test_wrapper_prefixes_never_hide_a_denied_command(
    command: str, tmp_path: Path
) -> None:
    risk, reason = classify_command(command, workspace=tmp_path)

    assert (risk, reason) == (Risk.DENY, "powers the machine down")


def test_an_option_that_consumes_its_argument_does_not_hide_the_command(
    tmp_path: Path,
) -> None:
    """``sudo -u root shutdown``: both successors of ``-u`` are scanned, so the
    denial is found whether or not the option took an argument."""
    risk, reason = classify_command("sudo -u root shutdown", workspace=tmp_path)

    assert (risk, reason) == (Risk.DENY, "powers the machine down")


def test_a_leading_environment_assignment_does_not_change_an_allowed_command(
    tmp_path: Path,
) -> None:
    risk, _ = classify_command("FOO=1 pwd", workspace=tmp_path)

    assert risk is Risk.ALLOW


# ---------------------------------------------------------------------------
# Redirection
# ---------------------------------------------------------------------------


def test_a_redirect_into_a_block_device_is_denied(tmp_path: Path) -> None:
    risk, reason = classify_command("echo x > /dev/sda", workspace=tmp_path)

    assert (risk, reason) == (
        Risk.DENY, "writes directly to the block device /dev/sda"
    )


def test_a_block_device_redirect_wins_over_an_earlier_harmless_one(
    tmp_path: Path,
) -> None:
    risk, reason = classify_command(
        "echo x > out.txt > /dev/sda", workspace=tmp_path
    )

    assert (risk, reason) == (
        Risk.DENY, "writes directly to the block device /dev/sda"
    )


def test_a_redirect_into_a_workspace_file_needs_confirmation(
    tmp_path: Path,
) -> None:
    """Writing a file is not read-only, so ``ls > out.txt`` leaves ALLOW."""
    risk, reason = classify_command("ls > out.txt", workspace=tmp_path)

    assert (risk, reason) == (Risk.CONFIRM, "redirects output into out.txt")


@pytest.mark.parametrize("command", ["ls 2>&1", "ls >&2", "ls >"])
def test_descriptor_plumbing_is_not_treated_as_a_write(
    command: str, tmp_path: Path
) -> None:
    """``2>&1`` duplicates a descriptor and a ``>`` with no target writes
    nothing, so neither may lift a read-only command."""
    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is Risk.ALLOW


def test_a_harmless_sink_does_not_replace_the_real_reason(tmp_path: Path) -> None:
    """``/dev/null`` swallows the output, so the reason stays the command's own."""
    risk, reason = classify_command("rm build.log > /dev/null", workspace=tmp_path)

    assert (risk, reason) == (Risk.CONFIRM, "deletes files")


# ---------------------------------------------------------------------------
# Workspace confinement without a configured workspace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["cat /etc/passwd", "cat ../secret", "cat sub/../../secret", "cat $HOME/x"],
)
def test_without_a_workspace_any_escaping_operand_is_denied(command: str) -> None:
    """With no root to resolve against, absolute and ``..`` paths are refused."""
    risk, reason = classify_command(command)

    assert risk is Risk.DENY
    assert "outside the workspace boundary" in reason


@pytest.mark.parametrize("command", ["cat file.txt", "grep -n foo src/main.py"])
def test_without_a_workspace_relative_operands_stay_allowed(command: str) -> None:
    risk, _ = classify_command(command)

    assert risk is Risk.ALLOW


def test_an_empty_operand_is_not_a_catastrophic_path(tmp_path: Path) -> None:
    """``rm -rf ""`` deletes nothing, so it is a confirm and not a denial."""
    risk, reason = classify_command('rm -rf ""', workspace=tmp_path)

    assert (risk, reason) == (Risk.CONFIRM, "deletes files")


def test_a_double_dash_is_not_treated_as_a_path_operand(tmp_path: Path) -> None:
    """``--`` only ends option parsing; it is not a file that could escape."""
    (tmp_path / "file.txt").write_text("x\n", encoding="utf-8")

    risk, _ = classify_command("cat -- file.txt", workspace=tmp_path)

    assert risk is Risk.ALLOW


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("rm --no-preserve-root -rf /",
         "removes the filesystem root with --no-preserve-root"),
        ("rm --recursive /etc", "recursively deletes /etc"),
        ("chmod --recursive 777 /usr", "recursively changes permissions on /usr"),
    ],
)
def test_long_flags_are_parsed_by_name_for_the_deny_rules(
    command: str, reason: str, tmp_path: Path
) -> None:
    risk, actual = classify_command(command, workspace=tmp_path)

    assert (risk, actual) == (Risk.DENY, reason)


def test_a_flag_terminator_disarms_a_recursive_delete(tmp_path: Path) -> None:
    """``rm -- -rf /`` deletes files named ``-rf`` and ``/``, not recursively."""
    risk, reason = classify_command("rm -- -rf", workspace=tmp_path)

    assert (risk, reason) == (Risk.CONFIRM, "deletes files")


def test_a_flag_value_is_checked_as_a_path(tmp_path: Path) -> None:
    """``--file=../x`` carries an operand that would otherwise never be seen."""
    risk, reason = classify_command("grep --file=../x y", workspace=tmp_path)

    assert risk is Risk.DENY
    assert reason.endswith("../x")


def test_an_input_redirect_target_is_checked_as_a_path(tmp_path: Path) -> None:
    risk, _ = classify_command("cat 2</etc/passwd", workspace=tmp_path)

    assert risk is Risk.DENY


# ---------------------------------------------------------------------------
# One splitter, not two
# ---------------------------------------------------------------------------


def test_the_classifier_uses_the_exported_splitter() -> None:
    """There must not be a second copy to drift from (#33).

    ``classifier.py`` used to carry a private duplicate of the splitter, and
    only the duplicate decided what got classified — so ``shellsplit.py``
    could be tested to exhaustion and prove nothing about the enforcing code.
    The two had already drifted in their reason strings when this was found.
    """
    assert not hasattr(classifier, "_split_segments")
    assert classifier.split_segments is split_segments


@pytest.mark.parametrize(
    "command",
    [
        "ls; rm -rf /",
        "ls && rm -rf / || true",
        "echo $(rm -rf /)",
        "echo `rm -rf /`",
        "cat <(rm -rf /)",
        "ls 2>&1 &> log & rm -rf /",
        "cat $(echo $(rm -rf /))",
        "$(rm -rf /",
        "cat `rm -rf /",
    ],
)
def test_a_denied_command_is_found_in_every_shell_construct(
    command: str, tmp_path: Path
) -> None:
    """Whatever the plumbing, the shell still runs it — so it is classified.

    Each of these is a way to reach the same command; an unbalanced opener is
    included because the shell would still execute what follows it.
    """
    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is Risk.DENY


# ---------------------------------------------------------------------------
# Audit trail (closed in #33)
# ---------------------------------------------------------------------------


def test_an_allowed_command_reports_its_own_reason(tmp_path: Path) -> None:
    _risk, reason = classify_command("ls", workspace=tmp_path)

    assert reason == "read-only development command: ls"
