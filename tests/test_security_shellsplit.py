"""Tests for the splitter that decides *what* the classifier gets to judge.

A defect here is not cosmetic. Every segment the splitter fails to emit is a
command the shell runs and the classifier never sees; every segment it emits
with the wrong head shifts which policy row is matched. The classifier can be
flawless and still release the wrong thing.

So each assertion names the exact segment list rather than a count or a
substring, and the chaining/substitution cases carry a companion assertion on
``classify_command`` that states the verdict the split produces.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from harness.core import Risk
from harness.security import classify_command
from harness.security.shellsplit import SUBSTITUTION_PLACEHOLDER, split_segments

SUBST = SUBSTITUTION_PLACEHOLDER


def executable_segments(command: str) -> list[str]:
    """The non-blank segments, stripped — what the classifier iterates over."""
    return [segment.strip() for segment in split_segments(command) if segment.strip()]


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A separator inside single quotes is data, not syntax.
        ("echo 'a; b'", ["echo 'a; b'"]),
        ("echo 'a && b'", ["echo 'a && b'"]),
        ("echo 'a | b'", ["echo 'a | b'"]),
        # Double quotes suppress the separators too.
        ('echo "a; b"', ['echo "a; b"']),
        ('echo "a && b"', ['echo "a && b"']),
        # A backslash escapes the following character wherever it appears.
        ("echo a\\;b", ["echo a\\;b"]),
        ("echo a\\|b", ["echo a\\|b"]),
        # An escaped quote does not close the double-quoted run, so the ";"
        # after it is still inside quotes.
        ('echo "a\\"b; c"', ['echo "a\\"b; c"']),
    ],
)
def test_quoted_separators_do_not_start_a_new_command(
    command: str, expected: list[str]
) -> None:
    assert split_segments(command) == expected


def test_single_quotes_survive_the_shell_quote_escape_idiom() -> None:
    """``'don'\\''t`` closes and reopens a quote; the ";" stays quoted."""
    command = "echo 'don'\\''t; stop'"

    assert split_segments(command) == [command]


def test_quoted_text_keeps_a_deny_word_out_of_command_position(
    tmp_path: Path,
) -> None:
    """``echo 'shutdown'`` prints a word; it does not power anything down."""
    assert executable_segments("echo 'shutdown'") == ["echo 'shutdown'"]

    risk, _ = classify_command("echo 'shutdown'", workspace=tmp_path)

    assert risk is Risk.ALLOW


# ---------------------------------------------------------------------------
# Command chaining
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls; rm -rf /", ["ls", "rm -rf /"]),
        ("ls && rm -rf /", ["ls", "rm -rf /"]),
        ("ls || rm -rf /", ["ls", "rm -rf /"]),
        ("ls | rm -rf /", ["ls", "rm -rf /"]),
        ("ls & rm -rf /", ["ls", "rm -rf /"]),
        ("ls\nrm -rf /", ["ls", "rm -rf /"]),
        # ";;" and "|&" produce an empty segment between the two commands;
        # both real commands must still surface.
        ("ls;;rm -rf /", ["ls", "rm -rf /"]),
        ("ls |& rm -rf /", ["ls", "rm -rf /"]),
    ],
)
def test_every_chaining_operator_starts_a_new_segment(
    command: str, expected: list[str]
) -> None:
    assert executable_segments(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "ls; shutdown",
        "ls && shutdown",
        "ls || shutdown",
        "ls | shutdown",
        "ls & shutdown",
        "ls\nshutdown",
    ],
)
def test_a_denied_command_cannot_hide_behind_a_chaining_operator(
    command: str, tmp_path: Path
) -> None:
    """The split is what makes the deny rule reachable at all."""
    risk, reason = classify_command(command, workspace=tmp_path)

    assert risk is Risk.DENY
    assert reason == "powers the machine down"


def test_a_line_continuation_keeps_one_command_together() -> None:
    """``\\`` + newline is an escaped pair, so the newline does not split."""
    assert split_segments("ls \\\n  -la") == ["ls \\\n  -la"]


def test_a_continuation_before_an_operator_still_splits() -> None:
    """The continuation joins lines; the "&&" on the next line still separates."""
    assert executable_segments("ls \\\n && shutdown") == ["ls \\", "shutdown"]


def test_a_trailing_backslash_is_kept_rather_than_swallowed() -> None:
    """``i + 1 < n`` fails on a final backslash: it is appended verbatim."""
    assert split_segments("ls \\") == ["ls \\"]


# ---------------------------------------------------------------------------
# Command substitution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat $(id)", [f"cat {SUBST}", "id"]),
        ("cat `id`", [f"cat {SUBST}", "id"]),
        # Substitutions expand inside double quotes but not single quotes.
        ('cat "$(id)"', [f'cat "{SUBST}"', "id"]),
        ("cat '$(id)'", ["cat '$(id)'"]),
        ("cat '`id`'", ["cat '`id`'"]),
        # Nesting: every level becomes its own segment.
        (
            "echo $(echo $(id))",
            [f"echo {SUBST}", f"echo {SUBST}", "id"],
        ),
        # Balanced parentheses inside the body must not end it early.
        ("echo $(echo (x) y)", [f"echo {SUBST}", "echo (x) y"]),
        # A quoted ")" inside the body likewise.
        ("""echo $(echo ")")""", [f"echo {SUBST}", 'echo ")"']),
    ],
)
def test_substitutions_become_their_own_segments(
    command: str, expected: list[str]
) -> None:
    assert executable_segments(command) == expected


def test_the_placeholder_keeps_the_outer_head_command_in_position() -> None:
    """Replacing the substitution with a bare token would shift ``cat``'s args.

    The placeholder is why the outer segment still tokenises to a head plus one
    operand instead of collapsing to a single word.
    """
    outer, _inner = split_segments("cat $(realpath ../secret)")

    assert shlex.split(outer) == ["cat", SUBST]


@pytest.mark.parametrize(
    "command",
    [
        "cat $(shutdown)",
        "cat `shutdown`",
        'cat "$(shutdown)"',
        "echo $(echo $(shutdown))",
    ],
)
def test_a_denied_command_cannot_hide_inside_a_substitution(
    command: str, tmp_path: Path
) -> None:
    risk, reason = classify_command(command, workspace=tmp_path)

    assert risk is Risk.DENY
    assert reason == "powers the machine down"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # Unterminated: the remainder is treated as the body rather than
        # dropped, so the command inside is still classified.
        ("$(rm -rf /", [SUBST, "rm -rf /"]),
        ("`rm -rf /", [SUBST, "rm -rf /"]),
        ("cat `id", [f"cat {SUBST}", "id"]),
    ],
)
def test_an_unterminated_substitution_still_yields_its_body(
    command: str, expected: list[str]
) -> None:
    assert executable_segments(command) == expected


def test_an_escaped_backtick_does_not_close_the_substitution() -> None:
    """``_read_backtick`` skips escaped pairs, so the body runs to the real tick."""
    outer, inner = split_segments("cat `echo \\` x`")

    assert outer == f"cat {SUBST}"
    assert inner == "echo \\` x"


def test_a_backslash_inside_a_double_quoted_substitution_body_is_skipped() -> None:
    """``_read_balanced`` honours ``\\"`` so the quote is not seen as closing."""
    outer, inner = split_segments('cat $(echo "a\\"b)" )')

    assert outer == f"cat {SUBST}"
    assert inner == 'echo "a\\"b)" '


def test_the_substitution_depth_cap_bounds_the_recursion() -> None:
    """Nesting beyond the cap stops producing segments (see the xfail below)."""
    command = "echo x"
    for _ in range(12):
        command = f"echo $({command})"

    # One segment per level up to the cap, plus the outermost one.
    assert len(split_segments(command)) == 9


# ---------------------------------------------------------------------------
# Redirection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # "&" that belongs to a redirect is plumbing, not a separator.
        ("ls 2>&1", ["ls 2>&1"]),
        ("ls > out 2>&1", ["ls > out 2>&1"]),
        ("ls &> log", ["ls &> log"]),
        ("ls&>log", ["ls&>log"]),
        ("ls >> log", ["ls >> log"]),
        # A redirect does not end the command, so a following ";" still does.
        ("ls 2>>log; cat x", ["ls 2>>log", "cat x"]),
    ],
)
def test_redirection_plumbing_is_not_mistaken_for_a_separator(
    command: str, expected: list[str]
) -> None:
    assert executable_segments(command) == expected


def test_a_backgrounding_ampersand_after_a_redirect_target_still_splits() -> None:
    """``&`` is only plumbing directly after ">"; here it backgrounds ``ls``."""
    assert executable_segments("ls > out & shutdown") == ["ls > out", "shutdown"]


# ---------------------------------------------------------------------------
# Known gaps — see the report on issue #32
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GAP: process substitution '<(...)' / '>(...)' is never lifted into a "
        "segment, so the command inside it is executed by /bin/sh but never "
        "classified. Fixing this changes security behaviour and belongs in its "
        "own issue (#32 is scope-limited to making the current policy testable)."
    ),
)
def test_process_substitution_is_split_out() -> None:
    assert executable_segments("diff <(shutdown) file.txt") == [
        f"diff {SUBST} file.txt",
        "shutdown",
    ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GAP: process substitution reaches the classifier as an operand, not a "
        "command, so 'cat <(reboot)' is ALLOW while /bin/sh runs reboot. "
        "Behaviour change — own issue."
    ),
)
def test_a_denied_command_cannot_hide_inside_a_process_substitution(
    tmp_path: Path,
) -> None:
    risk, _ = classify_command("cat <(reboot)", workspace=tmp_path)

    assert risk is not Risk.ALLOW


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GAP: the substitution depth cap fails open. Beyond eight levels the "
        "nested body is dropped instead of being reported, so a deeply nested "
        "'shutdown' classifies as ALLOW. Behaviour change — own issue."
    ),
)
def test_substitution_deeper_than_the_cap_is_not_silently_allowed(
    tmp_path: Path,
) -> None:
    command = "shutdown"
    for _ in range(9):
        command = f"echo $({command})"

    risk, _ = classify_command(command, workspace=tmp_path)

    assert risk is not Risk.ALLOW
