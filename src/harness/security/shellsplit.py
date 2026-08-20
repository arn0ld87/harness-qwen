"""Shell-aware splitting of a command string into executable segments.

A classifier that looks only at the first word of a command is trivially
bypassed with ``ls; rm -rf /``. This module reproduces enough of a shell's
parsing to find every point at which a new command starts: the separators
``;``, ``&&``, ``||``, ``|``, ``&`` and newline, plus the command
substitutions ``$(...)`` and backticks, which are executed too.

Quoting is honoured, so ``echo "a; b"`` stays one segment and ``2>&1`` is not
mistaken for a background separator.
"""

from __future__ import annotations

# Substitutions are lifted into their own segments; the outer segment keeps a
# placeholder so token positions (and therefore the head command) survive.
SUBSTITUTION_PLACEHOLDER = "__harness_subst__"

TOO_DEEP = "__harness_substitution_too_deep__"
"""Emitted instead of an unread body when nesting exceeds the cap.

A segment rather than an exception: callers split commands they did not write
and must still get an answer, and this answer is "I could not read all of it",
which classifies as CONFIRM."""

_MAX_DEPTH = 8


def split_segments(command: str, depth: int = 0) -> list[str]:
    """Split into the segments a shell would execute, substitutions included."""
    segments: list[str] = []
    nested: list[str] = []
    current: list[str] = []
    in_single = in_double = False
    i, n = 0, len(command)

    def flush() -> None:
        segments.append("".join(current))
        current.clear()

    while i < n:
        ch = command[i]
        if in_single:
            current.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            current.append(command[i:i + 2])
            i += 2
            continue
        if command.startswith("$(", i):
            inner, i = _read_balanced(command, i + 2)
            nested.append(inner)
            current.append(SUBSTITUTION_PLACEHOLDER)
            continue
        # Process substitution runs its body exactly like $(...) does; only
        # the plumbing differs. Left in place it reaches the classifier as an
        # operand, and "<(reboot)" then reads as a relative path — which is
        # how a denied command travelled inside an allowed one.
        if command.startswith("<(", i) or command.startswith(">(", i):
            inner, i = _read_balanced(command, i + 2)
            nested.append(inner)
            current.append(SUBSTITUTION_PLACEHOLDER)
            continue
        if ch == "`":
            inner, i = _read_backtick(command, i + 1)
            nested.append(inner)
            current.append(SUBSTITUTION_PLACEHOLDER)
            continue
        if in_double:
            current.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            current.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            current.append(ch)
            i += 1
            continue
        if command.startswith("&&", i) or command.startswith("||", i):
            flush()
            i += 2
            continue
        # "2>&1" and "&>log" are file descriptor plumbing, not separators.
        if ch == "&" and ("".join(current).rstrip().endswith(">")
                          or command.startswith("&>", i)):
            current.append(ch)
            i += 1
            continue
        if ch in ";|&\n":
            flush()
            i += 1
            continue
        current.append(ch)
        i += 1
    flush()

    if depth < _MAX_DEPTH:
        for inner in nested:
            segments.extend(split_segments(inner, depth + 1))
    elif nested:
        # The cap stops runaway recursion, but dropping the body would let a
        # deeply nested command through unread — a limit that fails open is
        # worse than no limit, because it looks like a verdict.
        segments.append(TOO_DEEP)
    return segments


def _read_balanced(text: str, start: int) -> tuple[str, int]:
    """Read to the ``)`` closing a ``$(`` that opened just before ``start``."""
    depth = 1
    i = start
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    # Unbalanced: treat the remainder as the substitution body rather than
    # dropping it, so an unterminated "$(rm -rf /" is still classified.
    return text[start:], len(text)


def _read_backtick(text: str, start: int) -> tuple[str, int]:
    i = start
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == "`":
            return text[start:i], i + 1
        i += 1
    return text[start:], len(text)
