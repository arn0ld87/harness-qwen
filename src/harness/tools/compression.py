"""Tool output compression.

Raw tool output never enters the model context (CONTEXT.md section 6). What
survives is the part a next step can act on: the exit code, stderr, lines that
name an error or a file, and a head/tail window of the body. Everything else is
replaced by a marker stating how many lines were dropped, so a truncation is
always visible instead of silent.

Compression is lossy by design and the loss is recoverable: the caller writes
the full output to the run directory and puts its id in
``ToolResult.full_output_ref``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

OutputKind = Literal["generic", "pytest", "diff", "grep"]

DEFAULT_MAX_LINES = 120
DEFAULT_MAX_BYTES = 8_000
STDERR_MAX_LINES = 40
STDERR_MAX_BYTES = 2_000
MAX_LINE_CHARS = 500

ELISION = "... [{count} lines elided] ..."

# Lines worth keeping even when they sit in the elided middle. Deliberately
# broader than "error": a step that cannot see why a command failed spends its
# next turn re-running the same command.
_ERROR_PATTERN = re.compile(
    r"\b(?:error|errors|failed|failing|failure|fatal|traceback|exception|panic|"
    r"assertionerror|assertion|denied|refused|unsupported|segmentation fault|"
    r"core dumped|timeout|timed out|no such file|not found|permission)\b"
    r"|^\s*E\s|^\s*File \"",
    re.IGNORECASE,
)

# A path, so the model can name the file it wants to look at next. Bare words
# with a dot are excluded unless the suffix is a known source extension —
# otherwise every ``self.method`` reference counts as a file path.
_PATH_PATTERN = re.compile(
    r"(?:\.{0,2}/[\w.+@-]+(?:/[\w.+@-]+)*)"
    r"|(?:[\w.+@-]+/[\w.+@-]+)"
    r"|(?:\b[\w.+-]+\.(?:py|pyi|js|mjs|ts|tsx|jsx|go|rs|c|h|cc|cpp|hpp|java|kt|"
    r"rb|sh|zsh|toml|yaml|yml|json|md|txt|cfg|ini|sql|html|css|lock)\b)"
)

# Everything from the first failure banner to the end of a pytest run is the
# part with diagnostic value; the short summary and the final counts line both
# live inside that span.
_PYTEST_BANNER = re.compile(r"^=+\s*(?:FAILURES|ERRORS|short test summary info)\s*=+", re.I)
_PYTEST_KEEP = re.compile(r"^(?:FAILED|ERROR)\b|^E\s|^>\s|^_{4,}|^\S+\.py:\d+:")

_DIFF_KEEP = re.compile(
    r"^(?:diff --git |index |--- |\+\+\+ |@@ |old mode |new mode |new file mode |"
    r"deleted file mode |rename from |rename to |similarity index |Binary files )"
)

_GREP_KEEP = re.compile(r"^[^:]+:\d+:")


@dataclass(frozen=True, slots=True)
class CompressedOutput:
    """A context-sized view of tool output plus what it cost to make it."""

    text: str
    original_bytes: int
    original_lines: int
    kept_lines: int
    elided_lines: int
    truncated: bool


def detect_kind(command: str) -> OutputKind:
    """Guess the output shape from the command that produced it."""
    head = command.strip().lower()
    if "pytest" in head or " test" in head or head.startswith("test"):
        return "pytest"
    if "diff" in head or "git show" in head:
        return "diff"
    if head.startswith(("rg ", "grep ", "ag ")) or " | grep" in head or " | rg" in head:
        return "grep"
    return "generic"


def compress_output(
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    kind: OutputKind = "generic",
    max_line_chars: int = MAX_LINE_CHARS,
) -> CompressedOutput:
    """Reduce ``text`` to at most ``max_lines`` lines and ``max_bytes`` bytes.

    Selection runs in priority order — tail window, kind-specific keepers, head
    window, error lines, path lines — so that when the budget runs out it is the
    least informative lines that go. The result is never cut in the middle of a
    multi-byte character.
    """
    original_bytes = len(text.encode("utf-8"))
    lines = text.splitlines()
    original_lines = len(lines)

    if original_lines <= max_lines and original_bytes <= max_bytes:
        return CompressedOutput(
            text=text,
            original_bytes=original_bytes,
            original_lines=original_lines,
            kept_lines=original_lines,
            elided_lines=0,
            truncated=False,
        )

    order = _select(lines, max_lines=max_lines, kind=kind)
    rendered = _render(lines, order, max_line_chars=max_line_chars)

    # Dropping the lowest-priority lines beats blind byte truncation: it keeps
    # the elision marker honest instead of cutting the tail off mid-sentence.
    while len(rendered.encode("utf-8")) > max_bytes and len(order) > 1:
        order.pop()
        rendered = _render(lines, order, max_line_chars=max_line_chars)

    if len(rendered.encode("utf-8")) > max_bytes:
        rendered = _truncate_utf8(rendered, max_bytes)

    kept = len(set(order))
    return CompressedOutput(
        text=rendered,
        original_bytes=original_bytes,
        original_lines=original_lines,
        kept_lines=kept,
        elided_lines=original_lines - kept,
        truncated=True,
    )


def compress_command_output(
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    command: str | None = None,
    kind: OutputKind = "generic",
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> CompressedOutput:
    """Assemble the context view of a command: exit code, stderr, then stdout.

    stderr gets its own small budget rather than competing with stdout, because
    the failure reason is usually there and is usually short.
    """
    header: list[str] = []
    if command:
        header.append(f"$ {command}")
    header.append(f"exit code: {exit_code}")
    header_text = "\n".join(header)
    header_bytes = len(header_text.encode("utf-8"))

    err = compress_output(
        stderr, max_lines=STDERR_MAX_LINES, max_bytes=min(STDERR_MAX_BYTES, max_bytes), kind=kind
    )
    stdout_bytes = max(256, max_bytes - header_bytes - len(err.text.encode("utf-8")))
    out = compress_output(stdout, max_lines=max_lines, max_bytes=stdout_bytes, kind=kind)

    parts = [header_text]
    if stderr.strip():
        parts.append("--- stderr ---")
        parts.append(err.text)
    parts.append("--- stdout ---")
    parts.append(out.text if stdout.strip() else "(empty)")

    return CompressedOutput(
        text="\n".join(parts),
        original_bytes=err.original_bytes + out.original_bytes,
        original_lines=err.original_lines + out.original_lines,
        kept_lines=err.kept_lines + out.kept_lines,
        elided_lines=err.elided_lines + out.elided_lines,
        truncated=err.truncated or out.truncated,
    )


def _select(lines: list[str], *, max_lines: int, kind: OutputKind) -> list[int]:
    """Return kept line indices in descending priority (lowest priority last)."""
    total = len(lines)
    head_n = max(1, max_lines // 3)
    tail_n = max(1, max_lines // 2)

    order: list[int] = []
    seen: set[int] = set()

    def take(index: int) -> bool:
        if index in seen:
            return True
        if len(seen) >= max_lines:
            return False
        seen.add(index)
        order.append(index)
        return True

    for i in range(max(0, total - tail_n), total):
        take(i)

    for i in _kind_keepers(lines, kind):
        if not take(i):
            break

    for i in range(min(head_n, total)):
        if not take(i):
            break

    for pattern in (_ERROR_PATTERN, _PATH_PATTERN):
        for i in range(total - 1, -1, -1):
            if i in seen:
                continue
            if pattern.search(lines[i]) and not take(i):
                break

    return order


def _kind_keepers(lines: list[str], kind: OutputKind) -> list[int]:
    """Indices this output shape must not lose, most important first."""
    if kind == "pytest":
        banner = next((i for i, ln in enumerate(lines) if _PYTEST_BANNER.match(ln)), None)
        span = list(range(len(lines) - 1, banner - 1, -1)) if banner is not None else []
        extra = [i for i in range(len(lines) - 1, -1, -1) if _PYTEST_KEEP.match(lines[i])]
        return span + extra
    if kind == "diff":
        return [i for i, ln in enumerate(lines) if _DIFF_KEEP.match(ln)]
    if kind == "grep":
        return [i for i, ln in enumerate(lines) if _GREP_KEEP.match(ln)]
    return []


def _render(lines: list[str], order: list[int], *, max_line_chars: int) -> str:
    kept = sorted(set(order))
    out: list[str] = []
    previous = -1
    for index in kept:
        gap = index - previous - 1
        if gap > 0:
            out.append(ELISION.format(count=gap))
        out.append(_clip_line(lines[index], max_line_chars))
        previous = index
    trailing = len(lines) - previous - 1
    if trailing > 0:
        out.append(ELISION.format(count=trailing))
    return "\n".join(out)


def _clip_line(line: str, max_chars: int) -> str:
    """Stop one long line from consuming the whole byte budget."""
    if len(line) <= max_chars:
        return line
    return f"{line[:max_chars]} ... [+{len(line) - max_chars} chars]"


def _truncate_utf8(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` bytes on a character boundary."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    cut = raw[:limit]
    end = len(cut)
    start = end - 1
    while start >= 0 and (cut[start] & 0xC0) == 0x80:
        start -= 1
    if start >= 0:
        lead = cut[start]
        if lead < 0x80:
            width = 1
        elif lead >= 0xF0:
            width = 4
        elif lead >= 0xE0:
            width = 3
        elif lead >= 0xC0:
            width = 2
        else:
            width = 1
        if start + width > end:
            end = start
    return cut[:end].decode("utf-8")
