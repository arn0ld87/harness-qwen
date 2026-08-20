"""Filesystem tools: read, write, list, and search files."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from harness.core import ToolError, ToolResult
from harness.tools._security import default_resolver
from harness.tools.compression import compress_output


def read_file(workspace: Path, path: str, offset: int = 0, limit: int | None = None) -> ToolResult:
    """Read a file with optional line-based pagination, detecting binary files."""
    started = time.perf_counter()
    def _err(kind: str, msg: str) -> ToolResult:
        return ToolResult(tool="read_file", ok=False, error_kind=kind, content=msg,
                         duration_ms=(time.perf_counter() - started) * 1000.0)
    try:
        resolved = default_resolver(Path(workspace), path)
    except ToolError as exc:
        return _err(exc.kind, str(exc))
    if not resolved.exists():
        return _err("not_found", f"File not found: {path}")
    if resolved.is_dir():
        return _err("execution_failed", f"Path is a directory: {path}")
    try:
        raw = resolved.read_bytes()
    except PermissionError:
        return _err("permission_denied", "Cannot read file: permission denied")
    except OSError as exc:
        return _err("execution_failed", f"Cannot read file: {exc}")
    if b"\x00" in raw:
        return _err("execution_failed", "File is binary (contains NUL bytes)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _err("execution_failed", "File is not UTF-8 text")
    lines = text.splitlines()
    start, end = min(offset, len(lines)), len(lines)
    if limit is not None:
        end = min(start + limit, end)
    result_text = "\n".join(lines[start:end]) + ("\n" if lines[start:end] else "")
    compressed = compress_output(result_text, kind="generic")
    return ToolResult(tool="read_file", ok=True, content=compressed.text,
                     duration_ms=(time.perf_counter() - started) * 1000.0,
                     truncated=compressed.truncated, original_bytes=compressed.original_bytes)


def write_file(
    workspace: Path,
    path: str,
    content: str,
) -> ToolResult:
    """Write content to a file. Creates parent directories if needed."""
    started = time.perf_counter()
    try:
        resolved = default_resolver(Path(workspace), path)
    except ToolError as exc:
        return ToolResult(
            tool="write_file",
            ok=False,
            error_kind=exc.kind,
            content=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except (OSError, PermissionError) as exc:
        return ToolResult(
            tool="write_file",
            ok=False,
            error_kind=(
                "permission_denied"
                if isinstance(exc, PermissionError)
                else "execution_failed"
            ),
            content=f"Cannot write file: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    return ToolResult(
        tool="write_file",
        ok=True,
        content=f"Written {len(content.encode('utf-8'))} bytes to {path}",
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def list_files(
    workspace: Path,
    path: str = ".",
    depth: int = 3,
) -> ToolResult:
    """List files respecting .gitignore when present. Enforces depth limit."""
    started = time.perf_counter()
    try:
        resolved = default_resolver(Path(workspace), path)
    except ToolError as exc:
        return ToolResult(
            tool="list_files",
            ok=False,
            error_kind=exc.kind,
            content=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if not resolved.exists():
        return ToolResult(
            tool="list_files",
            ok=False,
            error_kind="not_found",
            content=f"Path not found: {path}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if not resolved.is_dir():
        return ToolResult(
            tool="list_files",
            ok=False,
            error_kind="execution_failed",
            content=f"Path is not a directory: {path}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    entries = _list_with_depth(resolved, max_depth=depth)
    result_text = "\n".join(sorted(entries))
    if result_text:
        result_text += "\n"

    return ToolResult(
        tool="list_files",
        ok=True,
        content=result_text,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def search_files(
    workspace: Path,
    pattern: str,
    path: str = ".",
    regex: bool = False,
) -> ToolResult:
    """Search for pattern in files using ripgrep or Python fallback."""
    started = time.perf_counter()
    try:
        resolved = default_resolver(Path(workspace), path)
    except ToolError as exc:
        return ToolResult(
            tool="search_files",
            ok=False,
            error_kind=exc.kind,
            content=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if not resolved.exists():
        return ToolResult(
            tool="search_files",
            ok=False,
            error_kind="not_found",
            content=f"Path not found: {path}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if rg_path := _find_ripgrep():
        result = _search_with_ripgrep(rg_path, resolved, pattern, regex)
    else:
        result = _search_python(resolved, pattern, regex)

    return ToolResult(
        tool="search_files",
        ok=True,
        content=result,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


# Directories that are build or tool artefacts. Listing them wastes context on
# output no agent decision ever depends on.
_SKIP_DIRS = frozenset({
    "__pycache__", "node_modules", ".venv", "venv", ".git", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".eggs",
})


def _list_with_depth(
    base: Path,
    current_depth: int = 0,
    max_depth: int = 3,
    root: Path | None = None,
) -> list[str]:
    """List entries relative to ``root``, recursing up to ``max_depth``.

    Paths are always expressed relative to the ORIGINAL root, not to the
    directory currently being walked — otherwise every nested entry collapses to
    a one-level name and the listing becomes useless for addressing files.
    """
    if current_depth > max_depth:
        return []
    anchor = base if root is None else root
    try:
        items = sorted(base.iterdir())
    except (OSError, PermissionError):
        return []

    entries: list[str] = []
    for item in items:
        if item.name.startswith(".") or item.name in _SKIP_DIRS:
            continue
        try:
            rel = item.relative_to(anchor)
        except ValueError:
            continue
        if item.is_dir():
            entries.append(f"{rel}/")
            if current_depth < max_depth:
                entries.extend(
                    _list_with_depth(item, current_depth + 1, max_depth, anchor)
                )
        else:
            entries.append(str(rel))
    return entries


def _find_ripgrep() -> Path | None:
    """Check if ripgrep is available on PATH."""
    try:
        result = subprocess.run(
            ["which", "rg"],
            capture_output=True,
            timeout=2,
            text=True,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _search_with_ripgrep(rg_path: Path, base: Path, pattern: str, regex: bool) -> str:
    """Search using ripgrep."""
    try:
        args = [str(rg_path), "--line-number", "--color", "never"]
        if not regex:
            args.append("-F")
        args.extend([pattern, str(base)])
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=10,
            text=True,
            cwd=str(base.parent),
        )
        return result.stdout or "(no matches)"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ripgrep error: {exc}"


def _search_python(base: Path, pattern: str, regex: bool) -> str:
    """Fallback Python search implementation."""
    try:
        regex_obj = re.compile(pattern) if regex else None
    except re.error as exc:
        return f"Invalid pattern: {exc}"
    matches = []
    for file_path in base.rglob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), 1):
                if (regex_obj and regex_obj.search(line)) or (not regex_obj and pattern in line):
                    matches.append(f"{file_path.relative_to(base)}:{line_no}:{line}")
        except (OSError, PermissionError):
            continue
    return "\n".join(matches) + ("\n" if matches else "") if matches else "(no matches)"
