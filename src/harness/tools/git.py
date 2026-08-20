"""Read-only git operations: status, diff, log."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from harness.core import ToolResult
from harness.tools._security import default_resolver
from harness.tools.compression import compress_command_output, detect_kind


def git_status(workspace: Path) -> ToolResult:
    """Show git status in the workspace."""
    started = time.perf_counter()
    cwd = str(workspace)

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool="git_status",
            ok=False,
            error_kind="timeout",
            content="git status exceeded timeout",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return ToolResult(
            tool="git_status",
            ok=False,
            error_kind="execution_failed",
            content=f"Failed to run git status: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if result.returncode != 0:
        if "not a git repository" in result.stderr.lower():
            return ToolResult(
                tool="git_status",
                ok=False,
                error_kind="execution_failed",
                content="Not a git repository",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        return ToolResult(
            tool="git_status",
            ok=False,
            error_kind="execution_failed",
            content=result.stderr or f"git status failed with exit code {result.returncode}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    kind = detect_kind("git status")
    compressed = compress_command_output(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command="git status",
        kind=kind,
    )

    return ToolResult(
        tool="git_status",
        ok=True,
        content=compressed.text,
        exit_code=0,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        truncated=compressed.truncated,
        original_bytes=compressed.original_bytes,
    )


def git_diff(
    workspace: Path,
    path: str | None = None,
    revision: str | None = None,
) -> ToolResult:
    """Show git diff, optionally for a specific file or revision."""
    started = time.perf_counter()
    cwd = str(workspace)

    args = ["git", "diff", "--no-color"]
    if revision:
        args.append(revision)
    if path:
        try:
            resolved = default_resolver(Path(workspace), path)
            args.append(str(resolved.relative_to(workspace)))
        except Exception:
            args.append(path)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool="git_diff",
            ok=False,
            error_kind="timeout",
            content="git diff exceeded timeout",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return ToolResult(
            tool="git_diff",
            ok=False,
            error_kind="execution_failed",
            content=f"Failed to run git diff: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if result.returncode != 0:
        if "not a git repository" in result.stderr.lower():
            return ToolResult(
                tool="git_diff",
                ok=False,
                error_kind="execution_failed",
                content="Not a git repository",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        return ToolResult(
            tool="git_diff",
            ok=False,
            error_kind="execution_failed",
            content=result.stderr or f"git diff failed with exit code {result.returncode}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    kind = detect_kind("git diff")
    compressed = compress_command_output(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command="git diff",
        kind=kind,
    )

    return ToolResult(
        tool="git_diff",
        ok=True,
        content=compressed.text,
        exit_code=0,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        truncated=compressed.truncated,
        original_bytes=compressed.original_bytes,
    )


def git_log(
    workspace: Path,
    max_count: int = 20,
    follow: bool = False,
) -> ToolResult:
    """Show git log with optional max count and follow flag."""
    started = time.perf_counter()
    cwd = str(workspace)

    args = ["git", "log", "--no-color", "--oneline", f"--max-count={max_count}"]
    if follow:
        args.append("--follow")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool="git_log",
            ok=False,
            error_kind="timeout",
            content="git log exceeded timeout",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return ToolResult(
            tool="git_log",
            ok=False,
            error_kind="execution_failed",
            content=f"Failed to run git log: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if result.returncode != 0:
        if "not a git repository" in result.stderr.lower():
            return ToolResult(
                tool="git_log",
                ok=False,
                error_kind="execution_failed",
                content="Not a git repository",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        return ToolResult(
            tool="git_log",
            ok=False,
            error_kind="execution_failed",
            content=result.stderr or f"git log failed with exit code {result.returncode}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    kind = detect_kind("git log")
    compressed = compress_command_output(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command="git log",
        kind=kind,
    )

    return ToolResult(
        tool="git_log",
        ok=True,
        content=compressed.text,
        exit_code=0,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        truncated=compressed.truncated,
        original_bytes=compressed.original_bytes,
    )
