"""Shell command execution with security classification and timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.core import Risk, ToolError, ToolResult
from harness.tools.compression import compress_command_output, detect_kind
from harness.tools._security import default_classifier


ConfirmCallback = Callable[[str, str], bool]
"""Maps (command, risk_reason) to approval: True to run, False to deny."""


def run_command(
    workspace: Path,
    command: str,
    timeout: float = 30.0,
    confirm_callback: ConfirmCallback | None = None,
) -> ToolResult:
    """Execute a shell command with security classification and timeout.

    Risk.DENY returns denied result without executing. Risk.CONFIRM consults
    the approval callback; when no callback was provided, DENY. Risk.ALLOW
    executes immediately.
    """
    started = time.perf_counter()
    cwd = str(workspace)

    risk = default_classifier(command)

    if risk is Risk.DENY:
        return ToolResult(
            tool="run_command",
            ok=False,
            error_kind="denied",
            content=f"Command is denied: {command}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if risk is Risk.CONFIRM:
        if confirm_callback is None:
            return ToolResult(
                tool="run_command",
                ok=False,
                error_kind="denied",
                content=f"Command requires confirmation but no approval mechanism is available: {command}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        approved = confirm_callback(command, "This command requires approval")
        if not approved:
            return ToolResult(
                tool="run_command",
                ok=False,
                error_kind="denied",
                content=f"Command was not approved: {command}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception as exc:
        return ToolResult(
            tool="run_command",
            ok=False,
            error_kind="execution_failed",
            content=f"Failed to start command: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return ToolResult(
            tool="run_command",
            ok=False,
            error_kind="timeout",
            content=f"Command exceeded timeout of {timeout}s",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    kind = detect_kind(command)
    compressed = compress_command_output(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        command=command,
        kind=kind,
    )

    return ToolResult(
        tool="run_command",
        ok=exit_code == 0,
        content=compressed.text,
        exit_code=exit_code,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        truncated=compressed.truncated,
        original_bytes=compressed.original_bytes,
    )


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill a process group on timeout."""
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
