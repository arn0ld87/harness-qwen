"""Shell command execution with security classification and timeout."""

from __future__ import annotations

import enum
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from harness.core import Risk, ToolResult
from harness.tools._security import default_classifier
from harness.tools.compression import compress_command_output, detect_kind

ConfirmCallback = Callable[[str, str], bool]
"""Maps (command, risk_reason) to approval: True to run, False to deny."""


class NetworkMode(enum.StrEnum):
    """Network policy a shell command runs under.

    ``ISOLATED`` is the default: the sandbox gets its own network namespace
    (``--unshare-net``) so an approved command still cannot reach the host
    network. ``ALLOWED`` is the explicit, auditable opt-out for commands a
    human approved knowing they need network access — it runs in the sandbox
    but shares the host network namespace.
    """

    ISOLATED = "isolated"
    ALLOWED = "allowed"


def run_command(
    workspace: Path,
    command: str,
    timeout: float = 30.0,
    confirm_callback: ConfirmCallback | None = None,
    network: NetworkMode = NetworkMode.ISOLATED,
) -> ToolResult:
    """Execute a shell command with security classification and timeout.

    Risk.DENY returns denied result without executing. Risk.CONFIRM consults
    the approval callback; when no callback was provided, DENY. Risk.ALLOW
    executes immediately.

    The sandbox is fail-closed: an untrusted (CONFIRM) command is never
    executed without bubblewrap. A trusted (ALLOW) read-only command is the
    one documented exception — it may run unsandboxed when bwrap is absent,
    and the result records ``network="unsandboxed"`` so that gap is auditable
    rather than silent. The default ``network`` mode isolates the network
    namespace; ``NetworkMode.ALLOWED`` is the explicit, auditable opt-out.
    """
    started = time.perf_counter()
    workspace = workspace.resolve()
    cwd = str(workspace)

    risk = default_classifier(command, workspace)

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
                content=(
                    "Command requires confirmation but no approval mechanism "
                    f"is available: {command}"
                ),
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

    sandbox = _sandbox_argv(workspace, command, network=network)
    if sandbox is None:
        # Fail closed: an untrusted command — even one a human approved — is
        # never executed without the sandbox. There is no fallback for it.
        if risk is Risk.CONFIRM:
            return ToolResult(
                tool="run_command",
                ok=False,
                error_kind="denied",
                content=(
                    "sandbox unavailable: bubblewrap (bwrap) is required to "
                    f"execute an untrusted command and was not found: {command}"
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        # Trusted read-only commands are the documented unsandboxed fallback.
        argv: list[str] = ["/bin/sh", "-c", command]
        network_mode = "unsandboxed"
    else:
        argv = sandbox
        network_mode = str(network)

    try:
        proc = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env={"HOME": "/tmp/home", "LANG": "C.UTF-8", "PATH": _sandbox_path(workspace)},
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
        network=network_mode,
    )


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill a process group on timeout."""
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def _sandbox_path(workspace: Path) -> str:
    entries = [workspace / ".venv" / "bin", Path("/usr/bin"), Path("/bin")]
    uv = shutil.which("uv")
    if uv:
        entries.insert(1, Path(uv).parent)
    return ":".join(str(entry) for entry in entries)


def _sandbox_argv(
    workspace: Path, command: str, *, network: NetworkMode = NetworkMode.ISOLATED
) -> list[str] | None:
    """Wrap a command in bubblewrap, or return ``None`` when bwrap is absent.

    Returning ``None`` (rather than a bare ``/bin/sh -c`` fallback) makes the
    absence fail-closed at the caller: :func:`run_command` decides whether
    the command's risk class may run unsandboxed, never this function. The
    default ``network`` mode adds ``--unshare-net`` so an approved command
    still gets an isolated network namespace; ``NetworkMode.ALLOWED`` omits
    it so the opt-out is a visible, auditable difference in the argv.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None

    argv = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if network is NetworkMode.ISOLATED:
        argv.append("--unshare-net")
    argv.extend((
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/bin",
        "/sbin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
    ))
    for source in (Path("/etc/ld.so.cache"), Path("/etc/localtime"), Path("/etc/ssl")):
        if source.exists():
            _append_parent_dirs(argv, source.parent)
            argv.extend(("--ro-bind", str(source), str(source)))

    toolchain_paths = [Path(sys.base_prefix)]
    uv = shutil.which("uv")
    if uv:
        toolchain_paths.append(Path(uv))
    for source in toolchain_paths:
        resolved = source.resolve()
        if resolved.is_relative_to("/usr") or resolved.is_relative_to(workspace):
            continue
        _append_parent_dirs(argv, resolved.parent)
        argv.extend(("--ro-bind", str(resolved), str(resolved)))

    _append_parent_dirs(argv, workspace.parent)
    argv.extend(
        (
            "--bind",
            str(workspace),
            str(workspace),
            "--chdir",
            str(workspace),
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp/home",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PATH",
            _sandbox_path(workspace),
            "/bin/sh",
            "-c",
            command,
        )
    )
    return argv


def _append_parent_dirs(argv: list[str], path: Path) -> None:
    parents = list(reversed(path.parents[:-1])) + [path]
    for parent in parents:
        if parent != Path("/"):
            argv.extend(("--dir", str(parent)))
