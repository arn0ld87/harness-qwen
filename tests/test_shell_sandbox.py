"""Regression tests for the shell sandbox boundary (issue #7).

The sandbox must fail closed: without bubblewrap an untrusted (CONFIRM)
command is never executed unsandboxed, and the default network mode
isolates the network namespace. An explicit, approved network mode is
separate and auditable on the ToolResult.
"""

from __future__ import annotations

import shutil
import socket
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.tools.shell import NetworkMode, _sandbox_argv, run_command


def _no_bwrap(_cmd: str, /, *_a, **_k) -> str | None:
    """shutil.which override that pretends bubblewrap is absent."""
    return None


_BWRAP_AVAILABLE = shutil.which("bwrap") is not None
requires_bwrap = pytest.mark.skipif(
    not _BWRAP_AVAILABLE, reason="bubblewrap not installed"
)


# --- fail-closed: no sandbox, no untrusted execution -----------------------


def test_untrusted_command_is_denied_when_sandbox_missing(tmp_path: Path) -> None:
    with patch("harness.tools.shell.shutil.which", _no_bwrap):
        result = run_command(
            tmp_path,
            "sh -c 'cat /etc/passwd'",
            confirm_callback=lambda _c, _r: True,
        )
    assert result.ok is False
    assert result.error_kind == "denied"
    assert "sandbox" in result.content.lower()


def test_trusted_command_runs_unsandboxed_when_bwrap_missing(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("visible\n", encoding="utf-8")
    with patch("harness.tools.shell.shutil.which", _no_bwrap):
        result = run_command(tmp_path, "cat file.txt")
    assert result.ok is True
    assert "visible" in result.content
    # Read-only trusted commands are the documented unsandboxed exception;
    # the result records that no isolation was applied, not that it was.
    assert result.network == "unsandboxed"


def test_denied_command_records_no_network(tmp_path: Path) -> None:
    result = run_command(tmp_path, "cat /etc/passwd")
    assert result.ok is False
    assert result.network is None


# --- network mode: argv + audit -------------------------------------------


def test_sandbox_argv_returns_none_when_bwrap_missing(tmp_path: Path) -> None:
    with patch("harness.tools.shell.shutil.which", _no_bwrap):
        assert _sandbox_argv(tmp_path, "ls", network=NetworkMode.ISOLATED) is None


def test_default_network_is_isolated_in_argv(tmp_path: Path) -> None:
    argv = _sandbox_argv(tmp_path, "ls", network=NetworkMode.ISOLATED)
    assert argv is not None
    assert "--unshare-net" in argv


def test_explicit_network_allowed_omits_unshare_net(tmp_path: Path) -> None:
    argv = _sandbox_argv(tmp_path, "ls", network=NetworkMode.ALLOWED)
    assert argv is not None
    assert "--unshare-net" not in argv


def test_default_network_mode_is_audited_on_result(tmp_path: Path) -> None:
    result = run_command(tmp_path, "ls")
    assert result.network == "isolated"


def test_explicit_network_mode_is_audited_on_result(tmp_path: Path) -> None:
    result = run_command(tmp_path, "ls", network=NetworkMode.ALLOWED)
    assert result.network == "allowed"


# --- isolation regression: requires bubblewrap ----------------------------


@requires_bwrap
def test_home_is_isolated_to_tmp_home(tmp_path: Path) -> None:
    result = run_command(
        tmp_path, "printenv HOME",
        confirm_callback=lambda _c, _r: True,
    )
    assert result.ok is True
    assert "/tmp/home" in result.content
    assert str(Path.home()) not in result.content


@requires_bwrap
def test_untrusted_command_cannot_reach_host_network(tmp_path: Path) -> None:
    """An approved command in the isolated namespace cannot reach a listener
    the host process opened; the same command in the allowed namespace can."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        try:
            while not stop.is_set():
                conn, _ = server.accept()
                conn.sendall(b"ok")
                conn.close()
        except (TimeoutError, OSError):
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        cmd = (
            "python3 -c 'import socket; s=socket.create_connection"
            f'(("127.0.0.1",{port}),timeout=3); print(s.recv(2).decode()); s.close()\''
        )
        isolated = run_command(
            tmp_path, cmd,
            confirm_callback=lambda _c, _r: True,
            network=NetworkMode.ISOLATED,
        )
        allowed = run_command(
            tmp_path, cmd,
            confirm_callback=lambda _c, _r: True,
            network=NetworkMode.ALLOWED,
        )
        assert isolated.ok is False
        assert allowed.ok is True
    finally:
        stop.set()
        server.close()
        thread.join(timeout=2)