"""Regression tests for the shell sandbox boundary (issue #7).

The sandbox must fail closed: without bubblewrap an untrusted (CONFIRM)
command is never executed unsandboxed, and the default network mode
isolates the network namespace. An explicit, approved network mode is
separate and auditable on the ToolResult.
"""

from __future__ import annotations

import os
import shutil
import socket
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.core import NetworkMode
from harness.tools.shell import _sandbox_argv, run_command


def _no_bwrap(_cmd: str, /, *_a, **_k) -> str | None:
    """shutil.which override that pretends bubblewrap is absent."""
    return None


_BWRAP_AVAILABLE = shutil.which("bwrap") is not None
requires_bwrap = pytest.mark.skipif(
    not _BWRAP_AVAILABLE, reason="bubblewrap not installed"
)


def _approve(_command: str, _reason: str) -> bool:
    return True


# --- fail-closed: no sandbox, no untrusted execution -----------------------


def test_untrusted_command_is_denied_when_sandbox_missing(tmp_path: Path) -> None:
    with patch("harness.tools.shell.shutil.which", _no_bwrap):
        result = run_command(
            tmp_path,
            "sh -c 'cat /etc/passwd'",
            confirm_callback=_approve,
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


def test_classifier_denied_command_records_no_network(tmp_path: Path) -> None:
    # /etc/passwd is denied by the classifier before the sandbox is involved;
    # a command that never executes records no network policy.
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
    assert result.network == NetworkMode.ISOLATED


def test_network_allowed_without_approval_downgrades_to_isolated(
    tmp_path: Path,
) -> None:
    # ALLOWED is the explicit opt-out, but network access is a separate
    # privilege: without an approval mechanism it is never granted — the
    # command runs isolated instead.
    result = run_command(tmp_path, "ls", network=NetworkMode.ALLOWED)
    assert result.network == NetworkMode.ISOLATED


def test_network_allowed_with_approval_is_audited(tmp_path: Path) -> None:
    result = run_command(
        tmp_path, "ls", network=NetworkMode.ALLOWED, confirm_callback=_approve,
    )
    assert result.network == NetworkMode.ALLOWED


def test_network_allowed_approval_denied_downgrades_to_isolated(
    tmp_path: Path,
) -> None:
    result = run_command(
        tmp_path,
        "ls",
        network=NetworkMode.ALLOWED,
        confirm_callback=lambda _c, _r: False,
    )
    assert result.network == NetworkMode.ISOLATED


# --- isolation regression: requires bubblewrap ----------------------------


@requires_bwrap
def test_home_is_isolated_to_tmp_home(tmp_path: Path) -> None:
    result = run_command(tmp_path, "printenv HOME", confirm_callback=_approve)
    assert result.ok is True
    assert "/tmp/home" in result.content
    assert str(Path.home()) not in result.content


@requires_bwrap
def test_system_paths_are_read_only(tmp_path: Path) -> None:
    # /usr is ro-bound: an approved write inside it must fail. Proves system
    # paths are mounted read-only, not writable from the sandbox.
    result = run_command(
        tmp_path, "touch /usr/bin/_harness_sandbox_marker", confirm_callback=_approve,
    )
    assert result.ok is False
    assert result.exit_code != 0


@requires_bwrap
def test_symlink_escape_to_etc_is_unreachable(tmp_path: Path) -> None:
    # Defense in depth: a workspace symlink pointing at /etc/passwd is denied
    # by the classifier (which resolves it), and would also be blocked by the
    # sandbox — /etc/passwd is not mounted. Either way it is unreachable.
    (tmp_path / "escape").symlink_to("/etc/passwd")
    result = run_command(tmp_path, "cat escape", confirm_callback=_approve)
    assert result.ok is False


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
            tmp_path, cmd, confirm_callback=_approve, network=NetworkMode.ISOLATED,
        )
        allowed = run_command(
            tmp_path, cmd, confirm_callback=_approve, network=NetworkMode.ALLOWED,
        )
        assert isolated.ok is False
        assert allowed.ok is True
    finally:
        stop.set()
        server.close()
        thread.join(timeout=2)

# --- boundary coercion: network mode from decoded tool arguments ----------


def test_string_isolated_still_unshares_net(tmp_path: Path) -> None:
    # NetworkMode is a StrEnum, so a decoded JSON argument arrives as a plain
    # ``str``. An identity check would miss it and silently omit
    # ``--unshare-net`` while the result still claimed "isolated".
    argv = _sandbox_argv(tmp_path, "ls", network="isolated")
    assert argv is not None
    assert "--unshare-net" in argv


def test_string_isolated_is_audited_as_enum(tmp_path: Path) -> None:
    result = run_command(tmp_path, "ls", network="isolated")
    assert result.network is NetworkMode.ISOLATED


def test_string_allowed_without_approval_downgrades_to_isolated(
    tmp_path: Path,
) -> None:
    # The approval gate must not be bypassable by passing the opt-out as a
    # string instead of the enum member.
    result = run_command(tmp_path, "ls", network="allowed")
    assert result.network is NetworkMode.ISOLATED


def test_string_allowed_with_approval_is_audited(tmp_path: Path) -> None:
    result = run_command(
        tmp_path, "ls", network="allowed", confirm_callback=_approve,
    )
    assert result.network is NetworkMode.ALLOWED


@pytest.mark.parametrize("mode", ["host", "none", "ISOLATED", "", "1"])
def test_unknown_network_mode_is_denied(tmp_path: Path, mode: str) -> None:
    # Reject rather than guess: an unrecognized policy never falls through to
    # host-network access.
    result = run_command(tmp_path, "ls", network=mode, confirm_callback=_approve)
    assert result.ok is False
    assert result.error_kind == "denied"
    assert "network" in result.content.lower()


# --- timeouts stay auditable ---------------------------------------------


def test_timeout_records_unsandboxed_network_policy(tmp_path: Path) -> None:
    # ``cat`` on a FIFO with no writer is trusted and blocks, so it times out
    # on the documented unsandboxed path. The command already ran; the result
    # must still say under which policy.
    os.mkfifo(tmp_path / "pipe")
    with patch("harness.tools.shell.shutil.which", _no_bwrap):
        result = run_command(tmp_path, "cat pipe", timeout=0.3)
    assert result.error_kind == "timeout"
    assert result.network == "unsandboxed"


@requires_bwrap
def test_timeout_records_isolated_network_policy(tmp_path: Path) -> None:
    result = run_command(
        tmp_path, "sleep 5", timeout=0.3, confirm_callback=_approve,
    )
    assert result.error_kind == "timeout"
    assert result.network is NetworkMode.ISOLATED
