"""Who holds a port, and is it the process we think it is.

The failure this exists to prevent is quiet: a start that appears to succeed
because something was already answering. Every number measured afterwards then
describes a process nobody chose, and nothing about the run looks wrong. Two
benchmark configurations would report identical results and no error — which
is why identical measurements across different runtime settings are worth
treating as a warning sign rather than a coincidence.

So "the port answers" is never enough. The question is which pid holds the
socket, and whether it is ours.
"""

from __future__ import annotations

import enum
import errno
import os
import socket
from pathlib import Path

import httpx
from pydantic import BaseModel

PROC_NET_TCP = ("/proc/net/tcp", "/proc/net/tcp6")
"""Read directly rather than shelling out to ``ss`` or ``lsof``: neither is
guaranteed present, both need parsing anyway, and this file is the source they
read themselves."""

TCP_LISTEN = "0A"
"""``st`` value for LISTEN in /proc/net/tcp, which is hex and unlabelled."""


class PortState(enum.StrEnum):
    """What is on a port, as far as can be determined without guessing."""

    FREE = "free"
    INFERENCE_SERVER = "inference_server"
    """Answers ``/health`` — attachable, if attaching was asked for."""
    FOREIGN_SERVICE = "foreign_service"
    """Something is listening and it is not an inference server."""
    UNKNOWN = "unknown"
    """Held, but this process cannot see by whom (another user's process)."""


class PortReport(BaseModel):
    """What holds a port, and enough identity to compare it against a pid."""

    port: int
    state: PortState
    pid: int | None = None
    process_name: str | None = None
    detail: str | None = None

    @property
    def free(self) -> bool:
        return self.state is PortState.FREE


class PortConflict(RuntimeError):
    """The port is not available for the start that was requested."""


class RuntimeIdentityMismatch(RuntimeError):
    """The endpoint on the port is not the process we started."""


def inspect_port(port: int, host: str = "127.0.0.1") -> PortReport:
    """Classify a port without touching HTTP.

    Cheap and synchronous: used before a start, when the only question is
    whether the port is free at all.
    """
    if _is_free(port, host):
        return PortReport(port=port, state=PortState.FREE)

    pid, name = _holder(port)
    return PortReport(
        port=port,
        state=PortState.UNKNOWN if pid is None else PortState.FOREIGN_SERVICE,
        pid=pid,
        process_name=name,
        detail=(
            f"port {port} is in use by pid {pid} ({name})"
            if pid
            else f"port {port} is in use by a process this user cannot see"
        ),
    )


async def inspect_port_async(
    port: int, host: str = "127.0.0.1", *, client: httpx.AsyncClient | None = None
) -> PortReport:
    """Classify a port and ask whoever holds it whether it serves models.

    The HTTP probe is what separates "our kind of server" from "a port": the
    decision to attach must not be inferred from a port number, because the
    port number is the one thing a stale process and a fresh one share.
    """
    report = inspect_port(port, host)
    if report.free:
        return report

    owned_client = client is None
    probe = client or httpx.AsyncClient()
    try:
        response = await probe.get(f"http://{host}:{port}/health", timeout=2.0)
        serving = response.status_code in (200, 503)
    except httpx.HTTPError:
        serving = False
    finally:
        if owned_client:
            await probe.aclose()

    if serving:
        return report.model_copy(
            update={
                "state": PortState.INFERENCE_SERVER,
                "detail": f"an inference server is already serving on port {port}",
            }
        )
    return report


async def verify_owner(
    port: int, pid: int, host: str = "127.0.0.1", *, client: httpx.AsyncClient | None = None
) -> PortReport:
    """Confirm ``pid`` is what holds ``port``, or raise.

    Called after a start: the process was launched and something answers, but
    those are two separate facts. An unheld port fails too — "cannot tell" is
    not "fine", and treating it as such is how an unverified start gets
    reported as a verified one.
    """
    report = await inspect_port_async(port, host, client=client)

    if report.free:
        raise RuntimeIdentityMismatch(
            f"nothing is listening on port {port}, so the process started for "
            f"it (pid {pid}) is not serving"
        )
    if report.pid is None:
        raise RuntimeIdentityMismatch(
            f"cannot determine which process holds port {port}; refusing to "
            f"assume it is pid {pid}"
        )
    if report.pid != pid and not _is_child_of(report.pid, pid):
        raise RuntimeIdentityMismatch(
            f"port {port} is held by pid {report.pid} "
            f"({report.process_name or 'unknown'}), not by the process started "
            f"here (pid {pid}); refusing to use a server this harness did not start"
        )
    return report


# -- internals -------------------------------------------------------------


def _is_free(port: int, host: str) -> bool:
    """Try to bind. The only answer the OS gives without ambiguity."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # No SO_REUSEADDR: the question is whether a *listener* is there, and
        # reuse would let the bind succeed alongside one in TIME_WAIT.
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                return False
            raise
        return True


def _holder(port: int) -> tuple[int | None, str | None]:
    """Find the pid listening on ``port`` by matching socket inodes.

    /proc gives the inode per listening socket and the inode per open file
    descriptor; joining them is the only way to get from a port to a pid
    without a helper binary. Processes owned by another user are invisible
    here, which is reported as unknown rather than guessed at.
    """
    inodes = _listening_inodes(port)
    if not inodes:
        return None, None

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return int(entry.name), _process_name(entry)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return None, None


def _listening_inodes(port: int) -> set[str]:
    wanted = f"{port:04X}"
    inodes: set[str] = set()
    for path in PROC_NET_TCP:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != TCP_LISTEN:
                continue
            if fields[1].rsplit(":", 1)[-1] == wanted:
                inodes.add(fields[9])
    return inodes


def _process_name(proc_entry: Path) -> str | None:
    try:
        return (proc_entry / "comm").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _is_child_of(pid: int, ancestor: int) -> bool:
    """Whether ``pid`` descends from ``ancestor``.

    llama-server does not fork, but a wrapper script that execs it would show
    the wrapper's pid here while the socket belongs to its child. Walking up
    the tree keeps that legitimate case from being reported as a stranger.
    """
    seen: set[int] = set()
    current = pid
    while current > 1 and current not in seen:
        seen.add(current)
        parent = _parent_of(current)
        if parent is None:
            return False
        if parent == ancestor:
            return True
        current = parent
    return False


def _parent_of(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # The comm field can contain spaces and parentheses; everything after the
    # final ')' is positional, and ppid is the second of those.
    _, _, rest = stat.rpartition(")")
    fields = rest.split()
    return int(fields[1]) if len(fields) > 1 else None


__all__ = [
    "PortConflict",
    "PortReport",
    "PortState",
    "RuntimeIdentityMismatch",
    "inspect_port",
    "inspect_port_async",
    "verify_owner",
]
