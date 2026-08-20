"""Start, watch and stop one local ``llama-server``.

The expensive fact this is built around: loading the model takes tens of
seconds, and a process that is loading looks identical to one that is broken
if you only test whether the port answers. So the wait distinguishes three
outcomes — still loading, dead, and out of time — because each calls for a
different response and only one of them is worth waiting longer for.

The other fact is ownership. A server this process started can be stopped; one
that was already running belongs to somebody else, and stopping it destroys a
model load that costs minutes to rebuild.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx

from harness.config.schema import HarnessConfig
from harness.core import HealthStatus
from harness.runtime.argv import build_argv
from harness.runtime.handle import (
    Ownership,
    RuntimeCrashed,
    RuntimeHandle,
    RuntimeNotOwned,
    RuntimeStartTimeout,
)
from harness.runtime.port import (
    PortConflict,
    PortState,
    inspect_port_async,
    verify_owner,
)
from harness.telemetry.redact import redact

POLL_INTERVAL_S = 0.2
"""How often health is retried while waiting. Short enough that a fast start
is not padded, long enough not to spin against a loading server."""

TAIL_LINES = 50
"""Output kept in memory for a crash report. The full log is on disk; this is
what gets quoted in the exception, and a wall of text there helps nobody."""

GRACE_S = 5.0
"""Time a server gets to exit on SIGTERM before SIGKILL. Long enough for it to
release the port and flush, short enough not to hang a CLI."""


class LlamaServerSupervisor:
    """Owns at most one server process, or attaches to one it does not own."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        log_dir: Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.log_dir = log_dir or Path(".harness/logs")
        self._client = client
        self._owns_client = client is None
        self._process: subprocess.Popen[str] | None = None
        self._handle: RuntimeHandle | None = None
        self._tail: deque[str] = deque(maxlen=TAIL_LINES)
        self._reader: threading.Thread | None = None
        self._log_file: Path | None = None

    @property
    def handle(self) -> RuntimeHandle | None:
        return self._handle

    @property
    def running(self) -> bool:
        """Whether the process we started is still alive.

        Only meaningful for an owned server: for an attached one the answer
        lives in :meth:`health`, since we hold no process handle at all.
        """
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> str:
        return self.config.runtime.base_url

    # -- lifecycle ---------------------------------------------------------

    async def ensure(self) -> RuntimeHandle:
        """Attach or start, whichever the configuration asks for."""
        if self.config.runtime.attach:
            return await self.attach()
        return await self.start()

    async def start(self, *, timeout_s: float | None = None) -> RuntimeHandle:
        """Launch the server and wait until it reports healthy.

        Cleans up after itself on every failure path: a half-started server
        left holding the port is what makes the *next* start fail too, with a
        less obvious message (#11).
        """
        deadline = timeout_s or self.config.runtime.startup_timeout_s
        await self._require_free_port()
        argv = build_argv(self.config)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / f"llama-server-{self.config.runtime.port}.log"

        started_at = datetime.now(UTC)
        self._process = subprocess.Popen(  # noqa: S603 (argv is built, not a shell string)
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Its own process group, so stopping it never signals this process
            # or a sibling that happens to share ours.
            start_new_session=True,
        )
        self._start_reader(self._log_file)

        try:
            await self._wait_for_health(deadline)
        except Exception:
            await self.stop()
            raise

        pid = self._process.pid
        try:
            # Something answering and our process serving are two facts, and
            # only the second one makes the numbers that follow meaningful.
            await verify_owner(
                self.config.runtime.port, pid, self.config.runtime.host,
                client=self._ensure_client(),
            )
        except Exception:
            await self.stop()
            raise

        self._handle = RuntimeHandle(
            base_url=self.base_url,
            ownership=Ownership.OWNED,
            pid=pid,
            started_at=started_at,
            log_path=self._log_file,
        )
        return self._handle

    async def attach(self, *, timeout_s: float = 10.0) -> RuntimeHandle:
        """Use a server this harness did not start.

        Waits rather than checking once, so attaching to a server that is
        still loading works — but never starts anything, and never claims a
        process it cannot see.
        """
        report = await inspect_port_async(
            self.config.runtime.port, self.config.runtime.host,
            client=self._ensure_client(),
        )
        if not report.free and report.state is not PortState.INFERENCE_SERVER:
            raise PortConflict(
                f"cannot attach to port {self.config.runtime.port}: "
                f"{report.detail or 'it is held by something that does not serve models'}"
            )
        await self._wait_for_health(timeout_s)
        self._handle = RuntimeHandle(
            base_url=self.base_url, ownership=Ownership.ATTACHED
        )
        return self._handle

    async def stop(self, *, grace_s: float = GRACE_S) -> None:
        """Stop an owned server. Idempotent; refuses anything else.

        SIGTERM to the process group first — llama-server releases the port
        and flushes on it — then SIGKILL once the grace period is up. Killing
        first would leave the port in TIME_WAIT more often, which turns a
        clean restart into a port conflict.
        """
        if self._handle is not None and not self._handle.owned:
            raise RuntimeNotOwned(
                f"{self.base_url} was attached, not started here; "
                "stopping it would take down a server this harness does not own"
            )

        process = self._process
        if process is None:
            self._handle = None
            return

        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            if not await self._await_exit(process, grace_s):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                await self._await_exit(process, grace_s)

        if process.stdout is not None:
            with contextlib.suppress(Exception):
                process.stdout.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None

        self._process = None
        self._handle = None

    async def _require_free_port(self) -> None:
        """Refuse to start against a port somebody else is already using.

        Without this the launched process loses the bind, exits, and the
        health check succeeds anyway — against the old server. The start looks
        clean and every measurement after it is somebody else's.
        """
        port = self.config.runtime.port
        report = await inspect_port_async(
            port, self.config.runtime.host, client=self._ensure_client()
        )
        if report.free:
            return

        hint = (
            " Set runtime.attach to use it deliberately."
            if report.state is PortState.INFERENCE_SERVER
            else ""
        )
        raise PortConflict(
            f"cannot start on port {port}: {report.detail or 'it is in use'}.{hint}"
        )

    # -- health ------------------------------------------------------------

    async def health(self) -> HealthStatus:
        """One health probe, classified.

        A dead owned process is reported as unreachable *with its exit status*
        rather than as a connection error: "connection refused" sends someone
        to the network, and the answer is in the log.
        """
        process = self._process
        if process is not None and process.poll() is not None:
            return HealthStatus(
                reachable=False,
                detail=(
                    f"server exited with status {process.returncode}; "
                    f"last output: {self._tail_text() or '(none)'}"
                ),
            )

        client = self._ensure_client()
        try:
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
        except httpx.HTTPError as exc:
            return HealthStatus(reachable=False, detail=f"cannot reach {self.base_url}: {exc}")

        if response.status_code == 200:
            return HealthStatus(reachable=True)
        if response.status_code == 503:
            # llama-server's own signal for "weights are still loading".
            return HealthStatus(reachable=False, loading=True, detail="loading model")
        return HealthStatus(
            reachable=False, detail=f"health returned HTTP {response.status_code}"
        )

    async def close(self) -> None:
        """Release the HTTP client if this supervisor created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- internals ---------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        return self._client

    async def _wait_for_health(self, timeout_s: float) -> None:
        """Poll until healthy, the process dies, or the deadline passes."""
        deadline = time.monotonic() + timeout_s
        last_detail = "no response yet"
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise RuntimeCrashed(
                    f"server exited with status {process.returncode} during "
                    f"startup; last output: {self._tail_text() or '(none)'}"
                )
            status = await self.health()
            if status.reachable:
                return
            last_detail = status.detail or last_detail
            await asyncio.sleep(POLL_INTERVAL_S)

        raise RuntimeStartTimeout(
            f"{self.base_url} did not become healthy within {timeout_s:g}s "
            f"({last_detail})"
        )

    def _start_reader(self, log_path: Path) -> None:
        """Drain the server's output into a redacted log file.

        A thread rather than an async task: the pipe read is blocking, and
        leaving it unread fills the OS buffer and stalls the server — which
        would look like a hang in the model, not in the plumbing.
        """
        process = self._process
        if process is None or process.stdout is None:
            return

        stream = process.stdout

        def pump() -> None:
            with log_path.open("a", encoding="utf-8") as sink:
                for line in stream:
                    # Redacted before it reaches disk, not after: the server
                    # may echo an api-key it was started with, and a log is
                    # exactly the file that gets pasted into a bug report.
                    safe = redact(line.rstrip("\n"))
                    self._tail.append(safe)
                    sink.write(f"{safe}\n")
                    sink.flush()

        self._reader = threading.Thread(target=pump, daemon=True)
        self._reader.start()

    def _tail_text(self) -> str:
        return " | ".join(list(self._tail)[-5:])

    @staticmethod
    async def _await_exit(process: subprocess.Popen[str], timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return True
            await asyncio.sleep(0.05)
        return process.poll() is not None


__all__ = ["LlamaServerSupervisor"]
