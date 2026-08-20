"""Pre-flight checks: does this host actually match what the run assumes.

Everything here answers one question — would a real run fail, and why — before
25 s of model loading and a few minutes of agent time are spent finding out.
Each check is a small function returning one :class:`Check`, so a host that
cannot answer a question reports ``SKIP`` instead of taking the whole
diagnosis down with it.

Nothing here changes the system. A diagnostic that fixes things quietly is a
diagnostic nobody can trust the output of.
"""

from __future__ import annotations

import enum
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from harness.config.resolve import ResolvedConfig
from harness.discovery.hardware import probe_sandbox
from harness.memory.migrations import SCHEMA_VERSION
from harness.runtime.port import PortState, inspect_port_async

MIN_PYTHON = (3, 12)
GIB = 1024 ** 3
LOW_DISK_BYTES = 2 * GIB
"""Below this, a run that writes a journal and a database is a bad bet."""


class Severity(enum.StrEnum):
    """How much a finding should stop someone.

    ``FAIL`` means a real run would not work. ``WARN`` means it would work and
    the result would be worth less than it looks — a distinction that matters
    on a machine where a misleading benchmark costs more than a crash.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Check(BaseModel):
    """One question asked of the host, and what it answered."""

    name: str
    severity: Severity
    detail: str
    hint: str | None = None
    """What to do about it. Absent when there is nothing to do."""


class Diagnosis(BaseModel):
    """Every check, and the single number a script can act on."""

    checks: list[Check] = Field(default_factory=list)

    def by_severity(self, severity: Severity) -> list[Check]:
        return [check for check in self.checks if check.severity is severity]

    @property
    def failed(self) -> bool:
        return bool(self.by_severity(Severity.FAIL))

    @property
    def warned(self) -> bool:
        return bool(self.by_severity(Severity.WARN))

    def exit_code(self, *, strict: bool = False) -> int:
        """0 usable, 1 broken, 2 usable-but-questionable under ``--strict``.

        Warnings do not fail by default: most of them describe a host that
        works and would measure badly, and refusing to run at all would be a
        worse answer than saying so.
        """
        if self.failed:
            return 1
        if strict and self.warned:
            return 2
        return 0


CheckFn = Callable[[ResolvedConfig], Check | None]
AsyncCheckFn = Callable[[ResolvedConfig], Awaitable[Check | None]]


# -- host ------------------------------------------------------------------


def check_python(_config: ResolvedConfig) -> Check:
    version = sys.version_info
    if version[:2] >= MIN_PYTHON:
        return Check(
            name="python",
            severity=Severity.OK,
            detail=f"{version.major}.{version.minor}.{version.micro}",
        )
    return Check(
        name="python",
        severity=Severity.FAIL,
        detail=f"{version.major}.{version.minor} is below the required "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        hint="Run through uv, which pins the interpreter for this project.",
    )


def check_uv(_config: ResolvedConfig) -> Check:
    path = shutil.which("uv")
    if path:
        return Check(name="uv", severity=Severity.OK, detail=path)
    return Check(
        name="uv",
        severity=Severity.WARN,
        detail="uv not found on PATH",
        hint="Not needed at runtime, but every documented command assumes it.",
    )


def check_workspace(config: ResolvedConfig) -> Check:
    """Writable workspace, checked by writing — not by reading permissions.

    Permission bits lie on read-only mounts, full filesystems and stale NFS
    handles, and each of those fails later at a point that looks like a bug in
    the agent.
    """
    workspace = config.config.workspace
    if not workspace.exists():
        return Check(
            name="workspace",
            severity=Severity.FAIL,
            detail=f"{workspace} does not exist",
            hint="Create it, or point workspace at the directory to work in.",
        )
    try:
        with tempfile.NamedTemporaryFile(dir=workspace, prefix=".harness-probe-"):
            pass
    except OSError as exc:
        return Check(
            name="workspace",
            severity=Severity.FAIL,
            detail=f"{workspace} is not writable: {exc}",
        )
    return Check(name="workspace", severity=Severity.OK, detail=str(workspace))


def check_disk_space(config: ResolvedConfig) -> Check:
    workspace = config.config.workspace
    if not workspace.exists():
        return Check(
            name="disk", severity=Severity.SKIP, detail="workspace does not exist"
        )
    usage = shutil.disk_usage(workspace)
    free_gib = usage.free / GIB
    if usage.free < LOW_DISK_BYTES:
        return Check(
            name="disk",
            severity=Severity.WARN,
            detail=f"{free_gib:.1f} GiB free on {workspace}",
            hint="A run writes a journal, a database and captured tool output.",
        )
    return Check(name="disk", severity=Severity.OK, detail=f"{free_gib:.1f} GiB free")


def check_database(config: ResolvedConfig) -> Check:
    """Open the store the way a run will, and report the schema it finds."""
    database = config.config.database
    if not database.is_absolute():
        database = config.config.workspace / database
    if not database.exists():
        return Check(
            name="database",
            severity=Severity.OK,
            detail=f"{database} will be created on first run",
        )
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return Check(
            name="database", severity=Severity.FAIL, detail=f"cannot open {database}: {exc}"
        )
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error as exc:
        return Check(
            name="database", severity=Severity.FAIL, detail=f"{database} is unreadable: {exc}"
        )
    finally:
        connection.close()

    if version > SCHEMA_VERSION:
        return Check(
            name="database",
            severity=Severity.FAIL,
            detail=f"{database} is at schema {version}, newer than the supported "
            f"{SCHEMA_VERSION}",
            hint="A newer harness wrote it; reading it would corrupt the resume point.",
        )
    if version < SCHEMA_VERSION:
        return Check(
            name="database",
            severity=Severity.OK,
            detail=f"schema {version} will be migrated to {SCHEMA_VERSION}",
        )
    return Check(name="database", severity=Severity.OK, detail=f"schema {version}")


def check_sandbox(_config: ResolvedConfig) -> Check:
    """Bubblewrap, and separately whether it may actually unshare namespaces.

    Installed-but-forbidden is the case worth naming on its own: the binary is
    there, the host policy denies unprivileged user namespaces, and the
    sandbox is as absent as if nothing were installed — while every "is bwrap
    present" check says yes. Ubuntu ships that configuration by default.
    """
    info = probe_sandbox()
    if not info.available:
        return Check(
            name="sandbox",
            severity=Severity.FAIL,
            detail="bubblewrap (bwrap) not found",
            hint="The shell sandbox is fail-closed: untrusted commands are "
            "denied rather than run unsandboxed. Install bwrap.",
        )

    usable, reason = probe_namespaces(info.bwrap_path or "bwrap")
    if usable:
        return Check(
            name="sandbox",
            severity=Severity.OK,
            detail=f"bubblewrap at {info.bwrap_path}, namespaces usable",
        )
    return Check(
        name="sandbox",
        severity=Severity.FAIL,
        detail=f"bubblewrap at {info.bwrap_path} cannot unshare namespaces: {reason}",
        hint="Unprivileged user namespaces are restricted on this host "
        "(kernel.apparmor_restrict_unprivileged_userns or "
        "kernel.unprivileged_userns_clone). Untrusted commands stay denied "
        "until this works.",
    )


def probe_namespaces(bwrap: str) -> tuple[bool, str]:
    """Run the smallest possible sandboxed command, and report why not.

    Actually executing is the only honest test: the permission depends on
    kernel sysctls, AppArmor policy, seccomp and container settings, and no
    combination of those can be read reliably from userspace.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [bwrap, "--unshare-all", "--ro-bind", "/", "/", "/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, ""
    detail = result.stderr.decode(errors="replace").strip().splitlines()
    return False, (detail[-1] if detail else f"exit status {result.returncode}")


# -- runtime ---------------------------------------------------------------


def check_server_binary(config: ResolvedConfig) -> Check:
    runtime = config.config.runtime
    if runtime.attach:
        return Check(
            name="server-binary",
            severity=Severity.SKIP,
            detail="attach mode: no binary is launched",
        )
    if runtime.server_binary is None:
        return Check(
            name="server-binary",
            severity=Severity.FAIL,
            detail="runtime.server_binary is not set and attach is off",
            hint="Set the llama-server path, or enable runtime.attach for a "
            "server that is already running.",
        )
    binary = Path(runtime.server_binary)
    if not binary.exists():
        return Check(
            name="server-binary",
            severity=Severity.FAIL,
            detail=f"{binary} does not exist",
        )
    if not os.access(binary, os.X_OK):
        return Check(
            name="server-binary",
            severity=Severity.FAIL,
            detail=f"{binary} is not executable",
        )
    return Check(name="server-binary", severity=Severity.OK, detail=str(binary))


def check_model_path(config: ResolvedConfig) -> Check:
    model = config.config.model
    if model.path is None:
        return Check(
            name="model",
            severity=Severity.SKIP,
            detail="no model path configured (attach mode uses whatever is loaded)",
        )
    if not model.path.exists():
        return Check(
            name="model", severity=Severity.FAIL, detail=f"{model.path} does not exist"
        )
    size_gib = model.path.stat().st_size / GIB
    return Check(
        name="model", severity=Severity.OK, detail=f"{model.path} ({size_gib:.1f} GiB)"
    )


async def check_port(config: ResolvedConfig) -> Check:
    """What is on the configured port, and whether that is what we want.

    An occupied port is a failure for a start and a precondition for an
    attach, so the same fact reads two ways depending on configuration —
    which is exactly why it is reported rather than acted on.
    """
    runtime = config.config.runtime
    report = await inspect_port_async(runtime.port, runtime.host)

    if report.free:
        if runtime.attach:
            return Check(
                name="port",
                severity=Severity.FAIL,
                detail=f"attach is configured but nothing is listening on port {runtime.port}",
            )
        return Check(
            name="port", severity=Severity.OK, detail=f"port {runtime.port} is free"
        )

    if report.state is PortState.INFERENCE_SERVER:
        if runtime.attach:
            return Check(
                name="port",
                severity=Severity.OK,
                detail=f"an inference server is serving on port {runtime.port}"
                + (f" (pid {report.pid})" if report.pid else ""),
            )
        return Check(
            name="port",
            severity=Severity.FAIL,
            detail=f"port {runtime.port} is already serving"
            + (f" (pid {report.pid}, {report.process_name})" if report.pid else ""),
            hint="Starting here would measure that process, not a new one. "
            "Stop it, choose another port, or set runtime.attach.",
        )

    return Check(
        name="port",
        severity=Severity.FAIL,
        detail=report.detail or f"port {runtime.port} is held by another service",
        hint="Choose a free port for the runtime.",
    )


async def check_server_health(
    config: ResolvedConfig, *, client: httpx.AsyncClient | None = None
) -> Check:
    """Ask the running server what it is serving, if anything is running.

    Only meaningful when something is already there; a server this harness
    would start does not exist yet, and saying "unreachable" about it would be
    noise rather than a finding.
    """
    runtime = config.config.runtime
    owned = client is None
    probe = client or httpx.AsyncClient()
    try:
        response = await probe.get(f"{runtime.base_url}/props", timeout=3.0)
        payload = response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        payload = None
    finally:
        if owned:
            await probe.aclose()

    if payload is None:
        if runtime.attach:
            return Check(
                name="server",
                severity=Severity.FAIL,
                detail=f"attach is configured but {runtime.base_url} does not answer",
            )
        return Check(
            name="server",
            severity=Severity.SKIP,
            detail="no server running yet; it will be started",
        )

    served = str(payload.get("model_path") or "unknown")
    n_ctx = _served_context(payload)
    checks: list[str] = [f"serving {Path(served).name}"]
    if n_ctx:
        checks.append(f"n_ctx {n_ctx}")

    configured = config.config.model.path
    if configured is not None and Path(served).name != configured.name:
        return Check(
            name="server",
            severity=Severity.WARN,
            detail=f"{', '.join(checks)}, but the configuration names "
            f"{configured.name}",
            hint="The run would use the loaded model, not the configured one.",
        )

    window = config.config.context.context_window
    if n_ctx and n_ctx < window:
        return Check(
            name="server",
            severity=Severity.WARN,
            detail=f"{', '.join(checks)}, below the configured context window {window}",
            hint="The token budget would grow prompts past what this server accepts.",
        )
    return Check(name="server", severity=Severity.OK, detail=", ".join(checks))


def _served_context(payload: dict[str, object]) -> int | None:
    settings = payload.get("default_generation_settings")
    if isinstance(settings, dict):
        value = settings.get("n_ctx")
        if isinstance(value, int):
            return value
    value = payload.get("n_ctx")
    return value if isinstance(value, int) else None


# -- measured facts --------------------------------------------------------


def check_against_profile(config: ResolvedConfig) -> Check | None:
    """Flag configuration that contradicts what was measured on this machine.

    The profile is advisory, so disagreeing with it is allowed — but doing so
    unknowingly is how a run gets attributed to a setting nobody chose. Only
    reported when a recommendation exists and was overridden.
    """
    profile = config.hardware_profile
    if not profile:
        return None
    recommended = profile.get("recommended")
    if not isinstance(recommended, dict):
        return None

    conflicts: list[str] = []
    threads = recommended.get("threads")
    if threads and config.config.model.threads and config.config.model.threads != threads:
        conflicts.append(
            f"threads {config.config.model.threads} against a measured {threads}"
        )
    if not conflicts:
        return Check(
            name="profile",
            severity=Severity.OK,
            detail="configuration agrees with the measured profile",
        )
    reasons = profile.get("recommended", {}).get("rationale", {})
    hint = reasons.get("threads") if isinstance(reasons, dict) else None
    return Check(
        name="profile",
        severity=Severity.WARN,
        detail="; ".join(conflicts),
        hint=hint,
    )


SYNC_CHECKS: tuple[CheckFn, ...] = (
    check_python,
    check_uv,
    check_workspace,
    check_disk_space,
    check_database,
    check_sandbox,
    check_server_binary,
    check_model_path,
    check_against_profile,
)

ASYNC_CHECKS: tuple[AsyncCheckFn, ...] = (
    check_port,
    check_server_health,
)


async def diagnose(config: ResolvedConfig) -> Diagnosis:
    """Run every check against the resolved configuration.

    A check that raises becomes a ``SKIP`` naming the exception: a diagnostic
    that dies on its own first surprise tells you less than one that finishes
    and admits a gap.
    """
    checks: list[Check] = []
    for sync_check in SYNC_CHECKS:
        checks.append(_safely(sync_check.__name__, lambda fn=sync_check: fn(config)))
    for async_check in ASYNC_CHECKS:
        try:
            result = await async_check(config)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            result = Check(
                name=_name_of(async_check.__name__),
                severity=Severity.SKIP,
                detail=f"check could not run: {type(exc).__name__}: {exc}",
            )
        if result is not None:
            checks.append(result)
    return Diagnosis(checks=[check for check in checks if check is not None])


def _safely(label: str, call: Callable[[], Check | None]) -> Check | None:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return Check(
            name=_name_of(label),
            severity=Severity.SKIP,
            detail=f"check could not run: {type(exc).__name__}: {exc}",
        )


def _name_of(function_name: str) -> str:
    return function_name.removeprefix("check_").replace("_", "-")


__all__ = [
    "Check",
    "Diagnosis",
    "Severity",
    "diagnose",
]
