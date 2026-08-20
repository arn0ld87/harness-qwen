"""Proving which process the numbers describe.

This is the precondition of the whole layer, not a diagnostic bolted onto it.
A benchmark that measures a server this harness did not start reports figures
in the perfectly normal range and raises nothing; two runs across different
launch configurations then come back identical and the only visible symptom
is that the flag sweep "found no difference" (#11, API.md "Runtime layer").

So a run against an *owned* server verifies that the pid holding the port is
the one that was started, and refuses to measure otherwise. A run against an
attached server cannot make that claim at all — it is measured, but the
artefact says so, and it does not pretend to know the launch flags.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from harness.benchmark.models import RuntimeFingerprint
from harness.config.schema import SECRET_FLAG_SUFFIXES, HarnessConfig
from harness.runtime import (
    Ownership,
    PortReport,
    RuntimeHandle,
    build_argv,
    inspect_port_async,
    verify_owner,
)

InspectPort = Callable[[int, str], Awaitable[PortReport]]
VerifyOwner = Callable[[int, int, str], Awaitable[PortReport]]

ATTACHED_NOTE = (
    "attached runtime: this harness did not start the server on this port, so "
    "its identity and its launch flags are not established by this run"
)
NO_HANDLE_NOTE = (
    "no runtime handle: nothing connects the measured endpoint to a process "
    "this harness started"
)
NO_PID_NOTE = (
    "owned runtime without a pid: the process cannot be matched against the "
    "port it is supposed to hold"
)


def redact_argv(argv: list[str]) -> list[str]:
    """Hide the value following a credential-carrying flag.

    The resolver has an equivalent for ``extra_flags``, but it is private to
    that module and shaped for a flag list, not for a full command line whose
    first element is a binary path. What is shared is the knowledge of which
    flag names carry a secret, and that lives in ``config.schema``.

    The flag itself stays visible: that a key is being passed is part of the
    launch configuration a later reader needs to see.
    """
    out: list[str] = []
    hide_next = False
    for item in argv:
        if hide_next:
            out.append("[redacted:flag_value]")
            hide_next = False
            continue
        if item.startswith("-"):
            name, sep, _ = item.partition("=")
            if any(name.lower().endswith(suffix) for suffix in SECRET_FLAG_SUFFIXES):
                out.append(f"{name}=[redacted:flag_value]" if sep else item)
                hide_next = not sep
                continue
        out.append(item)
    return out


def _launch_argv(config: HarnessConfig, handle: RuntimeHandle | None) -> list[str] | None:
    """The command line, but only where this harness actually chose it."""
    if handle is None or not handle.owned:
        return None
    try:
        return redact_argv(build_argv(config))
    except ValueError:
        # Started from something this config cannot reconstruct. Recording a
        # guess would be worse than recording nothing.
        return None


async def probe_identity(
    config: HarnessConfig,
    handle: RuntimeHandle | None,
    *,
    inspect: InspectPort = inspect_port_async,
    verify: VerifyOwner = verify_owner,
) -> RuntimeFingerprint:
    """Establish what holds the configured port.

    Raises :class:`harness.runtime.RuntimeIdentityMismatch` when the handle
    claims ownership and the port disagrees. That is deliberate: there is
    nothing useful to record about a run whose subject is unknown, and the
    caller has not measured anything yet.
    """
    port, host = config.runtime.port, config.runtime.host

    if handle is not None and handle.owned and handle.pid is not None:
        report = await verify(port, handle.pid, host)
        verified, detail = True, None
    else:
        report = await inspect(port, host)
        verified = False
        if handle is None:
            detail = NO_HANDLE_NOTE
        elif handle.ownership is Ownership.ATTACHED:
            detail = ATTACHED_NOTE
        else:
            detail = NO_PID_NOTE

    return RuntimeFingerprint(
        base_url=config.runtime.base_url,
        host=host,
        port=port,
        ownership=handle.ownership if handle is not None else None,
        pid=report.pid if report.pid is not None else (handle.pid if handle else None),
        process_name=report.process_name,
        port_state=str(report.state),
        identity_verified=verified,
        identity_detail=detail,
        started_at=handle.started_at if handle is not None else None,
        launch_argv=_launch_argv(config, handle),
    )


def identity_changed(before: RuntimeFingerprint,
                     after: RuntimeFingerprint) -> str | None:
    """Describe how the port's occupant moved during the run, or ``None``.

    Checked after the samples are in rather than instead of before them: a
    server that died and was replaced halfway through produces a run whose
    first half and second half describe different processes, and neither the
    pre-flight check nor the samples themselves can show that.
    """
    if before.pid != after.pid:
        return f"the process on port {before.port} changed from pid {before.pid} to {after.pid}"
    if before.port_state != after.port_state:
        return (
            f"port {before.port} went from {before.port_state} to {after.port_state} "
            "during the run"
        )
    return None


__all__ = [
    "ATTACHED_NOTE",
    "NO_HANDLE_NOTE",
    "NO_PID_NOTE",
    "InspectPort",
    "VerifyOwner",
    "identity_changed",
    "probe_identity",
    "redact_argv",
]
