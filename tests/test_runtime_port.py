"""Port ownership and start validation (issue #11).

The failure this prevents: a start that appears to succeed because something
was already answering on the port. Every number measured afterwards then
describes a process nobody chose — and it looks like a normal run, which is
what makes it expensive. Benchmarks comparing two configurations would report
identical results and no error.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.runtime import (
    LlamaServerSupervisor,
    Ownership,
    PortConflict,
    PortState,
    RuntimeIdentityMismatch,
    inspect_port,
    inspect_port_async,
    verify_owner,
)

STUB = Path(__file__).parent / "fixtures" / "stub_server.py"


def _config(port: int, **runtime: object) -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "runtime": {"server_binary": str(STUB), "port": port, **runtime},
            "model": {"path": "/nonexistent/model.gguf"},
        }
    )


def _supervisor(port: int, tmp_path: Path, *stub_flags: str, **runtime: object):
    config = _config(port, **runtime)
    config.model.extra_flags = list(stub_flags)
    return LlamaServerSupervisor(config, log_dir=tmp_path)


@pytest.fixture
def port(unused_tcp_port: int) -> int:
    return unused_tcp_port


# -- classifying what holds a port -----------------------------------------


def test_a_free_port_is_reported_free(port: int) -> None:
    assert inspect_port(port).state is PortState.FREE


def test_a_bound_port_is_reported_taken(port: int) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)

        report = inspect_port(port)

    assert report.state is not PortState.FREE


@pytest.mark.asyncio
async def test_a_llama_server_on_the_port_is_recognised_as_one(
    port: int, tmp_path: Path
) -> None:
    """A health endpoint is what separates "our kind of server" from "a port".

    The distinction decides whether attaching is even an option, so it is
    made by asking the endpoint, not by assuming from the port number.
    """
    supervisor = _supervisor(port, tmp_path)
    await supervisor.start()
    try:
        report = await inspect_port_async(port)

        assert report.state is PortState.INFERENCE_SERVER
        assert report.pid is not None
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_a_foreign_service_is_not_mistaken_for_a_server(port: int) -> None:
    """Something is listening, but it is not ours: never attach to that."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)

        report = await inspect_port_async(port)

    assert report.state is PortState.FOREIGN_SERVICE


# -- start refuses to run against someone else's process -------------------


@pytest.mark.asyncio
async def test_start_refuses_a_port_that_is_already_serving(
    port: int, tmp_path: Path
) -> None:
    """The scenario from the issue, end to end.

    An old server holds the port. The new start must fail loudly, and the old
    server must still be there afterwards — a start that "succeeded" by
    measuring the old process is the outcome this exists to prevent.
    """
    old = _supervisor(port, tmp_path)
    await old.start()
    try:
        new = _supervisor(port, tmp_path)

        with pytest.raises(PortConflict) as exc:
            await new.start()

        assert str(port) in str(exc.value)
        assert (await old.health()).reachable  # untouched
        assert new.handle is None
    finally:
        await old.stop()


@pytest.mark.asyncio
async def test_start_refuses_a_port_held_by_a_foreign_service(
    port: int, tmp_path: Path
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        supervisor = _supervisor(port, tmp_path)

        with pytest.raises(PortConflict):
            await supervisor.start()


@pytest.mark.asyncio
async def test_attaching_is_never_implicit(port: int, tmp_path: Path) -> None:
    """A busy port must not quietly turn a start into an attach.

    That is the same silent substitution, arrived at politely.
    """
    old = _supervisor(port, tmp_path)
    await old.start()
    try:
        new = _supervisor(port, tmp_path)  # attach not configured

        with pytest.raises(PortConflict):
            await new.start()
    finally:
        await old.stop()


@pytest.mark.asyncio
async def test_attach_is_allowed_when_it_is_asked_for(
    port: int, tmp_path: Path
) -> None:
    old = _supervisor(port, tmp_path)
    await old.start()
    try:
        attacher = LlamaServerSupervisor(_config(port, attach=True), log_dir=tmp_path)
        handle = await attacher.attach()

        assert handle.ownership is Ownership.ATTACHED
    finally:
        await old.stop()


@pytest.mark.asyncio
async def test_attach_refuses_a_port_that_is_not_an_inference_server(
    port: int, tmp_path: Path
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        attacher = LlamaServerSupervisor(_config(port, attach=True), log_dir=tmp_path)

        with pytest.raises(PortConflict):
            await attacher.attach(timeout_s=1.0)


# -- the endpoint that answers must be the process we started --------------


@pytest.mark.asyncio
async def test_the_answering_endpoint_is_verified_against_our_process(
    port: int, tmp_path: Path
) -> None:
    supervisor = _supervisor(port, tmp_path)
    try:
        handle = await supervisor.start()

        # Identity is more than "something answers": it is this pid, listening
        # on this port, started when we started it.
        assert handle.pid is not None
        report = await inspect_port_async(port)
        assert report.pid == handle.pid
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_an_endpoint_owned_by_another_process_is_rejected(
    port: int, tmp_path: Path
) -> None:
    """"Something answers" is not "our server answers".

    The real shape of this: a start fails while an older process keeps the
    port, and at the HTTP level that is indistinguishable from success. The
    check is against the pid holding the socket, not against the response.
    """
    supervisor = _supervisor(port, tmp_path)
    handle = await supervisor.start()
    try:
        assert handle.pid is not None
        await verify_owner(port, handle.pid)  # the true owner passes

        with pytest.raises(RuntimeIdentityMismatch) as exc:
            await verify_owner(port, handle.pid + 100000)

        assert str(port) in str(exc.value)
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_verifying_an_unheld_port_is_a_mismatch_not_a_pass(port: int) -> None:
    """Nothing listening means the process we started is not there.

    Treating "cannot tell" as "fine" is how an unverified start gets called a
    verified one.
    """
    with pytest.raises(RuntimeIdentityMismatch):
        await verify_owner(port, 999999)
