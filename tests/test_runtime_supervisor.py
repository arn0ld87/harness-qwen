"""Lifecycle of a local inference server (issue #10).

Everything here runs against `tests/fixtures/stub_server.py`, which answers
`/health` and can be told to start slowly, fail to start, or die mid-flight.
The supervisor's job is process lifecycle, not inference: exercising it
against the 35B model would cost 25 s per case, need a GPU in CI, and test
llama.cpp rather than this code.

The one test that does touch the real server is marked `local_llm` and only
attaches — it never stops a process it did not start.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.core import HealthStatus
from harness.runtime import (
    LlamaServerSupervisor,
    Ownership,
    RuntimeCrashed,
    RuntimeNotOwned,
    RuntimeStartTimeout,
    build_argv,
)

STUB = Path(__file__).parent / "fixtures" / "stub_server.py"


def _config(port: int, **runtime: object) -> HarnessConfig:
    """A config whose "llama-server" is the stub itself.

    The stub takes the binary's place rather than being appended as an
    argument: ``build_argv`` puts the typed flags before ``extra_flags``, so a
    stub passed as an extra flag would arrive after ``--model`` and never be
    run. It accepts and ignores the real server's flags, which is what makes
    the substitution honest.
    """
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


# -- argv ------------------------------------------------------------------


def test_argv_carries_the_typed_flags() -> None:
    config = HarnessConfig.model_validate(
        {
            "runtime": {"server_binary": "/usr/bin/llama-server", "port": 9001},
            "model": {
                "path": "/models/x.gguf",
                "alias": "x",
                "n_ctx": 4096,
                "n_gpu_layers": 20,
                "threads": 4,
            },
        }
    )

    argv = build_argv(config)

    assert argv[0] == "/usr/bin/llama-server"
    assert "--model" in argv and "/models/x.gguf" in argv
    assert "--port" in argv and "9001" in argv
    assert "--ctx-size" in argv and "4096" in argv
    assert "--n-gpu-layers" in argv and "20" in argv
    assert "--threads" in argv and "4" in argv


def test_unset_flags_are_left_off_entirely() -> None:
    """Absent means "let the server decide", not "pass an empty value"."""
    config = HarnessConfig.model_validate(
        {
            "runtime": {"server_binary": "/usr/bin/llama-server"},
            "model": {"path": "/models/x.gguf"},
        }
    )

    argv = build_argv(config)

    assert "--n-gpu-layers" not in argv
    assert "--threads" not in argv
    assert "--alias" not in argv


def test_extra_flags_come_last_so_they_can_override() -> None:
    config = HarnessConfig.model_validate(
        {
            "runtime": {"server_binary": "/usr/bin/llama-server"},
            "model": {"path": "/models/x.gguf", "extra_flags": ["--flash-attn"]},
        }
    )

    argv = build_argv(config)

    assert argv[-1] == "--flash-attn"


def test_starting_without_a_binary_is_refused() -> None:
    config = HarnessConfig.model_validate({"model": {"path": "/models/x.gguf"}})
    with pytest.raises(ValueError, match="server_binary"):
        build_argv(config)


# -- start, health, stop ---------------------------------------------------


@pytest.mark.asyncio
async def test_start_waits_for_health_and_reports_the_process(
    port: int, tmp_path: Path
) -> None:
    supervisor = _supervisor(port, tmp_path)
    try:
        handle = await supervisor.start()

        assert handle.ownership is Ownership.OWNED
        assert handle.pid is not None and handle.pid > 0
        assert handle.base_url.endswith(str(port))
        assert (await supervisor.health()).reachable
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_a_slow_start_is_waited_out_not_called_a_failure(
    port: int, tmp_path: Path
) -> None:
    """503 "loading model" is progress; only the deadline ends the wait."""
    supervisor = _supervisor(port, tmp_path, "--ready-after", "1.0")
    try:
        handle = await supervisor.start()
        assert handle.pid is not None
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_a_process_that_dies_during_startup_is_a_crash_not_a_timeout(
    port: int, tmp_path: Path
) -> None:
    """Waiting 180 s for a process that is already dead helps nobody."""
    supervisor = _supervisor(port, tmp_path, "--fail-to-start", startup_timeout_s=30.0)

    with pytest.raises(RuntimeCrashed) as exc:
        await supervisor.start()

    assert "2" in str(exc.value)  # the stub's exit code
    assert "could not load model" in str(exc.value)  # its own diagnosis


@pytest.mark.asyncio
async def test_a_server_that_never_becomes_healthy_times_out(
    port: int, tmp_path: Path
) -> None:
    supervisor = _supervisor(
        port, tmp_path, "--ready-after", "60", startup_timeout_s=1.0
    )

    with pytest.raises(RuntimeStartTimeout):
        await supervisor.start()

    # Whatever we started, we cleaned up.
    assert supervisor.handle is None


@pytest.mark.asyncio
async def test_stop_is_idempotent(port: int, tmp_path: Path) -> None:
    supervisor = _supervisor(port, tmp_path)
    await supervisor.start()

    await supervisor.stop()
    await supervisor.stop()

    assert supervisor.handle is None


@pytest.mark.asyncio
async def test_a_crash_after_startup_is_visible_as_unreachable(
    port: int, tmp_path: Path
) -> None:
    supervisor = _supervisor(port, tmp_path, "--die-after", "0.3", "--exit-code", "9")
    try:
        await supervisor.start()
        assert supervisor.running is True

        await _wait_until(lambda: not supervisor.running, timeout=5.0)

        status = await supervisor.health()
        assert status.reachable is False
        assert "9" in (status.detail or "")
    finally:
        await supervisor.stop()


# -- attach ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_uses_a_server_it_did_not_start(port: int, tmp_path: Path) -> None:
    owner = _supervisor(port, tmp_path)
    await owner.start()
    try:
        attacher = LlamaServerSupervisor(
            _config(port, attach=True), log_dir=tmp_path
        )
        handle = await attacher.attach()

        assert handle.ownership is Ownership.ATTACHED
        assert (await attacher.health()).reachable
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_stop_refuses_to_kill_a_process_it_does_not_own(
    port: int, tmp_path: Path
) -> None:
    """The whole point of tracking ownership.

    Someone else's server is someone else's — a supervisor that stopped it
    would take down a shared model load, which costs minutes to rebuild.
    """
    owner = _supervisor(port, tmp_path)
    await owner.start()
    try:
        attacher = LlamaServerSupervisor(_config(port, attach=True), log_dir=tmp_path)
        await attacher.attach()

        with pytest.raises(RuntimeNotOwned):
            await attacher.stop()

        assert (await owner.health()).reachable
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_attaching_to_nothing_fails_rather_than_pretending(
    port: int, tmp_path: Path
) -> None:
    attacher = LlamaServerSupervisor(_config(port, attach=True), log_dir=tmp_path)

    with pytest.raises(RuntimeStartTimeout):
        await attacher.attach(timeout_s=1.0)


@pytest.mark.asyncio
async def test_ensure_picks_start_or_attach_from_the_config(
    port: int, tmp_path: Path
) -> None:
    owner = _supervisor(port, tmp_path)
    handle = await owner.ensure()
    try:
        assert handle.ownership is Ownership.OWNED

        second = LlamaServerSupervisor(_config(port, attach=True), log_dir=tmp_path)
        assert (await second.ensure()).ownership is Ownership.ATTACHED
    finally:
        await owner.stop()


# -- logs ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_output_is_captured_for_diagnosis(
    port: int, tmp_path: Path
) -> None:
    supervisor = _supervisor(port, tmp_path)
    try:
        handle = await supervisor.start()

        assert handle.log_path is not None
        assert "stub-server listening" in handle.log_path.read_text(encoding="utf-8")
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_a_secret_printed_by_the_server_is_redacted_on_disk(
    port: int, tmp_path: Path
) -> None:
    """A server that logs its own key must not have it written to disk here."""
    supervisor = _supervisor(port, tmp_path, "--leak-secret")
    try:
        handle = await supervisor.start()

        assert handle.log_path is not None
        written = handle.log_path.read_text(encoding="utf-8")
        assert "sk-live-abcdef1234567890" not in written
        assert "redacted" in written
    finally:
        await supervisor.stop()


# -- against the real thing ------------------------------------------------


@pytest.mark.local_llm
@pytest.mark.asyncio
async def test_attach_to_the_running_llama_server(tmp_path: Path) -> None:
    """Attach to whatever is actually serving, and leave it exactly as found.

    The stub proves the state machine; this proves the wire format against a
    real llama-server. It never calls stop().
    """
    config = HarnessConfig.model_validate({"runtime": {"attach": True}})
    supervisor = LlamaServerSupervisor(config, log_dir=tmp_path)

    handle = await supervisor.attach(timeout_s=10.0)
    status = await supervisor.health()

    assert handle.ownership is Ownership.ATTACHED
    assert status.reachable is True
    with pytest.raises(RuntimeNotOwned):
        await supervisor.stop()


async def _wait_until(predicate, *, timeout: float) -> None:
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached within timeout")


def test_health_status_is_the_shared_type() -> None:
    """No parallel health model: the loop already understands this one."""
    assert HealthStatus(reachable=True).reachable is True
