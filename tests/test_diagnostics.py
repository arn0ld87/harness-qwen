"""Pre-flight diagnosis (issue #16).

Every check runs against a stub or a temporary directory: the point of a
readiness check is that it works on a host that is *not* ready, which is
exactly the host a test cannot rely on being. No GPU, no 35B model, no network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness.config import resolve_config
from harness.diagnostics import Severity, diagnose
from harness.diagnostics.checks import (
    Check,
    Diagnosis,
    check_database,
    check_disk_space,
    check_model_path,
    check_port,
    check_python,
    check_server_binary,
    check_server_health,
    check_workspace,
)

STUB = Path(__file__).parent / "fixtures" / "stub_server.py"


def _config(**overrides: object):
    return resolve_config(overrides=overrides)


# -- the exit code a script acts on ----------------------------------------


def test_a_clean_diagnosis_exits_zero() -> None:
    diagnosis = Diagnosis(
        checks=[Check(name="a", severity=Severity.OK, detail="fine")]
    )
    assert diagnosis.exit_code() == 0


def test_a_failure_exits_one() -> None:
    diagnosis = Diagnosis(
        checks=[Check(name="a", severity=Severity.FAIL, detail="broken")]
    )
    assert diagnosis.exit_code() == 1


def test_warnings_do_not_fail_by_default() -> None:
    """Most warnings describe a host that works and would measure badly.

    Refusing to run at all would be a worse answer than saying so.
    """
    diagnosis = Diagnosis(
        checks=[Check(name="a", severity=Severity.WARN, detail="questionable")]
    )

    assert diagnosis.exit_code() == 0
    assert diagnosis.exit_code(strict=True) == 2


def test_skipped_checks_are_neither_pass_nor_fail() -> None:
    diagnosis = Diagnosis(
        checks=[Check(name="a", severity=Severity.SKIP, detail="not applicable")]
    )
    assert diagnosis.exit_code(strict=True) == 0


# -- host ------------------------------------------------------------------


def test_python_version_is_reported() -> None:
    assert check_python(_config()).severity is Severity.OK


def test_a_missing_workspace_fails(tmp_path: Path) -> None:
    check = check_workspace(_config(workspace=str(tmp_path / "absent")))

    assert check.severity is Severity.FAIL
    assert "does not exist" in check.detail


def test_writability_is_tested_by_writing(tmp_path: Path) -> None:
    """Permission bits lie on read-only mounts and full filesystems.

    Each of those then fails later, at a point that looks like a bug in the
    agent rather than in the host.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert check_workspace(_config(workspace=str(workspace))).severity is Severity.OK

    workspace.chmod(0o500)
    try:
        check = check_workspace(_config(workspace=str(workspace)))
        assert check.severity is Severity.FAIL
        assert "not writable" in check.detail
    finally:
        workspace.chmod(0o700)


def test_disk_space_is_reported(tmp_path: Path) -> None:
    assert check_disk_space(_config(workspace=str(tmp_path))).severity in (
        Severity.OK,
        Severity.WARN,
    )


def test_a_database_from_a_newer_harness_is_refused(tmp_path: Path) -> None:
    """Reading it would corrupt the resume point it is meant to protect."""
    database = tmp_path / "memory.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    check = check_database(_config(workspace=str(tmp_path), database=str(database)))

    assert check.severity is Severity.FAIL
    assert "99" in check.detail


def test_an_absent_database_is_fine(tmp_path: Path) -> None:
    check = check_database(
        _config(workspace=str(tmp_path), database=str(tmp_path / "new.sqlite"))
    )

    assert check.severity is Severity.OK
    assert "will be created" in check.detail


# -- runtime ---------------------------------------------------------------


def test_no_binary_and_no_attach_is_a_failure() -> None:
    check = check_server_binary(_config())

    assert check.severity is Severity.FAIL
    assert "attach" in (check.hint or "")


def test_attach_mode_needs_no_binary() -> None:
    assert check_server_binary(
        _config(**{"runtime.attach": True})
    ).severity is Severity.SKIP


def test_a_binary_that_is_not_executable_fails(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o644)

    check = check_server_binary(_config(**{"runtime.server_binary": str(binary)}))

    assert check.severity is Severity.FAIL
    assert "not executable" in check.detail


def test_a_missing_model_file_fails(tmp_path: Path) -> None:
    check = check_model_path(_config(**{"model.path": str(tmp_path / "absent.gguf")}))
    assert check.severity is Severity.FAIL


@pytest.mark.asyncio
async def test_a_free_port_is_fine_for_a_start(unused_tcp_port: int) -> None:
    check = await check_port(_config(**{"runtime.port": unused_tcp_port}))
    assert check.severity is Severity.OK


@pytest.mark.asyncio
async def test_a_free_port_is_a_failure_for_an_attach(unused_tcp_port: int) -> None:
    check = await check_port(
        _config(**{"runtime.port": unused_tcp_port, "runtime.attach": True})
    )

    assert check.severity is Severity.FAIL


@pytest.mark.asyncio
async def test_an_occupied_port_reads_two_ways(unused_tcp_port: int, tmp_path: Path) -> None:
    """The same fact is a blocker for a start and a precondition for an attach.

    Which is why it is reported rather than acted on.
    """
    from harness.config import HarnessConfig
    from harness.runtime import LlamaServerSupervisor

    config = HarnessConfig.model_validate(
        {
            "runtime": {"server_binary": str(STUB), "port": unused_tcp_port},
            "model": {"path": "/nonexistent/model.gguf"},
        }
    )
    supervisor = LlamaServerSupervisor(config, log_dir=tmp_path)
    await supervisor.start()
    try:
        starting = await check_port(_config(**{"runtime.port": unused_tcp_port}))
        attaching = await check_port(
            _config(**{"runtime.port": unused_tcp_port, "runtime.attach": True})
        )

        assert starting.severity is Severity.FAIL
        assert "already serving" in starting.detail
        assert attaching.severity is Severity.OK
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_no_server_yet_is_not_a_finding(unused_tcp_port: int) -> None:
    """One that will be started does not exist yet; calling it unreachable is noise."""
    check = await check_server_health(_config(**{"runtime.port": unused_tcp_port}))
    assert check.severity is Severity.SKIP


@pytest.mark.asyncio
async def test_attach_against_nothing_is_a_failure(unused_tcp_port: int) -> None:
    check = await check_server_health(
        _config(**{"runtime.port": unused_tcp_port, "runtime.attach": True})
    )
    assert check.severity is Severity.FAIL


@pytest.mark.asyncio
async def test_a_server_loading_another_model_is_a_warning(
    unused_tcp_port: int, tmp_path: Path
) -> None:
    """It would run — against weights nobody asked for."""
    from harness.config import HarnessConfig
    from harness.runtime import LlamaServerSupervisor

    config = HarnessConfig.model_validate(
        {"runtime": {"server_binary": str(STUB), "port": unused_tcp_port}}
    )
    config.model.extra_flags = ["--model", "/served/other.gguf"]
    supervisor = LlamaServerSupervisor(config, log_dir=tmp_path)
    await supervisor.start()
    try:
        check = await check_server_health(
            _config(
                **{
                    "runtime.port": unused_tcp_port,
                    "model.path": "/configured/wanted.gguf",
                }
            )
        )

        assert check.severity is Severity.WARN
        assert "wanted.gguf" in check.detail
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_a_server_window_smaller_than_the_budget_is_a_warning(
    unused_tcp_port: int, tmp_path: Path
) -> None:
    """The stub serves n_ctx 4096; the budget would fill far more than that."""
    from harness.config import HarnessConfig
    from harness.runtime import LlamaServerSupervisor

    config = HarnessConfig.model_validate(
        {"runtime": {"server_binary": str(STUB), "port": unused_tcp_port}}
    )
    supervisor = LlamaServerSupervisor(config, log_dir=tmp_path)
    await supervisor.start()
    try:
        check = await check_server_health(_config(**{"runtime.port": unused_tcp_port}))

        assert check.severity is Severity.WARN
        assert "context window" in check.detail
    finally:
        await supervisor.stop()


# -- the whole thing -------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnose_runs_every_check(tmp_path: Path) -> None:
    diagnosis = await diagnose(_config(workspace=str(tmp_path)))

    names = {check.name for check in diagnosis.checks}
    assert {"python", "workspace", "database", "sandbox", "port", "server"} <= names


@pytest.mark.asyncio
async def test_a_check_that_explodes_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch
) -> None:
    """A diagnostic that dies on its first surprise tells you less than one
    that finishes and admits a gap."""
    from harness.diagnostics import checks as module

    def explode(_config: object) -> Check:
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(module, "SYNC_CHECKS", (explode, module.check_python))

    diagnosis = await diagnose(_config(workspace=str(tmp_path)))

    skipped = diagnosis.by_severity(Severity.SKIP)
    assert any("probe blew up" in check.detail for check in skipped)
    assert any(check.name == "python" for check in diagnosis.checks)


@pytest.mark.asyncio
async def test_diagnosis_changes_nothing(tmp_path: Path) -> None:
    """A diagnostic that fixes things quietly cannot be trusted to report."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    before = sorted(path.name for path in workspace.iterdir())

    await diagnose(_config(workspace=str(workspace)))

    assert sorted(path.name for path in workspace.iterdir()) == before
