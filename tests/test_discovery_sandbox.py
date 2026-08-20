"""Sandbox detection for `harness doctor` (issue #7).

`probe_sandbox` is what lets doctor surface a missing bubblewrap before a
run discovers it as denied commands. It must reflect the real bwrap
availability, and the doctor warning must fire when it is absent.
"""

from __future__ import annotations

import io
import shutil
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from rich.console import Console

from harness.cli import _render_warnings
from harness.discovery.hardware import probe_hardware, probe_sandbox
from harness.discovery.models import (
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    SandboxInfo,
)


def test_probe_sandbox_reports_available_when_bwrap_present() -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap not installed")
    info = probe_sandbox()
    assert info.available is True
    assert info.bwrap_path is not None


def test_probe_sandbox_reports_unavailable_when_bwrap_missing() -> None:
    with patch("harness.discovery.hardware.shutil.which", lambda _c: None):
        info = probe_sandbox()
    assert info.available is False
    assert info.bwrap_path is None


def test_probe_hardware_includes_sandbox() -> None:
    hw = probe_hardware()
    assert isinstance(hw["sandbox"], SandboxInfo)


def test_doctor_warns_when_sandbox_missing() -> None:
    profile = HardwareProfile(
        created_at=datetime.now(UTC),
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=0, available_bytes=0),
        sandbox=SandboxInfo(available=False),
    )

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    with patch("harness.cli.console", console):
        _render_warnings(profile)
    out = buf.getvalue()
    assert "bwrap" in out.lower()
    assert "fail-closed" in out.lower()


def test_doctor_no_warning_when_sandbox_present() -> None:
    profile = HardwareProfile(
        created_at=datetime.now(UTC),
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=0, available_bytes=0),
        sandbox=SandboxInfo(available=True),
    )

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    with patch("harness.cli.console", console):
        _render_warnings(profile)
    assert "bwrap" not in buf.getvalue().lower()