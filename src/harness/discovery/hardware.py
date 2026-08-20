"""Hardware probing via /proc, /sys and optional vendor tools.

Reads Linux interfaces directly where possible rather than shelling out, so
probing is fast and does not depend on locale-specific command output.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

from harness.discovery.models import CpuInfo, GpuInfo, MemoryInfo, SandboxInfo, StorageInfo

_MEMINFO = Path("/proc/meminfo")
_CPUINFO = Path("/proc/cpuinfo")
_SWAPS = Path("/proc/swaps")


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a mapping of field name to bytes."""
    out: dict[str, int] = {}
    try:
        for line in _MEMINFO.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                # Values are in kB unless a unit says otherwise.
                out[key.strip()] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def probe_cpu() -> CpuInfo:
    info = CpuInfo()
    try:
        text = _CPUINFO.read_text()
    except OSError:
        return info

    if m := re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE):
        info.model_name = m.group(1).strip()

    flags: set[str] = set()
    if m := re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE):
        flags = set(m.group(1).split())
    info.has_avx2 = "avx2" in flags
    info.has_avx512 = any(f.startswith("avx512") for f in flags)

    info.logical_cores = text.count("processor\t:") or None
    core_ids = set(re.findall(r"^core id\s*:\s*(\d+)$", text, re.MULTILINE))
    phys_ids = set(re.findall(r"^physical id\s*:\s*(\d+)$", text, re.MULTILINE))
    if core_ids:
        info.physical_cores = len(core_ids) * max(len(phys_ids), 1)

    # L3 size lives in sysfs; index3 is L3 on the common x86 layout.
    for idx in (3, 2):
        cache = Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}")
        try:
            if (cache / "level").read_text().strip() == "3":
                size = (cache / "size").read_text().strip()
                if size.endswith("K"):
                    info.l3_cache_bytes = int(size[:-1]) * 1024
                elif size.endswith("M"):
                    info.l3_cache_bytes = int(size[:-1]) * 1024 * 1024
                break
        except (OSError, ValueError):
            continue
    return info


def probe_memory() -> MemoryInfo:
    mem = _read_meminfo()
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)

    is_zram = False
    try:
        for line in _SWAPS.read_text().splitlines()[1:]:
            if line.split() and "zram" in line.split()[0]:
                is_zram = True
                break
    except OSError:
        pass

    return MemoryInfo(
        total_bytes=mem.get("MemTotal", 0),
        # MemAvailable is the kernel's own estimate and is far more honest than
        # MemFree, which ignores reclaimable page cache.
        available_bytes=mem.get("MemAvailable", mem.get("MemFree", 0)),
        swap_total_bytes=swap_total,
        swap_used_bytes=max(0, swap_total - swap_free),
        swap_is_zram=is_zram,
    )


def probe_gpus() -> list[GpuInfo]:
    """Query NVIDIA GPUs. Returns an empty list when no vendor tool is present."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    cuda_version: str | None = None
    try:
        v = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           timeout=15, check=False)
        if m := re.search(r"CUDA Version:\s*([\d.]+)", v.stdout):
            cuda_version = m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GpuInfo(
                name=parts[0],
                vram_total_bytes=int(float(parts[1])) * 1024 * 1024,
                vram_used_bytes=int(float(parts[2])) * 1024 * 1024,
                driver_version=parts[3],
                cuda_version=cuda_version,
            ))
        except ValueError:
            continue
    return gpus


def probe_storage(paths: list[str] | None = None) -> list[StorageInfo]:
    """Report free space for the mount points that matter to the harness."""
    candidates = paths or ["/", str(Path.home()), str(Path.cwd())]
    seen: set[str] = set()
    out: list[StorageInfo] = []
    for path in candidates:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        key = f"{usage.total}:{usage.free}"
        if key in seen:
            continue
        seen.add(key)
        out.append(StorageInfo(mount_point=path, total_bytes=usage.total,
                               free_bytes=usage.free))
    return out


def probe_hardware(storage_paths: list[str] | None = None) -> dict:
    """Probe everything at once. Returns a dict ready for HardwareProfile."""
    return {
        "hostname": platform.node() or None,
        "kernel": platform.release() or None,
        "os_release": _os_release_name(),
        "cpu": probe_cpu(),
        "memory": probe_memory(),
        "gpus": probe_gpus(),
        "storage": probe_storage(storage_paths),
        "sandbox": probe_sandbox(),
    }


def probe_sandbox() -> SandboxInfo:
    """Detect the bubblewrap sandbox the shell tool depends on.

    Bubblewrap is what makes an untrusted shell command safe to run: without
    it the shell tool refuses such commands outright (fail-closed). Its
    absence is a readiness gap for any agent run, so it is probed here rather
    than discovered only when a command is first denied.
    """
    bwrap = shutil.which("bwrap")
    return SandboxInfo(available=bwrap is not None, bwrap_path=bwrap)


def _os_release_name() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    except OSError:
        pass
    return None
