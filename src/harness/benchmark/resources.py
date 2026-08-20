"""Memory and VRAM at a point in time.

Separate from ``discovery`` because the question is different: discovery
describes the machine once, a benchmark needs the same numbers repeatedly and
cheaply, at points it chooses. What it reuses from discovery is the probing
itself — there is no second way to read ``/proc/meminfo``.

zram is the field that matters here. A configuration that only fits by
spilling into it has not found memory, it has traded memory for the CPU time
the model already saturates (DISCOVERY.md section 1), and the resulting
throughput number looks like a property of the flags rather than of the
memory pressure they caused.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.benchmark.models import ResourceSample
from harness.discovery.hardware import probe_gpus, probe_memory


def sample_resources(label: str = "") -> ResourceSample:
    """Read current memory and VRAM use.

    Missing values stay ``None``: a machine without ``nvidia-smi`` reports no
    VRAM rather than zero VRAM, because the two would be indistinguishable in
    the artefact and only one of them is true.
    """
    memory = probe_memory()
    gpus = probe_gpus()
    used = [gpu.vram_used_bytes for gpu in gpus if gpu.vram_used_bytes is not None]
    total = [gpu.vram_total_bytes for gpu in gpus if gpu.vram_total_bytes is not None]

    return ResourceSample(
        label=label,
        at=datetime.now(UTC),
        ram_total_bytes=memory.total_bytes,
        ram_available_bytes=memory.available_bytes,
        swap_total_bytes=memory.swap_total_bytes,
        swap_used_bytes=memory.swap_used_bytes,
        swap_is_zram=memory.swap_is_zram,
        vram_used_bytes=max(used) if used else None,
        vram_total_bytes=max(total) if total else None,
    )


__all__ = ["sample_resources"]
