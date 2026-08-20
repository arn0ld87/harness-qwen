"""Environment discovery: hardware, inference runtime, and model metadata.

Nothing in this package guesses. Every field is either read from the system,
read from the model file, or reported by the running inference server. Values
that could not be determined are ``None`` and are rendered as "unknown" rather
than replaced by a plausible default.
"""

from harness.discovery.gguf import GgufError, GgufMetadata, read_gguf_metadata
from harness.discovery.hardware import probe_cpu, probe_gpus, probe_hardware, probe_memory
from harness.discovery.models import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
    ModelInfo,
    Recommendations,
    RuntimeInfo,
    StorageInfo,
)
from harness.discovery.profile import (
    build_profile,
    load_profile,
    max_context_for_vram,
    ram_reserve_bytes,
    save_profile,
)
from harness.discovery.runtime import probe_runtimes

__all__ = [
    "CpuInfo",
    "GgufError",
    "GgufMetadata",
    "GpuInfo",
    "HardwareProfile",
    "MemoryInfo",
    "ModelInfo",
    "Recommendations",
    "RuntimeInfo",
    "StorageInfo",
    "build_profile",
    "load_profile",
    "max_context_for_vram",
    "probe_cpu",
    "probe_gpus",
    "probe_hardware",
    "probe_memory",
    "probe_runtimes",
    "ram_reserve_bytes",
    "read_gguf_metadata",
    "save_profile",
]
