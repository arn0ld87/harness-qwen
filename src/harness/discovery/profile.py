"""Assembly of the hardware profile and derivation of safe defaults.

Two rules govern this module:

1. A recommendation is emitted only when something measured supports it. Values
   that require a benchmark stay ``None`` and carry a rationale saying so. An
   invented default written into a profile file becomes indistinguishable from
   a measurement the moment it is saved.
2. Memory arithmetic never plans to use all of RAM. A configuration that only
   fits by spilling into swap has not found memory, it has traded it for CPU
   time — and on this class of hardware swap is zram, i.e. compressed RAM that
   costs the very CPU the model is already saturating.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from harness.discovery.gguf import GgufError, GgufMetadata, read_gguf_metadata
from harness.discovery.hardware import probe_hardware
from harness.discovery.models import (
    HardwareProfile,
    ModelInfo,
    Recommendations,
    RuntimeInfo,
)
from harness.discovery.runtime import probe_runtimes

GIB = 1024 ** 3

# Held back for the OS, page cache, editors, compilers, test runs and git.
# Scaled with total RAM rather than fixed, so the profile behaves on machines
# both smaller and larger than the development target.
RAM_RESERVE_FRACTION = 0.18
RAM_RESERVE_MIN = 6 * GIB
RAM_RESERVE_MAX = 12 * GIB

# Left free on the GPU for the compute buffer, driver overhead and the desktop.
VRAM_HEADROOM_FRACTION = 0.12
VRAM_HEADROOM_MIN = 384 * 1024 * 1024


def ram_reserve_bytes(total_ram: int) -> int:
    """Memory deliberately left unused, in bytes."""
    return int(min(max(total_ram * RAM_RESERVE_FRACTION, RAM_RESERVE_MIN), RAM_RESERVE_MAX))


def model_info_from_gguf(meta: GgufMetadata) -> ModelInfo:
    """Project GGUF metadata onto the typed model description."""
    split = meta.attention_layer_split()
    return ModelInfo(
        path=meta.path,
        file_size_bytes=meta.file_size_bytes,
        architecture=meta.architecture,
        name=meta.name,
        size_label=meta.kv.get("general.size_label"),
        quantization=meta.quantization,
        n_layers=meta.n_trunk_layers,
        n_dense_attention_layers=split[0] if split else None,
        n_recurrent_layers=split[1] if split else None,
        n_mtp_layers=meta.n_mtp_layers,
        context_length_train=meta.arch_key("context_length"),
        embedding_length=meta.arch_key("embedding_length"),
        head_count=meta.arch_key("attention.head_count"),
        head_count_kv=meta.arch_key("attention.head_count_kv"),
        key_length=meta.arch_key("attention.key_length"),
        value_length=meta.arch_key("attention.value_length"),
        expert_count=meta.arch_key("expert_count"),
        expert_used_count=meta.arch_key("expert_used_count"),
        full_attention_interval=meta.arch_key("full_attention_interval"),
        ssm_state_size=meta.arch_key("ssm.state_size"),
        ssm_inner_size=meta.arch_key("ssm.inner_size"),
        embedded_sampling=meta.embedded_sampling(),
    )


def max_context_for_vram(meta: GgufMetadata, vram_free_bytes: int,
                         cache_type: str = "f16") -> int | None:
    """Largest context whose KV cache plus recurrent state fits in VRAM.

    This is an upper bound on what is *possible*, never a recommendation of
    what is *useful*. The useful size is decided by the benchmark, because
    prompt processing degrades with context length well before memory runs out.
    """
    per_token = meta.kv_cache_bytes_per_token(cache_type, cache_type)
    recurrent = meta.recurrent_state_bytes() or 0
    if not per_token:
        return None
    headroom = max(vram_free_bytes * VRAM_HEADROOM_FRACTION, VRAM_HEADROOM_MIN)
    budget = vram_free_bytes - headroom - recurrent
    if budget <= 0:
        return 0
    return int(budget / per_token)


def derive_recommendations(profile: HardwareProfile,
                           meta: GgufMetadata | None) -> Recommendations:
    """Derive what can be derived; leave the rest to the benchmark."""
    rec = Recommendations()
    why = rec.rationale

    reserve = ram_reserve_bytes(profile.memory.total_bytes)
    rec.ram_reserve_bytes = reserve
    why["ram_reserve_bytes"] = (
        f"{reserve / GIB:.1f} GiB held back for OS, page cache, toolchain and tests"
        + (" (swap here is zram — compressed RAM, not extra capacity)"
           if profile.memory.swap_is_zram else "")
    )

    # A second concurrent request was measured to roughly halve generation
    # throughput on a single GPU. Raising this requires evidence, not optimism.
    rec.parallel_agents = 1
    why["parallel_agents"] = (
        "single GPU, single model: concurrent slots share the same compute and "
        "measurably reduce per-request throughput; raise only if a benchmark disagrees"
    )

    if profile.cpu.physical_cores:
        # Physical cores only. Hyperthreads contend for the same execution
        # units, and llama.cpp's matmul threads gain little from them.
        rec.threads = profile.cpu.physical_cores
        why["threads"] = (
            f"{profile.cpu.physical_cores} physical cores; hyperthreads share "
            "execution units and typically do not help matmul throughput"
        )

    if meta:
        vram_free = 0
        for gpu in profile.gpus:
            if gpu.vram_total_bytes:
                vram_free = max(vram_free,
                                gpu.vram_total_bytes - (gpu.vram_used_bytes or 0))
        if vram_free:
            ceiling = max_context_for_vram(meta, vram_free, "f16")
            if ceiling:
                why["context_length"] = (
                    f"KV cache fits up to ~{ceiling // 1024}k tokens at f16 in "
                    f"{vram_free / GIB:.1f} GiB free VRAM; the useful size is "
                    "smaller and must come from the benchmark, since prompt "
                    "processing degrades with context length before memory does"
                )
        # Deliberately left unset: context_length, max_output_tokens,
        # gpu_layers and batch_size all depend on measurements this module
        # does not have.
        why.setdefault("context_length", "requires benchmark: run 'harness benchmark'")
        why["gpu_layers"] = "requires flag sweep: offload split interacts with fused kernels"
        why["batch_size"] = "requires benchmark: prompt-processing throughput is the target"
        why["max_output_tokens"] = "requires benchmark: derived from context budget"

    return rec


async def build_profile(model_path: str | Path | None = None,
                        storage_paths: list[str] | None = None) -> HardwareProfile:
    """Probe the whole environment and assemble a profile.

    When ``model_path`` is omitted, the model is taken from a detected
    ``llama-server`` that reports its own ``model_path``.
    """
    hw = probe_hardware(storage_paths)
    runtimes = await probe_runtimes()

    profile = HardwareProfile(
        created_at=datetime.now(UTC),
        hostname=hw["hostname"],
        kernel=hw["kernel"],
        os_release=hw["os_release"],
        cpu=hw["cpu"],
        memory=hw["memory"],
        gpus=hw["gpus"],
        storage=hw["storage"],
        runtimes=runtimes,
    )

    resolved = Path(model_path) if model_path else _model_path_from(runtimes)
    meta: GgufMetadata | None = None
    if resolved and resolved.is_file():
        try:
            meta = read_gguf_metadata(resolved)
            profile.model = model_info_from_gguf(meta)
        except GgufError:
            profile.model = ModelInfo(path=resolved)

    profile.recommended = derive_recommendations(profile, meta)
    return profile


def _model_path_from(runtimes: list[RuntimeInfo]) -> Path | None:
    """Take the model path a running server reports, or parse it from its argv."""
    for rt in runtimes:
        if rt.model_path:
            return Path(rt.model_path)
    for rt in runtimes:
        argv = rt.server_argv
        for i, arg in enumerate(argv):
            if arg in ("-m", "--model") and i + 1 < len(argv):
                return Path(argv[i + 1])
    return None


def save_profile(profile: HardwareProfile, path: str | Path) -> Path:
    """Persist the profile as JSON, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(profile.model_dump_json(indent=2) + "\n")
    return target


def load_profile(path: str | Path) -> HardwareProfile:
    """Load a previously saved profile."""
    return HardwareProfile.model_validate(json.loads(Path(path).read_text()))
