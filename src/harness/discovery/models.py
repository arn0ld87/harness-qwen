"""Typed results of environment discovery.

Every optional field means "could not be determined on this system", never
"assume a sensible default". Consumers must handle ``None`` explicitly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class CpuInfo(BaseModel):
    model_name: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    has_avx2: bool = False
    has_avx512: bool = False
    l3_cache_bytes: int | None = None


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_is_zram: bool = False
    """zram is compressed RAM, not disk. Spilling into it trades memory for CPU
    time on a machine where the CPU is usually already the bottleneck, so it is
    tracked separately from disk swap."""


class GpuInfo(BaseModel):
    name: str | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None


class StorageInfo(BaseModel):
    mount_point: str
    total_bytes: int
    free_bytes: int


class RuntimeInfo(BaseModel):
    """A local inference endpoint that answered a probe."""

    kind: str
    """One of: llama-server, ollama, lmstudio, openai-compatible, unknown."""
    base_url: str
    reachable: bool
    requires_auth: bool = False
    model_ids: list[str] = Field(default_factory=list)
    build_info: str | None = None
    n_ctx: int | None = None
    total_slots: int | None = None
    model_path: str | None = None
    chat_template_caps: dict[str, bool] = Field(default_factory=dict)
    server_pid: int | None = None
    server_argv: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Facts read from the GGUF file itself, not from its filename."""

    path: Path | None = None
    file_size_bytes: int | None = None
    architecture: str | None = None
    name: str | None = None
    size_label: str | None = None
    quantization: str | None = None

    n_layers: int | None = None
    n_dense_attention_layers: int | None = None
    n_recurrent_layers: int | None = None
    n_mtp_layers: int = 0

    context_length_train: int | None = None
    embedding_length: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    key_length: int | None = None
    value_length: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    full_attention_interval: int | None = None
    ssm_state_size: int | None = None
    ssm_inner_size: int | None = None

    embedded_sampling: dict[str, float] = Field(default_factory=dict)
    """Sampler defaults the model author shipped inside the GGUF."""

    @property
    def is_hybrid(self) -> bool:
        """True when only a fraction of layers carry a growing KV cache."""
        return bool(self.full_attention_interval and self.full_attention_interval > 1)

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_count and self.expert_count > 1)

    @property
    def has_mtp(self) -> bool:
        return self.n_mtp_layers > 0


class Recommendations(BaseModel):
    """Derived suggestions. Populated only where a real measurement backs them."""

    context_length: int | None = None
    max_output_tokens: int | None = None
    parallel_agents: int = 1
    gpu_layers: int | None = None
    batch_size: int | None = None
    threads: int | None = None
    ram_reserve_bytes: int | None = None
    rationale: dict[str, str] = Field(default_factory=dict)
    """Why each value was chosen. A recommendation without a rationale is a guess."""


class HardwareProfile(BaseModel):
    """The complete discovered environment, persisted to config/hardware-profile.json."""

    schema_version: int = 1
    created_at: datetime
    hostname: str | None = None
    os_release: str | None = None
    kernel: str | None = None

    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo] = Field(default_factory=list)
    storage: list[StorageInfo] = Field(default_factory=list)
    runtimes: list[RuntimeInfo] = Field(default_factory=list)
    model: ModelInfo | None = None
    recommended: Recommendations = Field(default_factory=Recommendations)

    @property
    def primary_runtime(self) -> RuntimeInfo | None:
        """The runtime the harness should talk to.

        A reachable llama-server that has a model loaded wins over anything
        else, because it is the only one whose launch flags we can control.
        """
        reachable = [r for r in self.runtimes if r.reachable]
        if not reachable:
            return None
        for r in reachable:
            if r.kind == "llama-server" and r.model_ids:
                return r
        for r in reachable:
            if r.model_ids:
                return r
        return reachable[0]
