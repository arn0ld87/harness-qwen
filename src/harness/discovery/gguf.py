"""Minimal GGUF metadata reader.

Reads the header and key-value block of a GGUF file without loading tensor
data, so inspecting an 18 GB model costs a few milliseconds.

This exists instead of a dependency because the harness needs exactly three
things a generic reader will not give it: the hybrid attention/recurrent layer
split, the presence of multi-token-prediction tensors, and the sampler defaults
the model author embedded. Those drive the memory model in
:mod:`harness.discovery.profile`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGUF value type tags -> (struct format, byte width)
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# llama.cpp file_type tag -> human readable quantisation name.
_FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

# Bytes per element for KV cache types, from llama.cpp block layouts.
# q4_0 stores 32 values in an 18-byte block (16 data + 2 scale).
KV_CACHE_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

# Tensor name fragments that identify a multi-token-prediction block.
_MTP_MARKERS = ("nextn", "eh_proj", "shared_head")

# Metadata keys worth keeping verbatim; tokenizer vocab arrays are skipped
# because they are large and never needed here.
_SKIP_KEY_PREFIXES = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
                      "tokenizer.ggml.merges", "tokenizer.ggml.token_type")


class GgufError(RuntimeError):
    """Raised when a file is not readable as GGUF."""


@dataclass
class TensorEntry:
    name: str
    dims: tuple[int, ...]
    dtype: int


@dataclass
class GgufMetadata:
    """Parsed GGUF header. ``kv`` holds raw metadata, the rest is derived."""

    path: Path
    file_size_bytes: int
    version: int
    tensor_count: int
    kv: dict[str, Any] = field(default_factory=dict)
    mtp_tensors: list[str] = field(default_factory=list)
    block_indices: set[int] = field(default_factory=set)

    @property
    def architecture(self) -> str | None:
        return self.kv.get("general.architecture")

    @property
    def name(self) -> str | None:
        return self.kv.get("general.name")

    def arch_key(self, suffix: str) -> Any:
        """Look up an architecture-prefixed key, e.g. ``qwen35moe.block_count``."""
        arch = self.architecture
        return self.kv.get(f"{arch}.{suffix}") if arch else None

    @property
    def quantization(self) -> str | None:
        ft = self.kv.get("general.file_type")
        return _FILE_TYPES.get(ft, f"unknown({ft})") if ft is not None else None

    @property
    def n_mtp_layers(self) -> int:
        """Multi-token-prediction blocks declared by the model."""
        declared = self.arch_key("nextn_predict_layers")
        if isinstance(declared, int):
            return declared
        return 1 if self.mtp_tensors else 0

    @property
    def n_trunk_layers(self) -> int | None:
        """Transformer layers excluding any MTP block.

        ``block_count`` counts the MTP block, but that block is not part of the
        forward pass unless speculative decoding is explicitly enabled, so it
        must not be included in memory or offload arithmetic.
        """
        total = self.arch_key("block_count")
        if not isinstance(total, int):
            return None
        return total - self.n_mtp_layers

    def attention_layer_split(self) -> tuple[int, int] | None:
        """Return ``(dense_attention_layers, recurrent_layers)``.

        Mirrors llama.cpp: a layer is recurrent when
        ``(i + 1) % full_attention_interval != 0``. Returns ``None`` when the
        model is not hybrid, in which case every trunk layer is dense.
        """
        trunk = self.n_trunk_layers
        if trunk is None:
            return None
        interval = self.arch_key("full_attention_interval")
        if not isinstance(interval, int) or interval <= 1:
            return (trunk, 0)
        dense = sum(1 for i in range(trunk) if (i + 1) % interval == 0)
        return (dense, trunk - dense)

    def embedded_sampling(self) -> dict[str, float]:
        """Sampler defaults shipped inside the file by its author."""
        out: dict[str, float] = {}
        for key, value in self.kv.items():
            if key.startswith("general.sampling.") and isinstance(value, (int, float)):
                out[key.removeprefix("general.sampling.")] = float(value)
        return out

    def kv_cache_bytes_per_token(self, cache_type_k: str = "f16",
                                 cache_type_v: str = "f16") -> float | None:
        """Bytes of KV cache consumed per context token.

        Only dense-attention layers hold a context-proportional cache. Applying
        the usual dense formula to a hybrid model overestimates badly — for a
        model with ``full_attention_interval = 4`` it is off by roughly 4x.
        """
        split = self.attention_layer_split()
        n_kv_heads = self.arch_key("attention.head_count_kv")
        if split is None or not isinstance(n_kv_heads, int):
            return None
        dense_layers = split[0]

        k_len = self.arch_key("attention.key_length")
        v_len = self.arch_key("attention.value_length")
        if not isinstance(k_len, int) or not isinstance(v_len, int):
            n_embd = self.arch_key("embedding_length")
            n_head = self.arch_key("attention.head_count")
            if not isinstance(n_embd, int) or not isinstance(n_head, int) or n_head == 0:
                return None
            k_len = v_len = n_embd // n_head

        bpe_k = KV_CACHE_BYTES_PER_ELEMENT.get(cache_type_k.lower())
        bpe_v = KV_CACHE_BYTES_PER_ELEMENT.get(cache_type_v.lower())
        if bpe_k is None or bpe_v is None:
            return None
        per_layer = n_kv_heads * (k_len * bpe_k + v_len * bpe_v)
        return dense_layers * per_layer

    def recurrent_state_bytes(self, dtype_bytes: int = 4) -> float | None:
        """Fixed memory held by recurrent layers, independent of context length."""
        split = self.attention_layer_split()
        d_state = self.arch_key("ssm.state_size")
        d_inner = self.arch_key("ssm.inner_size")
        if split is None or not isinstance(d_state, int) or not isinstance(d_inner, int):
            return None
        return split[1] * d_state * d_inner * dtype_bytes


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, type_tag: int, *, max_array: int = 16) -> Any:
    if type_tag == _TYPE_STRING:
        return _read_string(f)
    if type_tag == _TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type == _TYPE_STRING:
            kept = [_read_string(f) for _ in range(min(count, max_array))]
            for _ in range(max(0, count - max_array)):
                _read_string(f)
            return kept if count <= max_array else {"_truncated": count, "head": kept}
        if elem_type == _TYPE_ARRAY:
            return {"_nested_array": count}
        fmt, width = _SCALARS[elem_type]
        raw = f.read(width * count)
        values = list(struct.unpack(f"<{count}{fmt[1]}", raw))
        return values if count <= max_array else {"_truncated": count, "head": values[:max_array]}
    fmt, width = _SCALARS[type_tag]
    (value,) = struct.unpack(fmt, f.read(width))
    return value


def read_gguf_metadata(path: str | Path, *, scan_tensors: bool = True) -> GgufMetadata:
    """Read GGUF header metadata without touching tensor data.

    Args:
        path: Path to the ``.gguf`` file.
        scan_tensors: Walk the tensor directory to detect MTP blocks and layer
            indices. Costs a single sequential read of the header region.

    Raises:
        GgufError: If the file is missing, truncated, or not GGUF.
    """
    p = Path(path)
    if not p.is_file():
        raise GgufError(f"not a file: {p}")

    try:
        with p.open("rb") as f:
            if f.read(4) != GGUF_MAGIC:
                raise GgufError(f"missing GGUF magic: {p}")
            (version,) = struct.unpack("<I", f.read(4))
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))

            kv: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (type_tag,) = struct.unpack("<I", f.read(4))
                value = _read_value(f, type_tag)
                if not key.startswith(_SKIP_KEY_PREFIXES):
                    kv[key] = value

            meta = GgufMetadata(
                path=p,
                file_size_bytes=p.stat().st_size,
                version=version,
                tensor_count=tensor_count,
                kv=kv,
            )

            if scan_tensors:
                for _ in range(tensor_count):
                    name = _read_string(f)
                    (n_dims,) = struct.unpack("<I", f.read(4))
                    struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                    struct.unpack("<I", f.read(4))  # dtype
                    struct.unpack("<Q", f.read(8))  # offset
                    lowered = name.lower()
                    if any(marker in lowered for marker in _MTP_MARKERS):
                        meta.mtp_tensors.append(name)
                    parts = name.split(".")
                    if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
                        meta.block_indices.add(int(parts[1]))
            return meta
    except GgufError:
        raise
    except (struct.error, OSError, KeyError, UnicodeDecodeError) as exc:
        raise GgufError(f"failed to parse {p}: {exc}") from exc
