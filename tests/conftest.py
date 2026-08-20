"""Shared fixtures.

The GGUF builder here lets the parser be tested against files whose contents
are known exactly, so no test depends on the 18 GB model being present.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

GGUF_MAGIC = b"GGUF"
T_UINT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64 = 4, 6, 8, 9, 10


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, type_tag: int, value: Any) -> bytes:
    out = _string(key) + struct.pack("<I", type_tag)
    if type_tag == T_STRING:
        return out + _string(value)
    if type_tag == T_UINT32:
        return out + struct.pack("<I", value)
    if type_tag == T_FLOAT32:
        return out + struct.pack("<f", value)
    if type_tag == T_UINT64:
        return out + struct.pack("<Q", value)
    if type_tag == T_ARRAY:
        elem_type, items = value
        body = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
        for item in items:
            body += struct.pack("<I", item) if elem_type == T_UINT32 else _string(item)
        return out + body
    raise ValueError(f"unsupported type tag {type_tag}")


def build_gguf(path: Path, kv: dict[str, tuple[int, Any]],
               tensors: list[str] | None = None) -> Path:
    """Write a syntactically valid GGUF file with no tensor payload."""
    tensor_names = tensors or []
    body = b""
    for key, (type_tag, value) in kv.items():
        body += _kv(key, type_tag, value)

    tensor_block = b""
    for offset, name in enumerate(tensor_names):
        tensor_block += _string(name)
        tensor_block += struct.pack("<I", 2)          # 2 dimensions
        tensor_block += struct.pack("<QQ", 16, 16)    # dims
        tensor_block += struct.pack("<I", 0)          # dtype F32
        tensor_block += struct.pack("<Q", offset * 1024)

    header = (
        GGUF_MAGIC
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensor_names))
        + struct.pack("<Q", len(kv))
    )
    path.write_bytes(header + body + tensor_block)
    return path


@pytest.fixture
def hybrid_moe_gguf(tmp_path: Path) -> Path:
    """A GGUF shaped like the target model: hybrid layers, MoE, one MTP block."""
    return build_gguf(
        tmp_path / "hybrid.gguf",
        {
            "general.architecture": (T_STRING, "qwen35moe"),
            "general.name": (T_STRING, "Test-Hybrid-MoE"),
            "general.file_type": (T_UINT32, 15),
            "general.size_label": (T_STRING, "35B-A3B"),
            "general.sampling.top_k": (T_UINT32, 20),
            "general.sampling.temp": (T_FLOAT32, 1.0),
            "qwen35moe.block_count": (T_UINT32, 41),
            "qwen35moe.nextn_predict_layers": (T_UINT32, 1),
            "qwen35moe.full_attention_interval": (T_UINT32, 4),
            "qwen35moe.context_length": (T_UINT32, 262144),
            "qwen35moe.embedding_length": (T_UINT32, 2048),
            "qwen35moe.attention.head_count": (T_UINT32, 16),
            "qwen35moe.attention.head_count_kv": (T_UINT32, 2),
            "qwen35moe.attention.key_length": (T_UINT32, 256),
            "qwen35moe.attention.value_length": (T_UINT32, 256),
            "qwen35moe.expert_count": (T_UINT32, 256),
            "qwen35moe.expert_used_count": (T_UINT32, 8),
            "qwen35moe.ssm.state_size": (T_UINT32, 128),
            "qwen35moe.ssm.inner_size": (T_UINT32, 4096),
        },
        tensors=[
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.39.attn_q.weight",
            "blk.40.nextn.eh_proj.weight",
            "blk.40.nextn.enorm.weight",
        ],
    )


@pytest.fixture
def dense_gguf(tmp_path: Path) -> Path:
    """A conventional dense model: no hybrid split, no MoE, no MTP."""
    return build_gguf(
        tmp_path / "dense.gguf",
        {
            "general.architecture": (T_STRING, "llama"),
            "general.name": (T_STRING, "Test-Dense"),
            "general.file_type": (T_UINT32, 1),
            "llama.block_count": (T_UINT32, 32),
            "llama.embedding_length": (T_UINT32, 4096),
            "llama.attention.head_count": (T_UINT32, 32),
            "llama.attention.head_count_kv": (T_UINT32, 8),
        },
        tensors=["token_embd.weight", "blk.0.attn_q.weight"],
    )
