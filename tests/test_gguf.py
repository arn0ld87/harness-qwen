"""Tests for the GGUF metadata reader.

The hybrid arithmetic is the part worth testing: applying the usual dense KV
formula to this architecture overestimates memory by roughly the attention
interval, which would push every downstream decision the wrong way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.discovery.gguf import GgufError, read_gguf_metadata


class TestBasicParsing:
    def test_reads_identity_fields(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert meta.architecture == "qwen35moe"
        assert meta.name == "Test-Hybrid-MoE"
        assert meta.quantization == "Q4_K_M"
        assert meta.version == 3

    def test_arch_key_resolves_prefix(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert meta.arch_key("expert_count") == 256
        assert meta.arch_key("does_not_exist") is None

    def test_rejects_non_gguf(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not.gguf"
        bogus.write_bytes(b"XXXX" + b"\x00" * 64)
        with pytest.raises(GgufError, match="magic"):
            read_gguf_metadata(bogus)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(GgufError, match="not a file"):
            read_gguf_metadata(tmp_path / "absent.gguf")

    def test_rejects_truncated_file(self, hybrid_moe_gguf: Path) -> None:
        data = hybrid_moe_gguf.read_bytes()
        hybrid_moe_gguf.write_bytes(data[: len(data) // 2])
        with pytest.raises(GgufError):
            read_gguf_metadata(hybrid_moe_gguf)


class TestLayerAccounting:
    def test_mtp_block_excluded_from_trunk(self, hybrid_moe_gguf: Path) -> None:
        """block_count includes the MTP block, which is not part of the forward pass."""
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert meta.arch_key("block_count") == 41
        assert meta.n_mtp_layers == 1
        assert meta.n_trunk_layers == 40

    def test_mtp_tensors_detected(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert any("nextn" in name for name in meta.mtp_tensors)

    def test_hybrid_split_matches_llama_cpp_rule(self, hybrid_moe_gguf: Path) -> None:
        """A layer is dense when (i + 1) % full_attention_interval == 0."""
        meta = read_gguf_metadata(hybrid_moe_gguf)
        dense, recurrent = meta.attention_layer_split()
        assert (dense, recurrent) == (10, 30)
        assert dense + recurrent == meta.n_trunk_layers

    def test_dense_model_has_no_recurrent_layers(self, dense_gguf: Path) -> None:
        meta = read_gguf_metadata(dense_gguf)
        assert meta.attention_layer_split() == (32, 0)
        assert meta.n_mtp_layers == 0


class TestMemoryArithmetic:
    def test_kv_counts_only_dense_layers(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        per_token = meta.kv_cache_bytes_per_token("f16", "f16")
        # 10 dense layers x 2 kv heads x (256 + 256) x 2 bytes
        assert per_token == pytest.approx(10 * 2 * (256 + 256) * 2)

    def test_naive_dense_formula_would_overestimate(self, hybrid_moe_gguf: Path) -> None:
        """Guards the reason this method exists at all."""
        meta = read_gguf_metadata(hybrid_moe_gguf)
        actual = meta.kv_cache_bytes_per_token("f16", "f16")
        naive = meta.n_trunk_layers * 2 * (256 + 256) * 2
        assert naive == pytest.approx(actual * 4)

    def test_quantised_cache_is_smaller(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        f16 = meta.kv_cache_bytes_per_token("f16", "f16")
        q4 = meta.kv_cache_bytes_per_token("q4_0", "q4_0")
        assert q4 < f16
        assert q4 == pytest.approx(f16 * (18 / 32) / 2)

    def test_unknown_cache_type_returns_none(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert meta.kv_cache_bytes_per_token("q3_nonsense", "f16") is None

    def test_recurrent_state_is_context_independent(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        # 30 recurrent layers x state_size x inner_size x 4 bytes
        assert meta.recurrent_state_bytes() == pytest.approx(30 * 128 * 4096 * 4)

    def test_dense_model_has_no_recurrent_state(self, dense_gguf: Path) -> None:
        meta = read_gguf_metadata(dense_gguf)
        assert meta.recurrent_state_bytes() is None

    def test_falls_back_to_embedding_derived_head_dim(self, dense_gguf: Path) -> None:
        """Models without explicit key_length still get a KV estimate."""
        meta = read_gguf_metadata(dense_gguf)
        per_token = meta.kv_cache_bytes_per_token("f16", "f16")
        head_dim = 4096 // 32
        assert per_token == pytest.approx(32 * 8 * (head_dim * 2 + head_dim * 2))


class TestEmbeddedSampling:
    def test_extracts_author_defaults(self, hybrid_moe_gguf: Path) -> None:
        meta = read_gguf_metadata(hybrid_moe_gguf)
        assert meta.embedded_sampling() == {"top_k": 20.0, "temp": 1.0}

    def test_absent_sampling_yields_empty(self, dense_gguf: Path) -> None:
        assert read_gguf_metadata(dense_gguf).embedded_sampling() == {}
