# Discovery Report

Every number below was measured on the target machine on 2026-08-20.
Nothing here is estimated, inherited from a spec sheet, or assumed.
Where a fact could not be established, it says so.

## 1. Hardware

| Component | Value | Source |
|---|---|---|
| CPU | Intel Core i7-7700 @ 3.60 GHz (Kaby Lake, 2017) | `lscpu` |
| Cores / Threads | 4 physical / 8 logical | `lscpu` |
| SIMD | AVX2 — **no AVX-512** | `lscpu` |
| L3 cache | 8 MiB (single instance) | `lscpu` |
| RAM | 46 GiB usable (48 GB nominal) | `free -h` |
| Swap | zram0, 46.9 GiB, **compressed RAM — not disk** | `swapon --show` |
| dGPU | NVIDIA GeForce GTX 1060 6 GB (GP106) | `lspci`, `nvidia-smi` |
| Driver / CUDA | 580.173.02 / CUDA 13.0 | `nvidia-smi` |
| iGPU | Intel HD Graphics 630 | `lspci` |
| OS | CachyOS, kernel 7.1.6-1-cachyos | `uname -a` |

**Storage layout** (matters — the model file is 18 GB):

| Mount | Size | Free |
|---|---|---|
| `/` | 220 G | 37 G |
| `/mnt/disk1` (holds the GGUF) | 440 G | 356 G |
| `/mnt/work` (holds projects) | 233 G | 188 G |
| `/mnt/brain` | 932 G | 512 G |

**The zram detail is load-bearing.** Swap here is compressed RAM, not disk.
A benchmark that "fits" by pushing 8 GB into zram has not found more memory —
it has traded memory for CPU time on a 4-core CPU that is already the
bottleneck. Every benchmark in this project records zram usage before, during,
and after, and marks a run **invalid** if it grew.

## 2. Inference runtime

The model is served by `llama-server` taken from an LM Studio backend bundle,
started manually — not by LM Studio itself.

```
/home/alex/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.29.0/llama-server
```

- Build string reported by `/props`: `b1-dd1ea52` (LM Studio's own versioning,
  not an upstream llama.cpp `b####` number)
- Requires CUDA **11.8** shared libraries, vendored at
  `extensions/backends/vendor/linux-llama-cuda-vendor-v1/`, despite the host
  driver providing CUDA 13.0. Any tooling that launches this binary must set
  `LD_LIBRARY_PATH` to both the backend directory and the vendor directory,
  or it fails with `libcudart.so.11.0: cannot open shared object file`.
- Endpoint: `http://127.0.0.1:18080`, OpenAI-compatible, `--jinja` enabled.
- Newest locally available backend: **2.29.1** (CUDA). Also present: vulkan and
  cpu-only variants down to 2.14.4.

Other runtimes on the box, deliberately **not** used as the primary target:

| Service | Port | Status |
|---|---|---|
| LM Studio | 1234 | responds `401` — API key protected |
| Ollama | 11434 | responds `200`, but its catalogue is almost entirely `:cloud` models |

Ollama is additionally unsuitable as a fallback for this model: its vendored
llama.cpp fork did not carry the `qwen35moe` architecture
([ollama#15898](https://github.com/ollama/ollama/issues/15898)).

### Current launch flags

```
-m /mnt/disk1/DL/Hermes3.6-35B-A3B-Uncensored-Genesis-V7-MTP-APEX-Compact.gguf
--chat-template-file /mnt/brain/Modelle/LuffyTheFox/Hermes-V7/chat_template.jinja
--host 127.0.0.1 --port 18080 --no-webui --jinja --alias hermes
--ctx-size 65536 --n-gpu-layers 40 --n-cpu-moe 40 --threads 4
--batch-size 2048 --ubatch-size 512
--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn auto --no-mmap
--metrics --slots
```

## 3. The model is not what its filename says

The file is named `Hermes3.6-35B-A3B-Uncensored-Genesis-V7-MTP-APEX-Compact.gguf`.
Its own metadata disagrees:

```
general.architecture  = qwen35moe
general.name          = Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
general.size_label    = 35B-A3B
general.finetune      = 3Ref
general.basename      = KL0.0764
```

This is a **Qwen3.6-35B-A3B derivative**, not a Hermes model. `qwen35moe` is
llama.cpp's internal architecture identifier (registered in `src/llama-arch.cpp`
with an implementation in `src/models/qwen35moe.cpp`) — it is not the model
generation. The `KL0.0764` basename suggests a KL-divergence-targeted merge or
prune against a reference model.

Treating it as a Hermes model — Hermes prompt conventions, Hermes tool-call
format, Hermes sampler defaults — would be a category error. This project
targets the Qwen3.6-35B-A3B family.

### Architecture: MoE **and** state-space hybrid

```
block_count                 41      (40 trunk layers + 1 MTP block)
nextn_predict_layers        1
expert_count                256
expert_used_count           8       -> ~3B active parameters
expert_feed_forward_length  512
context_length              262144
embedding_length            2048
attention.head_count        16
attention.head_count_kv     2       -> 8:1 GQA
attention.key_length        256     -> head_dim 256, not the usual 128
attention.value_length      256
rope.freq_base              10000000.0
rope.dimension_sections     [11, 11, 10, 0]   (M-RoPE; no vision projector present)
full_attention_interval     4
ssm.conv_kernel             4
ssm.state_size              128
ssm.group_count             16
ssm.time_step_rank          32
ssm.inner_size              4096
```

`full_attention_interval = 4` is decisive. In llama.cpp's implementation,
`is_recr[i] = (i < n_layer) && ((i+1) % full_attn_interval != 0)`. Of the 40
trunk layers, **10 use dense attention and 30 are recurrent Gated DeltaNet
layers**.

Consequences that shape every later decision:

- Only 10 layers carry a context-proportional KV cache. The other 30 hold a
  **fixed-size** recurrent state, independent of context length. This is why a
  65k context fits on a 6 GB card at all.
- KV cache sizing cannot be computed with the usual
  `layers × heads × head_dim × 2` formula. Using it overestimates by ~4×.
- `head_dim = 256` is double the common value and doubles per-token KV cost in
  the layers that do have one.

**Quantisation:** `general.file_type = 15` (Q4_K_M), imatrix-calibrated from
`groups_merged.txt`, 510 entries over 93 chunks. 753 tensors, GGUF v3, 18 GB.

**Tokenizer:** `gpt2` pre-tokenizer variant `qwen35`, vocabulary ≈ 248k,
EOS `248046`, BOS/PAD `248044`.

**Embedded sampler defaults** (the author shipped them inside the GGUF):
`top_k = 20`, `top_p = 0.95`, `temp = 1.0`.

## 4. Measured performance

All figures from the server's own `print_timing` output and from a controlled
`/completion` probe.

### Throughput

| Condition | Result |
|---|---|
| Generation, single slot | **16.5 tok/s** |
| Generation, second slot active concurrently | **8.55 tok/s** |
| Prompt processing, 2k–6k tokens | ~170 tok/s |
| Prompt processing, 16k tokens | 154 tok/s |
| Prompt processing, 18k+ tokens | ~120 tok/s |
| Prompt processing, cold 4816-token probe | 194 tok/s |

The second row settles a design question empirically: **a second concurrent
slot roughly halves generation throughput.** Local parallelism on this hardware
does not buy concurrency, it buys queueing with extra steps. The harness runs
`parallel_model_requests = 1`.

### Prompt cache — the single most important measurement

llama.cpp issue [#20225](https://github.com/ggml-org/llama.cpp/issues/20225)
reports that hybrid attention/recurrent models must reprocess the entire prompt
every turn, because the recurrent state cannot be partially rewound. If true
here, a 16k agent context would cost ~106 s **per step**.

Measured against this server:

| Run | Tokens reprocessed | Wall clock |
|---|---|---|
| 1 — cold, 4816 tokens | 4816 | **24.83 s** |
| 2 — byte-identical prompt | 4 | **0.49 s** |
| 3 — same prefix + 11 new tokens | 11 | **0.82 s** |
| 4 — same prefix + 22 new tokens | 11 | **0.89 s** |

**Incremental prompt caching works, at roughly 50× speedup.** The pathological
case does not apply to this build.

This inverts the naive reading of "adaptive context management":

> Context compression is not free. It costs a full prefix reprocess.

Appending to the context is nearly free. Rewriting anything early in it costs
25 s at 4.8k tokens, ~106 s at 16k. At 16.5 tok/s generation, an 8k reprocess
is worth about **825 discarded output tokens**. Compression must therefore be
*budgeted*, not applied on a timer — see [CONTEXT.md](../CONTEXT.md).

This also explains a live observation: an `opencode` process configured with
`compaction: {auto: true, prune: true}` took **over 6 minutes** on the prompt
*"How many files are here? Use a tool, answer with a number."* Automatic
compaction rewrites the prefix on every step and destroys the cache each time.

### Resource usage

| Metric | Value |
|---|---|
| Peak RSS (`VmHWM`) | 15.5 GiB |
| VRAM | 2757–2831 MiB of 6144 MiB |
| RAM available with server loaded | 19–23 GiB |
| zram in use during observation | 4.9 GiB → 7.3 GiB |

VRAM sits at under half the card. With MoE weights forced to CPU by
`--n-cpu-moe 40`, the GPU is largely idle capacity — which the flag sweep will
attempt to reclaim.

## 5. Three defects found in the running configuration

### 5.1 The MTP block is loaded and discarded

The filename advertises MTP, and the tensors are genuinely present:

```
blk.40.nextn.eh_proj.weight            (4096, 2048)
blk.40.nextn.enorm.weight              (2048)
blk.40.nextn.hnorm.weight              (2048)
blk.40.nextn.shared_head_norm.weight   (2048)
```

The server log shows the entire block being dropped:

```
W model has unused tensor blk.40.ffn_down_exps.weight (size = 285212672 bytes) -- ignoring
W model has unused tensor blk.40.ffn_gate_exps.weight (size = 285212672 bytes) -- ignoring
W model has unused tensor blk.40.ffn_up_exps.weight   (size = 285212672 bytes) -- ignoring
W model has unused tensor blk.40.nextn.eh_proj.weight (size = 8912896 bytes)   -- ignoring
...
```

That is roughly **900 MB read from disk and thrown away**.

Upstream llama.cpp gained native MTP via `--spec-type draft-mtp`
([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)); loading is
gated behind `mparams.load_mtp`, so without the flag the tensors are skipped.
`src/models/qwen35moe.cpp` builds a real execution graph from them and asserts
`n_layer_nextn == 1` — exactly this model's configuration.

**However:** `--spec-type` is absent from the `--help` output of both locally
available backends, 2.29.0 and 2.29.1. Native MTP is unreachable through LM
Studio bundles. What *is* available:

- `--spec-draft-model` / `-md` — classic draft model. Requires matching
  `vocab_type`, BOS/EOS, and vocabulary size within 128 tokens
  (`SPEC_VOCAB_MAX_SIZE_DIFFERENCE` in `common/speculative.cpp`). The
  `gemma-drafter/assistant-Q8_0.gguf` present on this machine has a different
  vocabulary and **cannot** serve as a draft model here.
- `--spec-ngram-*` — n-gram speculation, no draft model required. Promising for
  coding work, where file paths, identifiers and boilerplate repeat heavily
  within the context.

### 5.2 The Gated DeltaNet fused kernel is disabled

```
W resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
  (chunked) is assigned to device CUDA0 (usually due to missing support)
W resolve_fused_ops: fused Gated Delta Net (chunked) not supported, set to disabled
```

`--n-cpu-moe 40` pushes MoE weights of all 40 layers to CPU, placing layer 0
there. The fused GDN kernel then cannot be scheduled. Since 30 of 40 layers
*are* Gated DeltaNet, this disables the fast path of the model's dominant
component. Recovering it — likely via a targeted `--override-tensor` regex that
keeps layer 0 on GPU instead of the blanket `--n-cpu-moe` — is the largest
single lever identified.

### 5.3 Minor issues

- `--no-mmap` is deprecated in favour of `-lm/--load-mode`.
- CORS is open to `*` with no API key, on a server reachable from the box.
- The server explicitly suggests `--reasoning-preserve`; the chat template
  advertises `supports_preserve_reasoning`, but the flag is not set.
- `--metrics` is passed, yet `/metrics` returned empty during observation.

## 6. Capabilities confirmed at the runtime level

From `/props` `chat_template_caps`:

```
supports_tool_calls           true
supports_parallel_tool_calls  true
supports_object_arguments     true
supports_system_role          true
supports_preserve_reasoning   true
```

Two independent paths to structured output exist, both **hard sampling
constraints** rather than prompt hints:

1. `response_format: {"type": "json_schema", "schema": {...}}` on
   `/v1/chat/completions`, and `json_schema` / `grammar` (GBNF) on `/completion`.
   The schema constrains sampling and is *not* injected into the prompt.
2. Native tool calls under `--jinja`, constrained by a lazy grammar
   (`grammar_lazy` / `grammar_triggers` in `common/chat.h`) that activates once
   a tool-call trigger appears in the output.

Which of these the model actually complies with more reliably is an empirical
question, answered by the capability benchmark rather than assumed.

## 7. Known gaps

- Sampler recommendations for Qwen3.6 specifically are **not** officially
  published. Qwen3 model cards give thinking `temp 0.6 / top_p 0.95 / top_k 20 /
  min_p 0` and non-thinking `temp 0.7 / top_p 0.8 / top_k 20 / min_p 0`; HF
  discussions note inconsistencies across 3.6 variants. The GGUF's own embedded
  defaults sit closer to the thinking profile. Treated as a benchmark dimension.
- Whether the fused GDN kernel can be recovered under any MoE-offload
  configuration on a 6 GB card is unverified — it is the first sweep question.
- KV cache quantisation quality at `head_dim = 256` is undocumented. Symmetric
  K/V types are required for the fused flash-attention path
  ([discussion #22411](https://github.com/ggml-org/llama.cpp/discussions/22411));
  asymmetric types fall back to a slower path **without warning**.
- `/metrics` returned empty despite `--metrics`; cause not yet established.
