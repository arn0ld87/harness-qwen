# harness-qwen

A local agent harness built specifically for Qwen3.6-35B-A3B on modest hardware.

## What this is

`harness-qwen` is a local agent harness — task loop, context management, tool
execution, verification — built around a single model on a single machine.
The target hardware is an Intel Core i7-7700 (4 cores / 8 threads, 2017),
an NVIDIA GTX 1060 with 6 GB VRAM, and 46 GiB of RAM. On that machine, the
model generates a measured **16.5 tokens per second**. That number is the
hard floor the project is designed around, not an unfortunate edge case to
work past. The harness is tailored to this exact class of hardware — a
single consumer GPU with a 4-core CPU behind it — not to datacenter GPUs or
multi-GPU rigs. If the target hardware changes, large parts of the design
would need to be revisited.

## Why it exists

Three measurements from [docs/DISCOVERY.md](docs/DISCOVERY.md) shape every
later decision, and are worked out in full in [VISION.md](VISION.md):

- **Prompt cache reuse is ~50x faster than reprocessing, and compression
  costs a full reprocess.** A byte-identical 4816-token prompt reprocesses
  in 0.49 s instead of 24.83 s cold; appending new tokens to the end costs
  well under a second. Rewriting anything near the front of the prompt —
  which is what naive context compression does — forces the whole prefix
  back through the model. The harness treats the prompt as append-only by
  default and only compresses when a cost model says the saving is worth it.
- **A second concurrent request roughly halves generation throughput.**
  Measured at 16.5 tok/s with one slot active and 8.55 tok/s with a second
  slot active concurrently. On this hardware, parallel agents are not
  concurrency, they are queueing with extra bookkeeping — so the harness
  runs `parallel_model_requests = 1` and executes roles (planner, coder,
  tester, reviewer) sequentially instead.
- **The model is an attention/Gated DeltaNet hybrid, and its fast path is
  off by default.** Of the 40 trunk layers, 30 are recurrent Gated DeltaNet
  and only 10 use dense attention. In the current launch configuration,
  `--n-cpu-moe 40` places layer 0 on CPU, which disables the fused Gated
  DeltaNet kernel for the model's dominant component. Recovering it is the
  largest single performance lever identified so far.

## Status

Version 1 is in development. Nothing here should be read as finished
software yet.

Implemented:
- Hardware, runtime, and model discovery completed and documented
  (`docs/DISCOVERY.md`), with every figure traced back to a command run on
  the target machine.
- Architecture defined (`docs/ARCHITECTURE.md`): module layout, the prefix
  contract, cache economics, agent loop, action protocol, verification
  rules, and the security boundary.
- Model providers (`LlamaCppProvider`, `FakeProvider`), structured action
  codecs, prompt assembly, cache economics and compression strategies.
- Agent loop with bounded retries, sequential roles, typed evidence,
  workspace-baselined patch verification and SQLite resume state.
- Typed tools, output compression, deterministic command classification,
  bubblewrap filesystem isolation where available, and redacted telemetry.
- CI tests run without a model or GPU.

Planned, not implemented:
- llama-server process supervision (`runtime/`). The current provider attaches
  to an already running endpoint; it does not launch or restart the server.
- Retrieval adapters (`retrieval/`) and benchmark runners (`benchmark/`).
- The task-running, chat, benchmark, config and memory-inspection CLI commands.
- No benchmarks have been run yet. The claim that this harness beats a
  plain prompt loop on real tasks is the point of the project, not an
  assumption — see the "What 'done' means" section of `VISION.md`.

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- An OpenAI-compatible local inference endpoint — `llama-server` is the one
  this project is developed and measured against
- The Qwen3.6-35B-A3B GGUF model file itself (not included in this repo)

**A CUDA version mismatch to expect:** if you run `llama-server` from an LM
Studio backend bundle, it links against CUDA 11.8 shared libraries even when
the host driver provides a newer CUDA (13.0 on the discovery machine).
`LD_LIBRARY_PATH` must include both the backend directory and its
`vendor/linux-llama-cuda-vendor-v1/` subdirectory, or the binary fails to
start with `libcudart.so.11.0: cannot open shared object file`. Details in
[docs/DISCOVERY.md, section 2](docs/DISCOVERY.md#2-inference-runtime).

## Installation

```bash
git clone https://github.com/arn0ld87/harness-qwen.git
cd harness-qwen
uv sync
```

The provider defaults to the running OpenAI-compatible endpoint
`http://127.0.0.1:18080`.

## Usage

Implemented CLI commands:

```
harness doctor              # check hardware, runtime, and model against discovery
harness model-info PATH     # report metadata from a GGUF file
harness version             # print the package version
```

`run`, `chat`, `benchmark`, `config` and `memory inspect` are planned and do
not exist yet.

## Documentation

| File | Contents |
|---|---|
| [VISION.md](VISION.md) | Why this project exists, what it deliberately is not, and the priority order used to resolve design trade-offs. |
| [docs/DISCOVERY.md](docs/DISCOVERY.md) | Every measured hardware, runtime, and model fact this project is built on, with the command used to obtain it. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, the prefix contract, cache economics, agent loop, action protocol, verification, and security boundary — each traced back to a DISCOVERY.md measurement. |
| [CONTEXT.md](CONTEXT.md) | Context zones, token budgets, compression economics and retrieval design. |
| [API.md](API.md) | Public types and provider, protocol, context, tool and security interfaces. |
| [AGENTS.md](AGENTS.md) | Repository rules, module responsibilities and agent-loop invariants. |

## Design principles

From `VISION.md`, in order — when two designs compete, the earlier one wins:

1. reliability
2. quality of agent outcomes
3. RAM / VRAM footprint
4. latency
5. simplicity
6. extensibility
7. theoretical maximum features

## License

MIT.
