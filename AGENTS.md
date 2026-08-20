# Agent Guidelines for harness-qwen

## Project overview

harness-qwen is a minimal agentic runtime for the Qwen3.6-35B-A3B model running locally on constrained hardware (16.5 tok/s, 6 GB VRAM). It organizes the model's work by splitting reasoning from bookkeeping: the model decides; the harness remembers, retrieves, executes, verifies, and recovers. The design is derived from measurement, not assumption — every structural choice traces back to empirical facts in `docs/DISCOVERY.md`.

## Setup commands

```bash
uv sync                          # Install dependencies
uv run pytest                    # Run tests (excludes local_llm by default)
uv run pytest -m local_llm       # Run tests requiring the 35B model
uv run harness doctor            # Probe hardware and runtime
```

## Module map

| Module | Responsibility |
|--------|---|
| `src/harness/discovery/` | Hardware, runtime, and model probing → `hardware-profile.json` |
| `src/harness/models/` | `ModelProvider` interface, `LlamaCppProvider`, `FakeProvider` |
| `src/harness/config/` | Typed configuration: defaults < profile < file < env < CLI, with provenance |
| `src/harness/diagnostics/` | Pre-flight readiness checks behind `harness doctor` |
| `src/harness/runtime/` | `LlamaServerSupervisor`: start/attach/health/stop, owned vs attached; port verification |
| `src/harness/context/` | `PromptAssembler`, `TokenBudget`, `CacheEconomics`, compressors |
| `src/harness/agent/` | `AgentLoop`, roles, `Planner`, `TaskState`, `RetryPolicy` |
| `src/harness/protocol/` | `ActionCodec` (native tool_calls or constrained JSON), schemas |
| `src/harness/tools/` | Registry, typed tools, shell/filesystem/git exec, `ToolResult` compression, `builtin.py` specs, `retrieval_tool.py` (the `retrieve_facts` tool), internal `_security.py` helpers |
| `src/harness/memory/` | SQLite: task state, run journal, persistent facts (`facts.py`), read-only inspect (`inspect.py`), versioned schema and transactional migrations (`migrations.py`) |
| `src/harness/retrieval/` | `Retriever` interface, `SqliteFtsRetriever` over persistent facts, wired into the loop as the `retrieve_facts` tool and the `RetrieveAgain` compression rung |
| `src/harness/security/` | Command classification (allow/confirm/deny), shell splitting, approval gate |
| `src/harness/telemetry/` | Structured run log (no CoT, no secrets), redaction helpers |
| `src/harness/session.py` | Assembles a runnable `AgentLoop` from configuration |
| `src/harness/cli.py` | Typer entry point — `doctor`, `model-info`, `version` |
| `src/harness/cli_run.py` | `run` command with resume, overrides, `--approve-confirmable` |
| `src/harness/cli_chat.py` | `chat` command: `/status`, `/context`, `/usage`, `/exit` |
| `src/harness/cli_inspect.py` | `config show` and `memory inspect` with `--json` |
| `src/harness/cli_benchmark.py` | `benchmark` command group: `capability`/`flags`/`tasks` runs, `compare`, and `prefix` invariant probe; exit codes for valid/invalid/config/execution-failure |
| `src/harness/benchmark/` | Reproducible capability and performance runs: fingerprint, warmup/measure phases, percentiles, JSON artefact |
| `src/harness/benchmark/prefix_invariant.py` | Drives a real `PromptAssembler` through a scripted step sequence and reports whether the prefix hash held and the cache kept hitting (#22) |

## Code style

- **Python 3.12+** with strict type hints on all function signatures.
- **Pydantic** for all external data structures (schemas, API contracts, persisted state).
- **asyncio** throughout; no threads or sync-only functions in the hot path.
- **Small modules**: no single file over ~500 lines. Each module owns one decision.
- **No unnecessary dependencies**: measure before importing.
- **Imports organized**: stdlib, third-party, local. Alphabetical within groups.

## Testing

- **Default test suite**: `uv run pytest` runs all tests *except* those marked `local_llm`.
- **Tests without the 35B model** use `FakeProvider` with scripted responses. The entire loop, protocol, and tool layer are testable in CI.
- **Tests requiring the local model** carry the `@pytest.mark.local_llm` decorator and are excluded by default.
- **A feature without tests is unfinished.** Every code path must be exercised before merge.
- **FakeProvider scripts** live in `tests/fixtures/model_responses/`. Add scenarios as JSON before writing the test.
- **Capability markers**: `sandbox` marks tests needing a usable bubblewrap on
  the host. They skip where it is absent rather than failing, so select them
  with `-m "sandbox and not local_llm"` when you want to know they actually
  ran. Spell out `not local_llm` every time: `-m` takes a single expression
  and the command line replaces the one in `addopts`, so a bare `-m sandbox`
  quietly re-enables the tests that need the 35B model served.

### Coverage floors

CI runs the suite once with coverage and then `scripts/coverage_gate.py`,
which enforces a floor per package from `[tool.coverage_gate]` in
`pyproject.toml`. Per package rather than one number for the tree: a single
threshold either lets the agent loop rot or blocks work over the llama.cpp
client.

| Package | Floor | Measured 2026-08-20 |
|---|---|---|
| `agent` | 88% | 91.2% |
| `benchmark` | 93% | 95.6% |
| `config` | 95% | 98.3% |
| `diagnostics` | 85% | 86.5% |
| `context` | 80% | 83.5% |
| `memory` | 75% | 92.8% |
| `protocol` | 65% | 68.7% |
| `retrieval` | 97% | 100.0% |
| `runtime` | 90% | 93.2% |
| `security` | 97% | 99.3% |
| `tools` | 40% | 57.3% |
| `security/classifier.py` (file) | 97% | 99.5% |
| `security/shellsplit.py` (file) | 100% | 100% |
| `tools/shell.py` (file) | 85% | 91.5% |
| total | 66% | 82.1% |

Three files carry their own floor because a package average hides its riskiest
member. `tools` is held down by `filesystem.py` and `git.py` having no unit
tests, which says nothing about the bubblewrap boundary in `shell.py`. And in
`security`, `shellsplit.py` decides *which* command the classifier judges: a
gap there moves the subject of the decision, so the classifier can be right and
still release the wrong thing.

Floors start just below the measured baseline — a gate that ships red teaches
everyone to ignore it. Raise them in `pyproject.toml`, no code change needed.
`security` carries the highest floor in the tree rather than the lowest
(#32): this is the module where an untested path is a security claim nobody
checked. `benchmark` sits just under it for the parallel reason (#19) —
every measured claim in this repository comes out of that module, so a gap
there is a number nobody checked.

## Non-negotiables

1. **The stable prompt prefix is sacred.** It is cached and reprocessed only when deliberately invalidated. A prefix change is an explicit, logged, budgeted event — never a side effect or an accidental whitespace change. `PromptAssembler` raises if the prefix does not reproduce byte-for-byte.

2. **No secrets and no chain-of-thought in logs.** The telemetry journal records role, latency, tokens, cache hits, tool names, retry counts, errors. It never records file contents, environment variables, or CoT reasoning. Before logging anything, filter against the secret patterns.

3. **Security is deterministic code, not a model judgment.** Commands are classified by pattern into allow/confirm/deny *before* execution. Unclassified commands land in "confirm", not "allow". The model's willingness to run something is irrelevant to the boundary.

4. **Verification is mandatory.** Claims like "file written", "tests pass", "builds" are rejected unless the task produces matching typed evidence: file state changed since the run baseline, or a cited current-run command step has the matching class and exit code 0. `Verifier` downgrades unverified claims to "reported but unverified" in the summary.

5. **`parallel_model_requests` stays 1.** Measured: a second concurrent slot halves throughput on this hardware. Parallelism means queueing with extra steps. Sub-agents run sequentially as roles (planner, coder, tester, reviewer) sharing one model and one warm cache.

## Commit conventions

- **One logical change per commit.** A fix, a feature, a refactor — not a shopping list.
- **Imperative commit message subject** (50 characters max): "Add context compressor", not "Added" or "Adds".
- **No trailing period** in the subject line.
- **Reference issues and decisions** in the body if they exist.
- **Keep commits small**: 100 lines of code per commit is a healthy target, not a ceiling.
