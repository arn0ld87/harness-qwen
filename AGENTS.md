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
| `src/harness/runtime/` | Planned; not implemented |
| `src/harness/context/` | `PromptAssembler`, `TokenBudget`, `CacheEconomics`, compressors |
| `src/harness/agent/` | `AgentLoop`, roles, `Planner`, `TaskState`, `RetryPolicy` |
| `src/harness/protocol/` | `ActionCodec` (native tool_calls or constrained JSON), schemas |
| `src/harness/tools/` | Registry, typed tools, `ToolResult` compression |
| `src/harness/memory/` | SQLite: task state, run journal, persistent facts |
| `src/harness/retrieval/` | Planned; not implemented |
| `src/harness/security/` | Command classification (allow/confirm/deny) and approval gate |
| `src/harness/telemetry/` | Structured run log (no CoT, no secrets) |
| `src/harness/benchmark/` | Planned; not implemented |
| `src/harness/cli.py` | Typer entry point |

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
