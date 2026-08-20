# Architecture

Design derived from measurement, not from a reference diagram. Every structural
choice below traces back to a number in [DISCOVERY.md](DISCOVERY.md).

## Module layout

```
src/harness/
├── discovery/     hardware, runtime and model probing -> hardware-profile.json
├── models/        ModelProvider interface + LlamaCppProvider + FakeProvider
├── runtime/       planned: llama-server supervisor (not implemented)
├── context/       PromptAssembler, TokenBudget, CacheEconomics, compressors
├── agent/         AgentLoop, roles, Planner, TaskState, RetryPolicy
├── protocol/      ActionCodec (native tool_calls | constrained JSON), schemas
├── tools/         registry, typed tools, ToolResult compression
├── memory/        SQLite: task state, run journal, persistent facts
├── retrieval/     planned: Retriever interface (not implemented)
├── security/      command classification and approval gate
├── telemetry/     structured run log, no CoT, no secrets
├── benchmark/     planned: capability and task benchmarks (not implemented)
└── cli.py         Typer entry point
```

No `agent.py` holding the whole system. Each module owns one decision.

## The prefix contract

This is the central mechanism. Everything else is arranged around it.

The prompt is built in two zones:

```
┌─────────────────────────────────────────┐
│ STABLE PREFIX  — byte-identical for the │  cached, reprocessed only
│ entire run                              │  when deliberately invalidated
│   · system instructions                 │
│   · tool schemas                        │
│   · repository map                      │
│   · task statement                      │
├─────────────────────────────────────────┤
│ APPEND ZONE  — grows, never rewritten   │  ~0.8 s per step regardless
│   · role directive for this step        │  of how large the prefix is
│   · step history (actions + results)    │
│   · retrieved context for this step     │
└─────────────────────────────────────────┘
```

`PromptAssembler` guarantees the prefix is reproduced byte-for-byte on every
call and raises if it is not. A prefix change is an explicit, logged, budgeted
event — never a side effect.

**Role switching happens in the append zone.** Planner, coder, tester and
reviewer share one system prompt; the role is a directive appended at the
current position. A role-specific system prompt would invalidate the prefix and
cost ~50 s at 8k context, roughly 825 generated tokens, on every handover.

### Cache economics

Compression is a purchase with a known price. `CacheEconomics` decides:

```
reprocess_cost  = new_prefix_tokens / pp_rate
saving_per_step = freed_tokens / pp_rate           # only if prefix shrinks
remaining_steps = max_steps - current_step

compress if  saving_per_step * remaining_steps > reprocess_cost * safety_margin
```

`pp_rate` is not a constant — it degrades with context size (194 tok/s cold at
4.8k, ~120 tok/s past 18k) and is read from the measured profile, interpolated
at the current size.

When the budget is exhausted and compression does not amortise, the loop does
**not** compress anyway. It finishes the run, writes the task state, and reports
that the budget was reached. A truthful stop beats an expensive lie.

## Agent loop

```
   task
     │
     ▼
  normalise goal ──► TaskState (SQLite, resumable)
     │
     ▼
┌──► build context ──► model call ──► validate action
│         │                               │
│    (append only)                   invalid? ──► repair or re-ask
│                                          │        (bounded)
│                                          ▼
│                                    execute tool
│                                          │
│                                    compress result
│                                          │
│                                    append to context
│                                          │
│                                    update TaskState
│                                          │
└─────────────── progress? ◄────────────────┘
     │
     ▼
  verify (evidence required)
     │
     ▼
  finish
```

Hard bounds, all configurable, none optional: `max_steps`, `wall_clock_timeout`,
`token_budget`, `tool_budget`, `retry_budget`, `stop_conditions`.

Every step is journaled to SQLite before it executes. A killed run resumes from
the last committed step rather than starting over — which matters when a single
cold prompt costs 25 s.

Writing the step first is what makes resume correct, but it also means a step
left at `RUNNING` is ambiguous: the process may have died before the tool ran,
or after its side effect and before the checkpoint. The history cannot tell
those apart, so resume does not pretend to. A model call or a tool declared
`SideEffect.NONE`/`IDEMPOTENT` is marked `FAILED` and may simply be repeated;
anything mutating becomes `UNCERTAIN`, is reported on `RunResult`, and the
first identical repeat is refused with `uncertain_side_effect` so the model
verifies the world before applying the effect a second time. Tools that do not
declare a class count as mutating.

## Action protocol

Two codecs behind one interface, because the runtime supports both and only
measurement can say which this model obeys more reliably.

```python
class ActionCodec(Protocol):
    def render_tools(self, tools: list[ToolSpec]) -> PromptFragment: ...
    def request_kwargs(self, tools: list[ToolSpec]) -> dict: ...
    def parse(self, response: ModelResponse) -> Action | ParseError: ...
```

- **`NativeToolCallCodec`** — `--jinja` tool calls. The runtime reports
  `supports_tool_calls`, `supports_parallel_tool_calls` and
  `supports_object_arguments` as true, and constrains them with a lazy grammar.
- **`ConstrainedJsonCodec`** — a single JSON object in the content, enforced by
  `response_format: {"type": "json_schema", ...}`. This is a hard sampling
  constraint, not a prompt hint; the schema is not injected into the prompt.

The action vocabulary stays minimal — YAGNI applies. Shipping in v1:

```json
{"action": "tool",   "tool": "read_file", "arguments": {...}, "reason": "..."}
{"action": "answer", "content": "...", "evidence": [{"kind": "test", "step_id": 12}]}
```

`plan` is not a model action; planning is a role that emits a `TaskState`
update through the same tool channel. `delegate` is not an action because roles
are sequential. `request_context` is not an action because retrieval is a tool.
Actions get added when a benchmark shows their absence costs something.

Validation is Pydantic. A malformed response is repaired once from the parse
error, then re-asked with the error as feedback, then counted against the retry
budget. Retries never resend an identical prompt — that is the definition of
expecting different output from the same input.

## Verification

Claims of the form *implemented*, *fixed*, *tests pass* are rejected unless
matching evidence exists:

| Claim | Required evidence |
|---|---|
| file written | file exists, mtime advanced, content hash differs |
| patch applied | cited path differs from its persisted run-start fingerprint |
| tests pass | cited current-run step is a classified test command with exit code 0 |
| builds | cited current-run step is a classified build command with exit code 0 |
| lint / typecheck clean | cited current-run step has the matching command class and exit code 0 |

`Verifier` runs the checks the task type demands. Unverified claims are
downgraded to *reported but unverified* in the run summary, never silently
accepted.

## Tool output compression

Raw tool output does not enter the context. `ToolResult` carries a compressed
view; the full output is written to the run directory and addressable by id.

```
exit code · stderr (capped) · matched/error lines · file paths ·
head and tail windows · line count elided
```

The uncompressed output stays retrievable, so nothing is lost — it simply is not
paid for on every subsequent step.

## Security boundary

The model's willingness to run a command is not an input to the decision. The
boundary is deterministic code.

Three classes, resolved by pattern before execution:

- **allow** — `git status`, `git diff`, `git log`, `ls`, `cat`, `rg`, `grep`,
  `pytest`, `npm test`, and similar read-only development commands. No prompt.
- **confirm** — anything writing outside the workspace, network access,
  package installation, `git reset --hard`, `git clean -fdx`, `git push`.
- **deny** — `rm -rf /`, `mkfs`, `dd` to a block device, `wipefs`, `shutdown`,
  `reboot`, fork bombs, `DROP DATABASE`. Not confirmable; refused.

Deny wins over allow. Unclassified commands land in **confirm**, not allow —
an unknown command is not a safe command. File tools resolve symlinks before
enforcing workspace containment. Shell operands are checked against the same
boundary; on Linux, allowed commands additionally run in a bubblewrap
filesystem sandbox with the workspace writable and HOME absent. Network tools
remain confirmation-gated; network namespace creation is environment-dependent
and is not claimed as an unconditional sandbox property.

The model name containing *Uncensored* changes nothing here. The security
boundary was never the model.

## Runtime supervision

Runtime supervision is planned and not implemented. Today the provider attaches
to an already-running `llama-server`; it does not launch or reconfigure it.

The planned supervisor must handle the environment quirk discovered on this machine: LM Studio's CUDA
backends link against CUDA 11.8 while the host driver provides 13.0, so both the
backend directory and `vendor/linux-llama-cuda-vendor-v1` must be on
`LD_LIBRARY_PATH` or the binary will not start.

The planned scope covers: launch with a profile, readiness polling (the server answers
`503 Loading model` before it is ready), health checks, graceful shutdown before
a reconfiguration, and refusal to start a profile whose projected memory
footprint exceeds available RAM minus reserve.

Attaching to an already-running server is the only implemented mode today.

## Provider interface

```python
class ModelProvider(Protocol):
    async def generate(self, req: GenerationRequest) -> ModelResponse: ...
    async def stream(self, req: GenerationRequest) -> AsyncIterator[Chunk]: ...
    async def health(self) -> HealthStatus: ...
    async def model_info(self) -> ModelInfo: ...
```

`LlamaCppProvider` is implemented against the detected runtime. `FakeProvider`
replays scripted responses so the entire loop, protocol and tool layer are
testable without the 18 GB model — CI never needs it. Real-model tests are
marked `local_llm` and excluded by default.

Other providers are not written until something needs them.

## Memory

SQLite, two levels, no vector database.

- **Working memory** — goal, plan, step history, findings, open problems.
  Scoped to one run, resumable, discarded on completion except the summary.
- **Persistent memory** — architecture notes, conventions, decisions, known
  issues. Explicitly written, never harvested automatically from transcripts.

Retrieval adapters are planned and not implemented. Persistent facts already
use SQLite with optional FTS5; a future retriever may build on that only after
benchmarks justify it. No vector database is present.

## Telemetry

One structured JSONL journal per run: `run_id`, step index, role, latency,
prompt/completion tokens, cache hit tokens, context size, prefix hash, tool
name, tool duration, exit code, retry count, error class.

Never logged: chain-of-thought content, file contents, environment variables,
anything matching the secret patterns. The journal is what makes a benchmark
reproducible and a failure diagnosable; it is not a transcript archive.
