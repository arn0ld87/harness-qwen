# API Reference

Public interfaces of `harness-qwen`. Types shared across subsystems live in
`harness.core`, which imports from no other harness module and is the bottom of
the dependency graph.

This document describes interfaces, not implementation status. See the README
for what is currently built.

---

## `harness.core`

### Model layer

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None
    tool_call_id: str | None

class GenerationRequest(BaseModel):
    messages: list[Message]
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    stop: list[str]
    seed: int | None
    json_schema: dict | None      # constrains sampling when set
    tools: list[dict] | None      # native tool definitions
    timeout_s: float
    prefix_token_estimate: int | None
```

`json_schema` and `tools` are the two structured-output paths. Both are hard
sampling constraints in llama-server, not prompt hints — the schema is *not*
injected into the prompt.

```python
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
```

`cached_tokens` is load-bearing. The gap between `prompt_tokens` and
`cached_tokens` is exactly what a prefix change cost. A run where that gap is
large on every step has a prefix-stability bug, and this field is how it
becomes visible.

```python
class ModelResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage
    timings: Timings
    reasoning: str | None
```

`reasoning` is never logged and never fed back into context. It exists so the
loop can distinguish reasoning from content when the chat template separates
them.

**Exceptions:** `ProviderError` → `ProviderUnavailable`, `ProviderTimeout`,
`ContextOverflow`.

### Tool layer

```python
class Risk(StrEnum):
    ALLOW = "allow"       # read-only development commands, no prompt
    CONFIRM = "confirm"   # requires explicit approval
    DENY = "deny"         # refused, not confirmable

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict      # JSON Schema for the arguments object
    risk: Risk
    timeout_s: float

    def to_openai_tool(self) -> dict: ...

class ToolResult(BaseModel):
    tool: str
    ok: bool
    content: str              # compressed view; this is what enters context
    exit_code: int | None
    duration_ms: float
    full_output_ref: str | None   # complete output, addressable on disk
    truncated: bool
    original_bytes: int | None
    error_kind: str | None
```

`content` is the compressed view. `full_output_ref` addresses the complete
output, so nothing is lost — it simply is not paid for on every later step.

`error_kind` values: `not_found`, `timeout`, `denied`, `invalid_arguments`,
`execution_failed`, `permission_denied`.

### Protocol layer

```python
class ToolAction(BaseModel):
    action: Literal["tool"]
    tool: str
    arguments: dict
    reason: str

class AnswerAction(BaseModel):
    action: Literal["answer"]
    content: str
    evidence: list[CommandEvidence | FileEvidence]

class CommandEvidence(BaseModel):
    kind: Literal["test", "lint", "typecheck", "build"]
    step_id: int

class FileEvidence(BaseModel):
    kind: Literal["file", "patch"]
    path: str

Action = ToolAction | AnswerAction
```

Two actions, deliberately. `plan` is not a model action — planning is a role
that updates `TaskState` through the tool channel. `delegate` is absent because
roles run sequentially. `request_context` is absent; retrieval is planned as a tool.
Actions are added when a benchmark shows their absence costs something.

```python
class ParseError(BaseModel):
    raw: str
    reason: str
    recoverable: bool

    def as_feedback(self) -> str: ...
```

`as_feedback()` produces the text handed back on retry. Retries must change the
input; echoing the concrete failure is the cheapest way to do that.

### Agent layer

```python
class Role(StrEnum):
    PLANNER, CODER, TESTER, REVIEWER

class Budget(BaseModel):
    max_steps: int
    wall_clock_s: float
    max_output_tokens: int
    max_tool_calls: int
    max_retries: int

    def exhausted(self, *, steps, elapsed_s, tool_calls, retries) -> str | None: ...

class StopReason(StrEnum):
    ANSWERED, BUDGET_EXHAUSTED, NO_PROGRESS, UNRECOVERABLE_ERROR, CANCELLED

class TaskState(BaseModel):    # persisted before each step; resumable
    run_id: str
    goal: str
    workspace: str
    steps: list[PlanStep]
    findings: list[str]
    open_problems: list[str]
    step_index: int
```

Prompt limits belong exclusively to `TokenBudget`; `Budget` owns run-level
step, wall-clock, output, tool and retry bounds.

---

## `harness.models`

```python
class ModelProvider(Protocol):
    async def generate(self, req: GenerationRequest) -> ModelResponse: ...
    async def stream(self, req: GenerationRequest) -> AsyncIterator[Chunk]: ...
    async def health(self) -> HealthStatus: ...
    async def model_info(self) -> RuntimeModelInfo: ...
```

**`LlamaCppProvider`** targets an OpenAI-compatible llama-server.
`health()` maps HTTP 503 `"Loading model"` to
`HealthStatus(reachable=True, loading=True)` — the server answers that way for
roughly 40 seconds after start, and treating it as an error makes startup
unnecessarily fragile.

**`FakeProvider`** replays scripted responses, records every request it
received, can raise on the Nth call to exercise recovery paths, and exposes the
sequence of prompt-prefix hashes so tests can assert the prefix stayed stable
across a run.

---

## `harness.protocol`

```python
class ActionCodec(Protocol):
    def render_tools(self, tools: list[ToolSpec]) -> str: ...
    def request_kwargs(self, tools: list[ToolSpec]) -> dict: ...
    def parse(self, response: ModelResponse) -> Action | ParseError: ...
```

- **`NativeToolCallCodec`** — uses the runtime's native tool calling under
  `--jinja`, constrained by a lazy grammar.
- **`ConstrainedJsonCodec`** — one JSON object in the content, enforced by
  `response_format: {"type": "json_schema", ...}`.

Both ship because the runtime supports both and only measurement can say which
this model obeys more reliably. `parse()` never raises: unparseable output
returns a `ParseError` carrying a specific reason.

---

## `harness.context`

```python
class PromptAssembler:
    def set_prefix(self, *, system, tool_schemas, repo_map, task) -> None: ...
    def append(self, message: Message) -> None: ...
    def build(self, role: Role | None = None) -> tuple[list[Message], str]: ...
    def prefix_hash(self) -> str: ...
    def invalidate(self, reason: str) -> None: ...
```

`build()` returns the messages and the prefix hash. If the prefix changed
without `invalidate()` having been called first, it raises `PrefixViolation` —
turning the system's most expensive silent failure into a loud one.

**Role directives go in the append zone.** A role-specific system prompt would
invalidate the prefix and cost roughly 50 s at 8k context on every handover.

```python
class CacheEconomics:
    def should_compress(self, *, freed_tokens, new_prefix_tokens,
                        remaining_steps, context_size) -> tuple[bool, str]: ...
    def pp_rate_at(self, context_size: int) -> float: ...
```

Returns the decision *and* the arithmetic behind it, so the journal records why
a run did or did not pay for compression. `pp_rate_at` interpolates measured
throughput points, because prompt-processing rate degrades with context size —
it is not a constant.

---

## `harness.tools`

```python
class ToolRegistry:
    def register(self, spec: ToolSpec, fn: Callable) -> None: ...
    def specs(self) -> list[ToolSpec]: ...
    async def execute(self, name: str, arguments: dict) -> ToolResult: ...
```

`execute()` validates arguments against the spec's JSON Schema, enforces
`timeout_s`, and converts every exception into a `ToolResult` with `ok=False`.
An unknown tool name returns `error_kind="not_found"` listing available tools —
a wrong tool name is a recoverable mistake, not a crash.

**Shipped tools:** `read_file`, `write_file`, `list_files`, `search_files`,
`run_command`, `git_status`, `git_diff`, `git_log`.

---

## `harness.security`

```python
def classify_command(command: str) -> tuple[Risk, str]: ...
def resolve_in_workspace(path: str | Path, workspace_root: Path) -> Path: ...
```

Deterministic. The model's willingness to run a command is not an input.

`classify_command` splits on shell metacharacters (`;`, `&&`, `||`, `|`,
newline, backticks, `$(...)`) and classifies **every** segment, returning the
most severe result. `DENY` beats `ALLOW`; anything unrecognised resolves to
`CONFIRM`, because an unknown command is not a safe command.

`resolve_in_workspace` resolves symlinks **before** the containment check and
raises `ToolError(kind="denied")` when the result escapes the root.

---

## `harness.discovery`

```python
async def build_profile(model_path=None, storage_paths=None) -> HardwareProfile: ...
def save_profile(profile, path) -> Path: ...
def load_profile(path) -> HardwareProfile: ...
def read_gguf_metadata(path, *, scan_tensors=True) -> GgufMetadata: ...
```

`GgufMetadata` carries the arithmetic generic readers do not provide:

```python
meta.attention_layer_split()      -> (dense_layers, recurrent_layers) | None
meta.kv_cache_bytes_per_token(k, v) -> float | None
meta.recurrent_state_bytes()      -> float | None
meta.n_trunk_layers               -> int | None    # excludes MTP blocks
meta.n_mtp_layers                 -> int
meta.embedded_sampling()          -> dict[str, float]
```

`kv_cache_bytes_per_token` counts **only dense-attention layers**. Applying the
usual dense formula to a hybrid model overestimates by roughly the attention
interval — a factor of 4 for this one.

`n_trunk_layers` subtracts MTP blocks from `block_count`, because those blocks
are not part of the forward pass unless speculative decoding is explicitly
enabled, and including them corrupts every offload calculation downstream.

---

## Runtime HTTP surface

Endpoints the harness uses on `llama-server`:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | generation, with `response_format` / `tools` |
| `GET /v1/models` | model id discovery |
| `GET /props` | context size, slots, build info, `chat_template_caps` |
| `GET /health` | readiness — answers `503 Loading model` while loading |
| `GET /slots` | per-slot state |

Relevant launch flags: `--ctx-size`, `--n-gpu-layers`, `--n-cpu-moe`,
`--override-tensor`, `--cache-type-k`, `--cache-type-v`, `--flash-attn`,
`--batch-size`, `--ubatch-size`, `--threads`, `--parallel`, `--jinja`,
`--chat-template-file`, `-lm/--load-mode`.

**Environment caveat:** LM Studio's CUDA backends link against CUDA 11.8 even
when the host driver provides 13.0. Both the backend directory and
`extensions/backends/vendor/linux-llama-cuda-vendor-v1` must be on
`LD_LIBRARY_PATH`, or the binary fails with
`libcudart.so.11.0: cannot open shared object file`.
