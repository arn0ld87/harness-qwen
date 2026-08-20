"""Shared types crossing module boundaries.

Everything here is imported by more than one subsystem. Keeping these
definitions in one place is what allows the model layer, the protocol layer,
the tool layer and the agent loop to be developed and tested independently.

Nothing in this module may import from other harness subsystems — it is the
bottom of the dependency graph.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Model layer
# --------------------------------------------------------------------------


class Message(BaseModel):
    """One chat message. ``name`` carries the tool name for tool results."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class GenerationRequest(BaseModel):
    """A single model call.

    ``prefix_token_estimate`` lets the provider report cache behaviour back to
    telemetry: a call that reprocesses a prefix it should have reused is a bug
    worth seeing, not a slow request to shrug at.
    """

    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None

    json_schema: dict[str, Any] | None = None
    """When set, the runtime constrains sampling to this schema."""

    tools: list[dict[str, Any]] | None = None
    """OpenAI-style tool definitions for native tool calling."""

    timeout_s: float = 600.0
    prefix_token_estimate: int | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens served from cache. The gap between this and
    ``prompt_tokens`` is what a prefix change actually cost."""


class Timings(BaseModel):
    """Wall-clock measurements reported by the runtime, in milliseconds."""

    prompt_ms: float | None = None
    predicted_ms: float | None = None
    prompt_per_second: float | None = None
    predicted_per_second: float | None = None
    time_to_first_token_ms: float | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ModelResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)
    reasoning: str | None = None
    """Never logged and never fed back into context. Present so the loop can
    distinguish reasoning from content when the template separates them."""


class Chunk(BaseModel):
    delta: str = ""
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class HealthStatus(BaseModel):
    reachable: bool
    loading: bool = False
    detail: str | None = None


class RuntimeModelInfo(BaseModel):
    """What the serving runtime reports about the loaded model."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = None
    model_path: str | None = None
    n_ctx: int | None = None
    total_slots: int | None = None
    build_info: str | None = None
    chat_template_caps: dict[str, bool] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    """Base class for model provider failures."""


class ProviderUnavailable(ProviderError):
    """The endpoint did not answer, or the model is not loaded yet."""


class ProviderTimeout(ProviderError):
    """The request exceeded its deadline."""


class ContextOverflow(ProviderError):
    """The request did not fit the runtime's context window."""


# --------------------------------------------------------------------------
# Tool layer
# --------------------------------------------------------------------------


class Risk(enum.StrEnum):
    """Security classification of an operation.

    Resolution is deterministic and happens before execution. ``DENY`` beats
    ``ALLOW``; anything unrecognised resolves to ``CONFIRM``, because an
    unknown command is not a safe command.
    """

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class NetworkMode(enum.StrEnum):
    """Network policy a shell command runs under.

    ``ISOLATED`` is the default: the sandbox gets its own network namespace
    so an approved command still cannot reach the host network. ``ALLOWED`` is
    the explicit, human-approved opt-out for commands that need network access
    — it is never the default and is auditable on the :class:`ToolResult`.
    """

    ISOLATED = "isolated"
    ALLOWED = "allowed"


class ToolSpec(BaseModel):
    """Declaration of a tool as presented to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments object."""
    risk: Risk = Risk.ALLOW
    timeout_s: float = 120.0

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """Outcome of a tool execution.

    ``content`` is the compressed view that enters the model context.
    ``full_output_ref`` addresses the complete output on disk, so nothing is
    lost even though it is not paid for on every subsequent step.
    """

    tool: str
    ok: bool
    content: str
    exit_code: int | None = None
    duration_ms: float = 0.0
    full_output_ref: str | None = None
    truncated: bool = False
    original_bytes: int | None = None
    error_kind: str | None = None
    """Set when ok is False: not_found, timeout, denied, invalid_arguments,
    execution_failed, permission_denied."""
    network: NetworkMode | Literal["unsandboxed"] | None = None
    """Network policy the command ran under (``isolated``/``allowed``), or
    ``unsandboxed`` for the trusted read-only fallback when bubblewrap is
    absent. ``None`` means the command did not execute (denied/timeout)."""


class ToolError(RuntimeError):
    """Raised by a tool when it cannot produce a result at all."""

    def __init__(self, message: str, kind: str = "execution_failed") -> None:
        super().__init__(message)
        self.kind = kind


class ExecutedToolStep(BaseModel):
    """Durable tool execution used for resume and verification."""

    id: int
    run_id: str
    step_index: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult


# --------------------------------------------------------------------------
# Protocol layer
# --------------------------------------------------------------------------


class ToolAction(BaseModel):
    """The model asks for a tool to be executed."""

    action: Literal["tool"] = "tool"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class CommandEvidence(BaseModel):
    """Reference to a successful command step classified by the harness."""

    kind: Literal["test", "lint", "typecheck", "build"]
    step_id: int = Field(gt=0)


class FileEvidence(BaseModel):
    """Reference to a workspace path checked against run-start state."""

    kind: Literal["file", "patch"]
    path: str


Evidence = Annotated[CommandEvidence | FileEvidence, Field(discriminator="kind")]


class AnswerAction(BaseModel):
    """The model considers the task complete.

    ``evidence`` lists references the verifier can check. An answer claiming
    success with no evidence is reported as unverified, never as done.
    """

    action: Literal["answer"] = "answer"
    content: str
    evidence: list[Evidence] = Field(default_factory=list)


Action = ToolAction | AnswerAction


class ParseError(BaseModel):
    """A model response that did not yield a valid action."""

    raw: str
    reason: str
    recoverable: bool = True

    def as_feedback(self) -> str:
        """Phrasing handed back to the model on retry.

        Retries must change the input. Echoing the concrete failure is the
        cheapest way to do that.
        """
        return f"Your previous response could not be parsed: {self.reason}"


# --------------------------------------------------------------------------
# Agent layer
# --------------------------------------------------------------------------


class Role(enum.StrEnum):
    """Sequential roles sharing one model, one slot and one warm cache.

    These are not concurrent processes. A second concurrent request was
    measured to roughly halve generation throughput on this hardware.
    """

    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"


class StepStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    id: int
    task: str
    status: StepStatus = StepStatus.PENDING
    note: str | None = None


class WorkspaceBaseline(BaseModel):
    """Auditable workspace state captured before a run may change files."""

    head_sha: str | None = None
    status_sha256: str
    diff_sha256: str
    files: dict[str, str] = Field(default_factory=dict)
    captured_at: datetime


class RunRuntimeState(BaseModel):
    """Runtime bookkeeping that must survive reconstruction of AgentLoop."""

    run_id: str
    executed_steps: list[ExecutedToolStep] = Field(default_factory=list)
    recent_calls: list[tuple[str, str]] = Field(default_factory=list)
    append_history: list[Message] = Field(default_factory=list)
    active_step_index: int | None = None
    tool_calls: int = 0
    retries_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    elapsed_s: float = 0.0


class TaskState(BaseModel):
    """Durable task state. Written before each step so a killed run resumes."""

    run_id: str
    goal: str
    workspace: str
    steps: list[PlanStep] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    step_index: int = 0
    workspace_baseline: WorkspaceBaseline | None = None
    created_at: datetime
    updated_at: datetime


class Budget(BaseModel):
    """Hard bounds on a run. None of these are optional."""

    max_steps: int = 20
    wall_clock_s: float = 1800.0
    max_output_tokens: int = 2048
    max_tool_calls: int = 60
    max_retries: int = 3

    def exhausted(self, *, steps: int, elapsed_s: float,
                  tool_calls: int, retries: int) -> str | None:
        """Return the name of the first exceeded bound, or None."""
        if steps >= self.max_steps:
            return "max_steps"
        if elapsed_s >= self.wall_clock_s:
            return "wall_clock_s"
        if tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if retries >= self.max_retries:
            return "max_retries"
        return None


class StopReason(enum.StrEnum):
    ANSWERED = "answered"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    UNRECOVERABLE_ERROR = "unrecoverable_error"
    CANCELLED = "cancelled"


class RunResult(BaseModel):
    run_id: str
    stop_reason: StopReason
    answer: str | None = None
    verified: bool = False
    verification_notes: list[str] = Field(default_factory=list)
    steps_taken: int = 0
    tool_calls: int = 0
    retries: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    elapsed_s: float = 0.0
