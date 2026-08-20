"""Prompt assembly under the prefix contract.

Two zones (ARCHITECTURE.md, CONTEXT.md §2): a stable prefix that must be
byte-identical for the whole run, and an append zone that only grows. The
prefix is hashed on every build and compared to the previous hash. An
undeclared mismatch raises :class:`PrefixViolation` instead of quietly costing
a full reprocess — at 8k context that is ~50 s, roughly 825 tokens of output
the run does not get to produce.

Roles live in the append zone. A role-specific system prompt would turn every
handover into a prefix change; appending the directive costs the ~0.8 s tail
rate however often roles switch.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from harness.context.budget import BudgetReport, TokenBudget
from harness.core import Message, Role, ToolResult, ToolSpec

SEGMENT_ORDER: tuple[str, ...] = ("system", "tools", "repo_map", "task")
"""The four genuinely run-constant segments (CONTEXT.md §2). Everything else
changes at least once per step and belongs in the append zone."""

SEGMENT_HEADERS: dict[str, str] = {
    "system": "### SYSTEM",
    "tools": "### TOOLS",
    "repo_map": "### REPOSITORY MAP",
    "task": "### TASK",
}

FULL_OUTPUT_MARKER = "[full output:"
"""Marks a tool result whose complete output is still on disk, addressable by
id. Compression may drop such a message outright; without the marker the
content in context is the only copy."""

DEFAULT_ROLE_DIRECTIVES: dict[Role, str] = {
    Role.PLANNER: "Act as planner: decompose the task into checkable steps.",
    Role.CODER: "Act as coder: make the smallest change that satisfies the step.",
    Role.TESTER: "Act as tester: run the checks and report exit codes verbatim.",
    Role.REVIEWER: "Act as reviewer: judge the evidence, not the claim.",
}


class InvalidationReason(enum.StrEnum):
    """The fixed set of legitimate prefix changes (CONTEXT.md §4).

    A closed vocabulary is the point: a reason that is not on this list is a
    prefix change nobody budgeted for.
    """

    INITIAL = "initial"
    TASK_CHANGED = "task_changed"
    TOOLS_CHANGED = "tools_changed"
    REPO_MAP_REFRESHED = "repo_map_refreshed"
    COMPRESSION_APPROVED = "compression_approved"


class PrefixViolation(RuntimeError):
    """The stable prefix changed without a declared invalidation.

    Carries the hashes and the changed segment names so the journal records
    what drifted, not merely that something did.
    """

    def __init__(
        self, *, expected: str, actual: str, segments: Sequence[str]
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.segments = list(segments)
        changed = ", ".join(self.segments) or "unknown"
        super().__init__(
            f"stable prefix changed without invalidate(): segments [{changed}]; "
            f"expected {expected[:12]}, got {actual[:12]}"
        )


class InvalidationRecord(BaseModel):
    """Telemetry for one prefix change, applied or not."""

    reasons: list[InvalidationReason]
    notes: list[str] = Field(default_factory=list)
    previous_hash: str | None
    new_hash: str
    changed_segments: list[str] = Field(default_factory=list)
    prefix_tokens: int = 0
    applied: bool = True
    """False when an invalidation was declared but the prefix did not change.
    Recorded rather than carried forward, so a stale declaration cannot cover a
    later, genuinely unannounced change."""


class AssembledPrompt(BaseModel):
    """One built prompt: the messages, and the hash of the prefix they carry."""

    messages: list[Message]
    prefix_hash: str
    prefix_tokens: int
    append_tokens: int
    budget: BudgetReport | None = None


def render_tool_schemas(tools: Sequence[ToolSpec]) -> str:
    """Serialise tool schemas canonically.

    Sorted by name with sorted keys: the registry's iteration order must not
    be able to change the prefix bytes and cost a reprocess for nothing.
    """
    payload = [spec.to_openai_tool() for spec in sorted(tools, key=lambda s: s.name)]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class PromptAssembler:
    """Builds prompts and enforces the prefix contract.

    The prefix may only change after :meth:`invalidate` has declared why.
    Every other mutation lands in the append zone.
    """

    def __init__(
        self,
        *,
        system: str,
        task: str,
        tools: Sequence[ToolSpec] = (),
        repo_map: str = "",
        budget: TokenBudget | None = None,
    ) -> None:
        self._system = system
        self._task = task
        self._tools: tuple[ToolSpec, ...] = tuple(tools)
        self._repo_map = repo_map
        self._budget = budget
        self._append: list[Message] = []
        self._last_hash: str | None = None
        self._last_segments: dict[str, str] = {}
        self._pending: list[tuple[InvalidationReason, str]] = []
        self._invalidations: list[InvalidationRecord] = []
        self._append_rewrites: list[str] = []

    # -- prefix ------------------------------------------------------------

    def _segments(self) -> dict[str, str]:
        return {
            "system": self._system,
            "tools": render_tool_schemas(self._tools) if self._tools else "",
            "repo_map": self._repo_map,
            "task": self._task,
        }

    def prefix_text(self) -> str:
        """The four prefix segments in fixed order.

        Empty segments are omitted entirely; adding one later is a declared
        invalidation like any other content change.
        """
        segments = self._segments()
        blocks = [
            f"{SEGMENT_HEADERS[name]}\n{segments[name].strip()}"
            for name in SEGMENT_ORDER
            if segments[name].strip()
        ]
        return "\n\n".join(blocks)

    def prefix_messages(self) -> list[Message]:
        return [Message(role="system", content=self.prefix_text())]

    def serialise_prefix(self) -> bytes:
        """The exact bytes the prefix hash is taken over.

        Roles are serialised with the content: a segment moving between
        messages changes the token stream even when the text is identical.
        """
        payload = [
            message.model_dump(exclude_none=True) for message in self.prefix_messages()
        ]
        return json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

    def prefix_hash(self) -> str:
        return hashlib.sha256(self.serialise_prefix()).hexdigest()

    def _segment_digests(self) -> dict[str, str]:
        return {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in self._segments().items()
        }

    def invalidate(self, reason: InvalidationReason, note: str = "") -> None:
        """Declare a prefix change before making it.

        Declaring twice before a build is legitimate — a new goal that also
        registers a tool is one reprocess, two reasons — and both are recorded.
        """
        self._pending.append((InvalidationReason(reason), note))

    def set_task(self, task: str) -> None:
        self._task = task

    def set_system(self, system: str) -> None:
        self._system = system

    def set_repo_map(self, repo_map: str) -> None:
        self._repo_map = repo_map

    def set_tools(self, tools: Sequence[ToolSpec]) -> None:
        self._tools = tuple(tools)

    @property
    def task(self) -> str:
        """The current task segment — read-only, for snapshotting before a
        change a caller intends to revert (e.g. a benchmark violation probe)."""
        return self._task

    @property
    def invalidations(self) -> list[InvalidationRecord]:
        return list(self._invalidations)

    @property
    def append_rewrites(self) -> list[str]:
        return list(self._append_rewrites)

    # -- append zone -------------------------------------------------------

    @property
    def append_messages(self) -> list[Message]:
        return list(self._append)

    def append(self, message: Message) -> None:
        self._append.append(message)

    def append_role_directive(self, role: Role, directive: str | None = None) -> None:
        """Switch role by appending a directive at the current tail.

        Sent as a user turn rather than a system turn: mid-conversation system
        messages are not portable across chat templates, and the whole point is
        that this never touches the one system message in the prefix.
        """
        text = directive or DEFAULT_ROLE_DIRECTIVES.get(role, f"Act as {role}.")
        self._append.append(Message(role="user", content=f"[role: {role}]\n{text}"))

    def append_tool_result(self, result: ToolResult) -> None:
        """Append the compressed view of a tool result (CONTEXT.md §6)."""
        content = result.content
        if result.truncated and result.full_output_ref:
            content = f"{content}\n{FULL_OUTPUT_MARKER} {result.full_output_ref}]"
        self._append.append(Message(role="tool", content=content, name=result.tool))

    def append_retrieved(self, text: str, *, source: str = "retrieval") -> None:
        """Append retrieved context — structurally a tool result at the tail."""
        self._append.append(Message(role="user", content=f"[retrieved: {source}]\n{text}"))

    def rewrite_append(self, messages: Sequence[Message], *, note: str) -> None:
        """Replace the append zone after an approved compression.

        No invalidation is needed: an append-zone rewrite leaves the prefix
        bytes untouched, so the cached prefix still hits. It is still recorded,
        because everything after the edit point is reprocessed.
        """
        self._append = list(messages)
        self._append_rewrites.append(note)

    def restore_append(self, messages: Sequence[Message]) -> None:
        """Restore a persisted append zone without recording a new rewrite."""
        self._append = list(messages)

    # -- build -------------------------------------------------------------

    def build(self) -> AssembledPrompt:
        """Assemble the prompt and enforce the contract.

        Raises:
            PrefixViolation: the prefix changed and no invalidation was
                declared first. The assembler is left untouched so a caller
                that can revert the change may continue on the cached prefix.
        """
        digest = self.prefix_hash()
        segments = self._segment_digests()
        changed = [
            name
            for name, value in segments.items()
            if self._last_segments.get(name, value) != value
        ]

        if self._last_hash is None:
            self._record(digest, changed=list(SEGMENT_ORDER), applied=True, initial=True)
        elif digest != self._last_hash:
            if not self._pending:
                raise PrefixViolation(
                    expected=self._last_hash, actual=digest, segments=changed
                )
            self._record(digest, changed=changed, applied=True)
        elif self._pending:
            # Declared, then nothing changed. Dropping the declaration keeps it
            # from silently authorising a later, genuinely unannounced change.
            self._record(digest, changed=[], applied=False)

        self._last_hash = digest
        self._last_segments = segments

        prefix = self.prefix_messages()
        messages = prefix + self._append
        report = self._budget.report(prefix, self._append) if self._budget else None
        prefix_tokens = report.prefix.tokens if report else 0
        append_tokens = report.append.tokens if report else 0
        return AssembledPrompt(
            messages=messages,
            prefix_hash=digest,
            prefix_tokens=prefix_tokens,
            append_tokens=append_tokens,
            budget=report,
        )

    def _record(
        self, digest: str, *, changed: list[str], applied: bool, initial: bool = False
    ) -> None:
        reasons = [reason for reason, _ in self._pending]
        notes = [note for _, note in self._pending if note]
        if initial and not reasons:
            reasons = [InvalidationReason.INITIAL]
        self._pending = []
        self._invalidations.append(
            InvalidationRecord(
                reasons=reasons,
                notes=notes,
                previous_hash=self._last_hash,
                new_hash=digest,
                changed_segments=changed,
                prefix_tokens=(
                    self._budget.count_messages(self.prefix_messages()) if self._budget else 0
                ),
                applied=applied,
            )
        )
