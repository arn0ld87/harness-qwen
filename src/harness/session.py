"""Assemble a runnable agent from configuration.

Every component this builds is separately testable, which is why they take
each other as arguments rather than constructing one another. The cost of that
is a caller who has to know how eight pieces fit together — so exactly one
place knows, and it is this one. ``harness run`` and ``harness chat`` differ in
how they talk to a person, not in what they assemble.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from harness.agent.loop import AgentLoop
from harness.config.schema import HarnessConfig
from harness.context.assembler import PromptAssembler
from harness.context.budget import TokenBudget
from harness.core import Budget, NetworkMode
from harness.memory.store import MemoryStore
from harness.models.base import ModelProvider
from harness.models.llamacpp import LlamaCppProvider
from harness.protocol.codec import ActionCodec
from harness.protocol.constrained import ConstrainedJsonCodec
from harness.protocol.native import NativeToolCallCodec
from harness.retrieval import as_retrieve_fn
from harness.retrieval.sqlite import SqliteFtsRetriever
from harness.telemetry.journal import RunJournal
from harness.tools.builtin import build_registry
from harness.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are a coding agent working in a local repository.

Do the work; do not describe it. Never answer with a plan, a list of steps, or \
what someone "should" do — call the tool that does it. An answer is for \
reporting work that is already finished.

Take one step at a time: call a single tool, read its result, then decide the \
next step from what you actually saw. Read the code before changing it.

When the task is done, answer with the evidence: a command you ran that exited \
zero, or a file you changed. An answer that claims success without evidence is \
reported as unverified."""
"""The stable prefix. It is cached and reprocessed only when deliberately
invalidated, so it is written once and left alone — a reworded sentence here
costs a full prompt reprocess on every subsequent run.

The opening instruction is blunt on purpose: a 35B model asked to fix a bug
answers with the three steps it would take, which reads as success (nothing
was claimed, so nothing failed verification) while the file is untouched."""


def new_run_id() -> str:
    """A run id that sorts by time and does not collide between processes."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{uuid.uuid4().hex[:6]}"


def build_provider(config: HarnessConfig) -> ModelProvider:
    return LlamaCppProvider(
        config.runtime.base_url,
        model=config.model.alias,
        api_key=config.runtime.api_key,
    )


def build_codec(tools: ToolRegistry, *, native: bool) -> ActionCodec:
    """Pick the protocol the model is asked to speak.

    Native tool calls are what the runtime advertises; constrained JSON is the
    fallback that works regardless of chat template. Which one this model
    actually obeys is a measurement, not a preference — until that measurement
    exists, the constrained codec is the safer default because its failure
    mode is a parse error the loop can retry, not a silently dropped call.
    """
    specs = tools.specs()
    return NativeToolCallCodec(specs) if native else ConstrainedJsonCodec(specs)


def build_loop(
    config: HarnessConfig,
    *,
    goal: str,
    run_id: str | None = None,
    provider: ModelProvider | None = None,
    memory: MemoryStore | None = None,
    journal: RunJournal | None = None,
    confirm: Callable[[str, str], bool] | None = None,
    native_tool_calls: bool = False,
    read_only: bool = False,
) -> AgentLoop:
    """Build one agent loop from configuration.

    The injectable arguments exist for tests and for ``chat``, which needs to
    hand in its own approval callback. Everything else follows from the config,
    so two runs with the same configuration assemble the same agent.
    """
    workspace = config.workspace.resolve()
    identifier = run_id or new_run_id()
    store = memory or MemoryStore(_database_path(config))
    # One retriever feeds two consumers: the ``retrieve_facts`` tool the model
    # calls, and the compression ladder's ``RetrieveAgain`` rung. Both only
    # append, so neither touches the cached prefix; wiring both here is what
    # makes retrieval a first-class part of the context pipeline rather than a
    # tool floating beside it (issue #18).
    retriever = SqliteFtsRetriever(store)
    tools = build_registry(
        workspace,
        confirm=confirm,
        network=config.sandbox.network or NetworkMode.ISOLATED,
        read_only=read_only,
        retriever=retriever,
    )
    assembler = PromptAssembler(
        system=SYSTEM_PROMPT, task=goal, tools=tools.specs()
    )
    token_budget = TokenBudget(
        context_window=config.context.context_window,
        soft_ceiling=config.context.soft_ceiling,
        generation_reserve_ratio=config.context.generation_reserve_ratio,
    )

    return AgentLoop(
        run_id=identifier,
        goal=goal,
        workspace=workspace,
        provider=provider or build_provider(config),
        codec=build_codec(tools, native=native_tool_calls),
        tools=tools,
        assembler=assembler,
        token_budget=token_budget,
        memory=store,
        journal=journal or RunJournal(_journal_dir(config, identifier), identifier),
        budget=Budget(**config.budget.model_dump()),
        retrieve=as_retrieve_fn(retriever),
    )


def _database_path(config: HarnessConfig) -> Path:
    database = config.database
    return database if database.is_absolute() else config.workspace / database


def _journal_dir(config: HarnessConfig, run_id: str) -> Path:
    return _database_path(config).parent / "runs" / run_id


__all__ = [
    "SYSTEM_PROMPT",
    "build_codec",
    "build_loop",
    "build_provider",
    "new_run_id",
]
