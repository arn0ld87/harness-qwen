"""``harness chat``: the same agent, driven a turn at a time.

There is no second agent loop here, and deliberately so. A chat mode with its
own control flow would be a second place where budgets, verification and the
security boundary are enforced — and the second place is the one that drifts,
gets less testing, and is where the interesting bug ends up. Each turn runs
the same :class:`AgentLoop` against the same store; what chat adds is a person
in between, who can watch, approve and stop.

The cost of a turn is what shapes the interface: a cold prompt is ~25 s here,
so the session keeps one run id and one warm cache rather than starting fresh
per message.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from harness.agent.loop import AgentLoop
from harness.config import ConfigError, ResolvedConfig, load_config
from harness.core import Message, RunResult, StopReason
from harness.memory.migrations import StoreError
from harness.memory.store import MemoryStore
from harness.session import build_loop, new_run_id
from harness.telemetry.journal import RunJournal

console = Console()

EXIT_OK = 0
EXIT_CONFIG = 2

COMMANDS = {
    "/status": "run id, workspace, model endpoint",
    "/context": "prefix and append zone against the token budget",
    "/usage": "tokens, cache hits and elapsed time so far",
    "/help": "this list",
    "/exit": "leave the session",
}


@dataclass(slots=True)
class Totals:
    """What the session has spent, across every turn.

    Accumulated here because each turn returns its own ``RunResult``: a
    per-turn number answers "was that slow", the running total answers "is
    this session still worth continuing", and only the second one decides
    whether to start over with a smaller context.
    """

    turns: int = 0
    tool_calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    elapsed_s: float = 0.0

    def add(self, result: RunResult) -> None:
        self.turns += 1
        self.tool_calls += result.tool_calls
        self.retries += result.retries
        self.prompt_tokens += result.total_prompt_tokens
        self.completion_tokens += result.total_completion_tokens
        self.cached_tokens += result.total_cached_tokens
        self.elapsed_s += result.elapsed_s


def _confirm(command: str, reason: str) -> bool:
    """Ask the person in front of the terminal, which is the point of chat.

    ``run`` has nobody to ask and therefore denies; here the approval is a
    real decision, made with the command and the classifier's reason visible.
    """
    console.print(f"\n[yellow]The agent wants to run:[/yellow] {command}")
    console.print(f"[dim]classified as needing confirmation: {reason}[/dim]")
    return typer.confirm("Allow it?", default=False)


def _render_result(result: RunResult) -> None:
    if result.answer:
        console.print(f"\n[bold]agent[/bold] {result.answer}")
    if not result.verified and result.verification_notes:
        console.print("[yellow]unverified:[/yellow] " + "; ".join(result.verification_notes))
    if result.stop_reason is not StopReason.ANSWERED:
        console.print(f"[dim]turn ended: {result.stop_reason}[/dim]")
    for step in result.uncertain_steps:
        console.print(f"[yellow]![/yellow] {step.describe()}")


def _render_status(loop: AgentLoop, resolved: ResolvedConfig) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Run", loop.run_id)
    table.add_row("Workspace", str(loop.workspace))
    table.add_row("Endpoint", resolved.config.runtime.base_url)
    table.add_row("Tools", ", ".join(loop.tools.names))
    console.print(table)


def _render_context(loop: AgentLoop) -> None:
    """Show the budget the way the loop sees it.

    Visible on demand rather than after every turn: the number that matters is
    how close the append zone is to its ceiling, and watching it is how
    someone notices a session degrading before the compression ladder fires.
    """
    report = loop.token_budget.report(
        loop.assembler.prefix_messages(), loop.assembler.append_messages
    )
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(justify="right")
    table.add_row("Prefix", f"{report.prefix.tokens:,} tokens (cached)")
    table.add_row("Append", f"{report.append.tokens:,} / {report.append.limit:,}")
    table.add_row("Within budget", "yes" if report.append.within_budget else "[yellow]no[/yellow]")
    table.add_row("Hard ceiling", "[red]exceeded[/red]" if report.over_hard_ceiling else "ok")
    console.print(table)
    invalidations = loop.assembler.invalidations
    if invalidations:
        console.print(f"[dim]prefix invalidated {len(invalidations)}x this session[/dim]")


def _render_usage(totals: Totals) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(justify="right")
    table.add_row("Turns", str(totals.turns))
    table.add_row("Tool calls", str(totals.tool_calls))
    table.add_row("Retries", str(totals.retries))
    share = (
        f" ({100.0 * totals.cached_tokens / totals.prompt_tokens:.0f}% cached)"
        if totals.prompt_tokens
        else ""
    )
    table.add_row("Prompt tokens", f"{totals.prompt_tokens:,}{share}")
    table.add_row("Completion tokens", f"{totals.completion_tokens:,}")
    table.add_row("Elapsed", f"{totals.elapsed_s:.1f}s")
    console.print(table)


def _handle_command(
    line: str, loop: AgentLoop, resolved: ResolvedConfig, totals: Totals
) -> bool:
    """Run a control command. Returns False when the session should end."""
    command = line.split()[0].lower()
    if command in {"/exit", "/quit"}:
        return False
    if command == "/status":
        _render_status(loop, resolved)
    elif command == "/context":
        _render_context(loop)
    elif command == "/usage":
        _render_usage(totals)
    elif command == "/help":
        for name, description in COMMANDS.items():
            console.print(f"  [cyan]{name}[/cyan]  [dim]{description}[/dim]")
    else:
        console.print(f"[yellow]unknown command {command}; try /help[/yellow]")
    return True


def chat(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", "-w", help="Directory to work in.")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue an existing session by run id.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Set the id instead of generating one.")
    ] = None,
    read_only: Annotated[
        bool, typer.Option("--read-only", help="Offer no tools that change anything.")
    ] = False,
) -> None:
    """Talk to the agent, one turn at a time, in a session that survives exit.

    Same loop, same tools, same security boundary as ``harness run``; what
    this adds is a person who can approve a command and stop the run.
    """
    try:
        overrides = {"workspace": str(workspace)} if workspace else {}
        resolved = load_config(overrides=overrides)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG) from exc

    identifier = resume or run_id or new_run_id()
    config = resolved.config
    database = (
        config.database
        if config.database.is_absolute()
        else config.workspace / config.database
    )

    if resume:
        store = MemoryStore(database)
        try:
            if store.get_run(resume) is None:
                console.print(f"[red]no session with id {resume!r}[/red]")
                raise typer.Exit(EXIT_CONFIG)
            state = store.load_task_state(resume)
        finally:
            store.close()
        console.print(f"[dim]resuming {resume}[/dim]")
        if state is not None:
            console.print(f"[dim]goal: {state.goal}[/dim]")

    for warning in resolved.warnings:
        console.print(f"[yellow]![/yellow] {warning}")
    console.print(
        f"[bold]harness chat[/bold] [dim]{identifier} — /help for commands[/dim]"
    )

    totals = Totals()
    memory = MemoryStore(database)
    journal = RunJournal(database.parent / "runs" / identifier, identifier)
    try:
        _session(resolved, identifier, memory, journal, totals, read_only=read_only)
    finally:
        journal.close()
        memory.close()

    _render_usage(totals)
    raise typer.Exit(EXIT_OK)


def _session(
    resolved: ResolvedConfig,
    identifier: str,
    memory: MemoryStore,
    journal: RunJournal,
    totals: Totals,
    *,
    read_only: bool,
) -> None:
    """One prompt loop. Each turn is a full agent run against a shared store.

    The session's goal is fixed by the first message and never changes, because
    the goal sits in the cached prefix: giving every turn its own goal would
    rewrite the prefix and pay a full reprocess — ~25 s here — for each thing
    the person types. Later messages go into the append zone instead, which is
    where the conversation belongs anyway.
    """
    state = memory.load_task_state(identifier)
    goal = state.goal if state is not None else ""
    loop: AgentLoop | None = None

    while True:
        try:
            line = console.input("\n[bold cyan]you[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue

        if line.startswith("/"):
            if loop is None:
                loop = _build(resolved, identifier, goal, memory, journal, read_only)
            if not _handle_command(line, loop, resolved, totals):
                return
            continue

        if not goal:
            goal = line
        else:
            _queue_message(memory, identifier, line)

        loop = _build(resolved, identifier, goal, memory, journal, read_only)
        try:
            result = asyncio.run(loop.run())
        except StoreError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        except KeyboardInterrupt:
            # Cancelled mid-turn: the store already holds every completed
            # step, so the session continues rather than starting over.
            console.print("\n[yellow]turn cancelled[/yellow]")
            continue

        totals.add(result)
        _render_result(result)


def _queue_message(memory: MemoryStore, identifier: str, line: str) -> None:
    """Add the person's message to the persisted conversation.

    Written into the stored runtime state rather than onto the assembler,
    because the loop restores that state on entry and would overwrite anything
    set beforehand. This is the same path a tool result takes, which is what
    keeps the turn resumable: a process killed here comes back with the
    message still in the conversation.
    """
    runtime = memory.load_runtime_state(identifier)
    if runtime is None:
        return
    runtime.append_history.append(Message(role="user", content=line))
    memory.save_runtime_state(runtime)


def _build(
    resolved: ResolvedConfig,
    identifier: str,
    goal: str,
    memory: MemoryStore,
    journal: RunJournal,
    read_only: bool,
) -> AgentLoop:
    return build_loop(
        resolved.config,
        goal=goal,
        run_id=identifier,
        memory=memory,
        journal=journal,
        confirm=_confirm,
        read_only=read_only,
    )


__all__ = ["COMMANDS", "Totals", "chat"]
