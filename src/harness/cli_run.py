"""``harness run``: one agent task from the command line.

Separate from ``cli.py`` to keep both under the size where a file stops being
readable, and because this is where the process-level contract lives: what a
run prints, and what its exit code means to whatever called it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from harness.config import ConfigError, load_config
from harness.core import RunResult, StopReason
from harness.memory.migrations import StoreError
from harness.memory.store import MemoryStore
from harness.session import build_loop, new_run_id

console = Console()

EXIT_OK = 0
EXIT_UNVERIFIED = 3
EXIT_BUDGET = 4
EXIT_NO_PROGRESS = 5
EXIT_CANCELLED = 6
EXIT_ERROR = 1
EXIT_CONFIG = 2

_STOP_EXIT: dict[StopReason, int] = {
    StopReason.ANSWERED: EXIT_OK,
    StopReason.BUDGET_EXHAUSTED: EXIT_BUDGET,
    StopReason.NO_PROGRESS: EXIT_NO_PROGRESS,
    StopReason.CANCELLED: EXIT_CANCELLED,
    StopReason.UNRECOVERABLE_ERROR: EXIT_ERROR,
}
"""Distinct codes because the responses differ. A budget that ran out wants a
bigger budget; no progress wants a different task; an unverified answer wants
a human to look at the claim. One generic failure code would hide all three."""


def exit_code_for(result: RunResult) -> int:
    """Map a finished run onto a process exit code.

    An answered-but-unverified run is deliberately not 0: the harness could
    not confirm what the model claimed, and a caller that treats that as
    success has bought exactly the assurance this project exists to refuse.
    """
    if result.stop_reason is StopReason.ANSWERED and not result.verified:
        return EXIT_UNVERIFIED
    return _STOP_EXIT.get(result.stop_reason, EXIT_ERROR)


def _overrides(
    workspace: Path | None,
    base_url: str | None,
    attach: bool | None,
    max_steps: int | None,
    max_tool_calls: int | None,
    wall_clock: float | None,
) -> dict[str, Any]:
    """CLI flags as configuration overrides, so they take the top layer.

    Passed through the same resolution as every other source rather than
    applied afterwards: that is what keeps ``config show`` honest about where
    a value came from.
    """
    flags: dict[str, Any] = {}
    if workspace is not None:
        flags["workspace"] = str(workspace)
    if base_url is not None:
        flags["runtime.base_url"] = base_url
    if attach is not None:
        flags["runtime.attach"] = attach
    if max_steps is not None:
        flags["budget.max_steps"] = max_steps
    if max_tool_calls is not None:
        flags["budget.max_tool_calls"] = max_tool_calls
    if wall_clock is not None:
        flags["budget.wall_clock_s"] = wall_clock
    return flags


def _render(result: RunResult) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Run", result.run_id)
    table.add_row("Stopped", str(result.stop_reason))
    table.add_row("Verified", "[green]yes[/green]" if result.verified else "[yellow]no[/yellow]")
    table.add_row("Steps", str(result.steps_taken))
    table.add_row("Tool calls", str(result.tool_calls))
    table.add_row("Retries", str(result.retries))
    cached = result.total_cached_tokens
    prompt = result.total_prompt_tokens
    share = f" ({100.0 * cached / prompt:.0f}% cached)" if prompt else ""
    table.add_row("Prompt tokens", f"{prompt:,}{share}")
    table.add_row("Completion tokens", f"{result.total_completion_tokens:,}")
    table.add_row("Elapsed", f"{result.elapsed_s:.1f}s")
    console.print(table)

    if result.answer:
        console.print("\n[bold]Answer[/bold]")
        console.print(result.answer)

    if result.verification_notes:
        console.print("\n[bold yellow]Unverified[/bold yellow]")
        for note in result.verification_notes:
            console.print(f"  [yellow]•[/yellow] {note}")

    if result.uncertain_steps:
        console.print("\n[bold yellow]Interrupted side effects[/bold yellow]")
        for step in result.uncertain_steps:
            console.print(f"  [yellow]•[/yellow] {step.describe()}")


def run(
    goal: Annotated[str, typer.Argument(help="What the agent should accomplish.")],
    workspace: Annotated[
        Path | None, typer.Option("--workspace", "-w", help="Directory to work in.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Set the id instead of generating one.")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue an existing run by id.")
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Inference server to talk to.")
    ] = None,
    attach: Annotated[
        bool | None, typer.Option("--attach/--no-attach", help="Use a running server.")
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    max_tool_calls: Annotated[int | None, typer.Option("--max-tool-calls")] = None,
    wall_clock: Annotated[
        float | None, typer.Option("--wall-clock", help="Seconds before the run stops.")
    ] = None,
    read_only: Annotated[
        bool, typer.Option("--read-only", help="Offer no tools that change anything.")
    ] = False,
    approve: Annotated[
        bool,
        typer.Option(
            "--approve-confirmable",
            help="Approve commands classified as needing confirmation.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the result as JSON.")] = False,
) -> None:
    """Run one agent task to completion.

    Exit codes: 0 answered and verified, 3 answered but unverified, 4 budget
    exhausted, 5 no progress, 6 cancelled, 1 unrecoverable, 2 bad config.
    """
    try:
        resolved = load_config(
            overrides=_overrides(
                workspace, base_url, attach, max_steps, max_tool_calls, wall_clock
            )
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG) from exc

    identifier = resume or run_id or new_run_id()
    if resume and not _run_exists(resolved.config, resume):
        console.print(f"[red]no run with id {resume!r} to resume[/red]")
        raise typer.Exit(EXIT_CONFIG)

    for warning in resolved.warnings:
        console.print(f"[yellow]![/yellow] {warning}")

    try:
        loop = build_loop(
            resolved.config,
            goal=goal,
            run_id=identifier,
            read_only=read_only,
            confirm=_approver(resolved.config.workspace) if approve else None,
        )
    except StoreError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG) from exc

    try:
        result = asyncio.run(loop.run())
    except StoreError as exc:
        # A resumed run whose goal or workspace no longer matches: refusing is
        # the point, since continuing would append to somebody else's history.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG) from exc
    except KeyboardInterrupt:
        console.print("[yellow]interrupted; the run is resumable by its id[/yellow]")
        raise typer.Exit(EXIT_CANCELLED) from None
    finally:
        loop.journal.close()
        loop.memory.close()

    code = exit_code_for(result)
    if as_json:
        # The reasoning never leaves the model layer, so there is nothing to
        # filter here: RunResult carries outcomes, not chain of thought.
        console.print_json(json.dumps(result.model_dump(mode="json")))
    else:
        _render(result)
    raise typer.Exit(code)


def _approver(workspace: Path) -> Callable[[str, str], bool]:
    """Standing approval for commands the classifier flags as confirmable.

    Without one, an unattended run cannot do anything the classifier does not
    already trust — which is the correct default, and also why a run asked to
    edit a file with ``sed`` spends its budget being denied. This is the
    explicit, auditable opt-out: a person typed the flag, every approved
    command is printed as it is approved, and DENY is still DENY.
    """

    def approve(command: str, reason: str) -> bool:
        console.print(f"[yellow]approved[/yellow] [dim]({reason})[/dim] {command}")
        return True

    return approve


def _run_exists(config: Any, run_id: str) -> bool:
    database = config.database
    if not database.is_absolute():
        database = config.workspace / database
    if not database.exists():
        return False
    store = MemoryStore(database)
    try:
        return store.get_run(run_id) is not None
    finally:
        store.close()


__all__ = ["EXIT_UNVERIFIED", "exit_code_for", "run"]
