"""``harness config show`` and ``harness memory inspect`` (issue #15).

Two read-only commands, kept out of ``cli.py`` because they answer a different
question than ``doctor``: not what this machine can do, but what this
installation is currently configured to do and what it has already done.

Everything printed here passes the journal's redactor first. These commands
exist to be pasted into a bug report, which is exactly the moment an API key
in a stored tool argument would leave the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from harness.config import ConfigError, Origin, ResolvedConfig, load_config
from harness.core import StepStatus
from harness.memory.inspect import (
    DEFAULT_RUN_LIMIT,
    damaged_runs,
    inspect_memory,
    open_for_inspection,
)
from harness.memory.migrations import StoreError
from harness.telemetry.redact import redact_data

config_app = typer.Typer(
    help="Inspect the effective configuration.", no_args_is_help=True
)
memory_app = typer.Typer(
    help="Inspect persisted run and memory state.", no_args_is_help=True
)

console = Console()
# Diagnostics go to stderr so that --json on stdout stays parseable even when
# the report also has something to complain about.
errors = Console(stderr=True)


def _fields(title: str) -> Table:
    """Header plus an empty label/value table for one section of a report."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    console.print(f"\n[bold]{title}[/bold]")
    return table


def _fail(message: object) -> typer.Exit:
    """Report an error verbatim and hand back the exit for the caller to raise.

    Printed as text, not markup: a message quoting a redacted value or a path
    with brackets in it would otherwise be read as a style tag and vanish.
    """
    errors.print(Text(str(message), style="red"))
    return typer.Exit(1)


def _cell(value: Any) -> Text:
    """One table cell of stored data, never interpreted as markup."""
    if value is None:
        return Text("—", style="dim")
    if isinstance(value, (list, dict)):
        return Text(json.dumps(value))
    return Text(str(value))


def _leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested settings onto the dotted paths provenance is keyed by."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_leaves(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _settings_view(resolved: ResolvedConfig) -> dict[str, dict[str, Any]]:
    """Every setting with its value, the layer that set it and the source.

    Redacted in the two passes ``ResolvedConfig.render`` uses: ``as_dict``
    replaces declared secret fields by name, which is exact, and the text
    scrubber then runs over each value on its own to catch a credential in a
    free-form field such as ``extra_flags``, which nobody declared. Scrubbing
    the rendered line instead would redact ``max_output_tokens`` for
    containing the word TOKEN.
    """
    return {
        path: {
            "value": redact_data(value),
            "origin": str(resolved.origin_of(path)),
            "source": resolved.source_of(path),
        }
        for path, value in sorted(_leaves(resolved.as_dict()).items())
    }


@config_app.command("show")
def config_show(
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Read this file instead of harness.json."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout."),
    ] = False,
) -> None:
    """Print the effective configuration and the layer each value came from."""
    try:
        resolved = load_config(config_file=config_file)
    except ConfigError as exc:
        raise _fail(exc) from exc

    settings = _settings_view(resolved)
    # `warnings` arrives additively on ResolvedConfig; an object without it
    # simply has nothing to say.
    warnings: list[str] = list(getattr(resolved, "warnings", []))

    if as_json:
        typer.echo(json.dumps({"settings": settings, "warnings": warnings}, indent=2))
        return

    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("setting", style="cyan", no_wrap=True)
    table.add_column("value")
    table.add_column("from", no_wrap=True)
    table.add_column("source", overflow="fold")
    for path, entry in settings.items():
        table.add_row(
            Text(path), _cell(entry["value"]),
            Text(entry["origin"]), Text(entry["source"]),
            style="dim" if entry["origin"] == Origin.DEFAULT else "",
        )
    console.print(table)

    if warnings:
        console.print("\n[bold yellow]Attention[/bold yellow]")
        for warning in warnings:
            console.print(Text(f"  • {warning}", style="yellow"))


@memory_app.command("inspect")
def memory_inspect(
    database: Annotated[
        Path | None,
        typer.Option("--database", "-d", help="SQLite file; defaults to the "
                                              "configured database."),
    ] = None,
    run: Annotated[
        str | None,
        typer.Option("--run", "-r", help="Report on this run id alone, in full."),
    ] = None,
    status: Annotated[
        StepStatus | None,
        typer.Option("--status", "-s", help="Only steps in this state."),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="How many runs to list, newest first."),
    ] = DEFAULT_RUN_LIMIT,
    facts: Annotated[
        bool, typer.Option("--facts/--no-facts", help="Include persistent facts."),
    ] = True,
    fact: Annotated[
        str | None, typer.Option("--fact", help="Only facts matching this keyword."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout."),
    ] = False,
) -> None:
    """Show runs, task and runtime state, steps and facts. Never writes."""
    try:
        path = database if database is not None else load_config().config.database
    except ConfigError as exc:
        raise _fail(exc) from exc

    try:
        with open_for_inspection(path) as store:
            payload = inspect_memory(
                store, run_id=run, status=status, limit=limit,
                include_facts=facts, fact_query=fact,
            )
    except StoreError as exc:
        raise _fail(exc) from exc

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        _render_memory(payload, detailed=run is not None)

    # Reported and still an error: a report that could not read half the store
    # must not exit 0 into a script that then believes the run was clean.
    if damaged := damaged_runs(payload):
        raise _fail(
            f"unreadable stored state in: {', '.join(damaged)} — what is above "
            "is what could still be read"
        )


def _render_memory(payload: dict[str, Any], *, detailed: bool) -> None:
    console.print(Text(f"{payload['database']}  schema v{payload['schema_version']}",
                       style="dim"))
    runs = payload["runs"]
    if not runs:
        console.print(Text("no runs match", style="yellow"))
    else:
        _render_runs(runs)
    if detailed:
        for entry in runs:
            _render_run_detail(entry)
    if "facts" in payload:
        _render_facts(payload["facts"])


def _render_runs(entries: list[dict[str, Any]]) -> None:
    console.print("\n[bold]Runs[/bold]")
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("started", no_wrap=True)
    table.add_column("goal")
    table.add_column("outcome", no_wrap=True)
    table.add_column("steps")
    for entry in entries:
        run = entry["run"]
        table.add_row(
            Text(str(run["run_id"])),
            Text(str(run["created_at"])[:19]),
            Text(str(run["goal"])),
            Text(str(run["stop_reason"] or "open")),
            _steps_summary(entry),
        )
    console.print(table)


def _steps_summary(entry: dict[str, Any]) -> Text:
    """Step counts by state, with the ones a human has to resolve marked.

    An uncertain or never-finished step is what this report exists to surface:
    a side effect that may or may not have landed disappears into a plain
    "3 done" (#8), and it is the one line worth reading.
    """
    counts: dict[str, int] = entry["step_status_counts"]
    summary = Text(", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
                   or "none")
    if counts.get(StepStatus.UNCERTAIN.value):
        summary.stylize("yellow")
    if entry["unfinished"]:
        summary.append(f"  ({entry['unfinished']} unfinished)", style="yellow")
    return summary


def _render_run_detail(entry: dict[str, Any]) -> None:
    run = entry["run"]
    t = _fields(f"Run {escape(str(run['run_id']))}")
    t.add_row("Goal", Text(str(run["goal"])))
    t.add_row("Workspace", Text(str(run["workspace"])))
    t.add_row("Updated", Text(str(run["updated_at"])[:19]))
    t.add_row("Outcome", Text(str(run["stop_reason"] or "open")))
    if run.get("answer"):
        t.add_row("Answer", Text(str(run["answer"])))
    console.print(t)

    if (task := entry["task_state"]) is not None:
        t = _fields("Task state")
        t.add_row("Step index", Text(str(task["step_index"])))
        t.add_row("Plan", Text(f"{len(task['steps'])} steps"))
        for label, key in (("Findings", "findings"), ("Open problems", "open_problems")):
            for line in task[key]:
                t.add_row(label, Text(str(line)))
                label = ""
        console.print(t)

    if (runtime := entry["runtime_state"]) is not None:
        t = _fields("Runtime state")
        t.add_row("Tool calls", Text(str(runtime["tool_calls"])))
        t.add_row("Retries used", Text(str(runtime["retries_used"])))
        t.add_row("Tokens", Text(f"{runtime['prompt_tokens']} prompt, "
                                 f"{runtime['completion_tokens']} completion, "
                                 f"{runtime['cached_tokens']} cached"))
        for uncertain in runtime["uncertain_steps"]:
            t.add_row("Uncertain", Text(
                f"step {uncertain['step_index']}: {uncertain['tool']} — "
                f"{uncertain['detail']}", style="yellow"))
        console.print(t)

    for message in entry["errors"]:
        errors.print(Text(f"  {message}", style="red"))

    if entry["steps"]:
        _render_steps(entry["steps"])


def _render_steps(steps: list[dict[str, Any]]) -> None:
    console.print("\n[bold]Steps[/bold]")
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("#", justify="right", style="cyan")
    table.add_column("action")
    table.add_column("tool", no_wrap=True)
    table.add_column("arguments", overflow="fold")
    table.add_column("status", no_wrap=True)
    table.add_column("exit", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("note", overflow="fold")
    highlight = {
        StepStatus.UNCERTAIN.value: "yellow",
        StepStatus.FAILED.value: "red",
        StepStatus.RUNNING.value: "yellow",
    }
    for step in steps:
        duration = step["duration_ms"]
        table.add_row(
            Text(str(step["step_index"])),
            Text(str(step["action"])),
            _cell(step["tool"]),
            _cell(step["arguments"]),
            Text(str(step["status"]), style=highlight.get(step["status"], "")),
            _cell(step["exit_code"]),
            _cell(None if duration is None else f"{duration:.0f}"),
            _cell(step["note"]),
        )
    console.print(table)


def _render_facts(facts: list[dict[str, Any]]) -> None:
    console.print("\n[bold]Facts[/bold]")
    if not facts:
        console.print(Text("none recorded", style="dim"))
        return
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("category", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_column("updated", no_wrap=True)
    for fact in facts:
        table.add_row(
            Text(str(fact["key"])), Text(str(fact["category"])),
            Text(str(fact["value"])), Text(str(fact["updated_at"])[:19]),
        )
    console.print(table)
