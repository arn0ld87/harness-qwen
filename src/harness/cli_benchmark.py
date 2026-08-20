"""``harness benchmark``: reproducible capability/flag/task runs from the CLI.

Separate from ``cli.py`` for the same reason ``cli_run`` is: this is where the
process-level contract for a benchmark lives — which suite a subcommand runs,
where the artefact is written, and which exit code a measurement that was not
trustworthy returns to its caller. The framework that produces the numbers
lives in ``harness.benchmark``; this module only wires it to Typer and decides
what a caller sees.

Three run subcommands (``capability``, ``flags``, ``tasks``) share one runner
and differ only in the suite file they default to. Each accepts ``--suite`` to
run any suite through it, so a subcommand is a default path rather than a
constraint — ``flags`` and ``tasks`` land their canonical suites with #20 and
#21, and until then a custom suite is runnable through either. ``compare``
sets two run artefacts side by side and states whether the delta between them
is a capability change or a fingerprint that moved.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from harness.benchmark import (
    BenchmarkRun,
    BenchmarkRunner,
    BenchmarkSuite,
    load_suite,
    read_run,
    render_comparison,
    render_summary,
    write_run,
)
from harness.config import ConfigError, ResolvedConfig, load_config
from harness.core import ProviderError
from harness.runtime import (
    PortConflict,
    RuntimeCrashed,
    RuntimeError_,
    RuntimeIdentityMismatch,
    RuntimeNotOwned,
    RuntimeStartError,
    RuntimeStartTimeout,
)
from harness.runtime.supervisor import LlamaServerSupervisor
from harness.session import build_provider

console = Console()
err_console = Console(stderr=True)

# Exit codes: three states the issue requires, plus config kept consistent with
# ``cli_run``. An invalid run is not a failure (the runner produced a record)
# and not a success (the record says do not compare it) — it gets its own code.
EXIT_OK = 0
EXIT_INVALID = 3
EXIT_ERROR = 1
EXIT_CONFIG = 2

BENCHMARKS_DIR = Path("benchmarks")
SUITE_PATHS: dict[str, Path] = {
    "capability": BENCHMARKS_DIR / "model-capabilities.json",
    "flags": BENCHMARKS_DIR / "flag-sweep.json",
    "tasks": BENCHMARKS_DIR / "task-suite.json",
}
DEFAULT_OUT = BENCHMARKS_DIR / "runs"

# Everything the runtime layer can raise while starting or verifying a server.
# Each is an execution failure: no run was produced, so there is nothing to
# invalidate, only a message and a non-zero exit.
_RUNTIME_FAILURES: tuple[type[BaseException], ...] = (
    RuntimeIdentityMismatch,
    PortConflict,
    RuntimeStartError,
    RuntimeStartTimeout,
    RuntimeCrashed,
    RuntimeError_,
    RuntimeNotOwned,
    ProviderError,
)

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run reproducible capability, flag and task suites against a runtime.",
    no_args_is_help=True,
    add_completion=False,
)


# -- shared option aliases (one declaration, three subcommands) -------------


SuiteOpt = Annotated[
    Path | None,
    typer.Option("--suite", help="Suite file to run (overrides the subcommand default)."),
]
OutOpt = Annotated[
    Path, typer.Option("--out", "-o", help="Directory to write the JSON artefact into.")
]
RunIdOpt = Annotated[
    str | None, typer.Option("--run-id", help="Set the run id instead of generating one.")
]
BaseUrlOpt = Annotated[
    str | None, typer.Option("--base-url", help="Inference server to talk to.")
]
AttachOpt = Annotated[
    bool | None,
    typer.Option("--attach/--no-attach", help="Use a running server this harness did not start."),
]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Configuration file to load.")
]
ProfileOpt = Annotated[
    Path | None,
    typer.Option(
        "--profile",
        help="Hardware profile file (config/hardware-profile.json by default).",
    ),
]
CaseOpt = Annotated[
    list[str] | None,
    typer.Option("--case", help="Run only these case ids (repeatable). Refuses unknown ids."),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Emit the run as JSON instead of the text summary.")
]


# -- the runner seam -------------------------------------------------------


async def build_runner(
    resolved: ResolvedConfig,
) -> tuple[BenchmarkRunner, Callable[[], Awaitable[None]]]:
    """Construct the runner and the handle that proves who served it.

    The benchmark owns the server on purpose: an attached server has no
    ``launch_argv`` in its fingerprint, and the launch flags are the single
    largest lever on the result (DISCOVERY.md 5.2). ``ensure()`` starts one
    unless the configuration asks to attach, and the cleanup tears down only
    what this process started — an attached server is left alone.

    Replaced wholesale in tests with a factory that returns a runner wired to
    ``FakeProvider`` and stub probes, so the CLI is exercised without a model,
    a GPU or a socket.
    """
    provider = build_provider(resolved.config)
    supervisor = LlamaServerSupervisor(resolved.config)
    handle = await supervisor.ensure()
    runner = BenchmarkRunner(provider, resolved, handle=handle)

    async def cleanup() -> None:
        await supervisor.stop()
        await supervisor.close()

    return runner, cleanup


async def _run_suite(
    resolved: ResolvedConfig, suite: BenchmarkSuite, *, run_id: str | None
) -> BenchmarkRun:
    runner, cleanup = await build_runner(resolved)
    try:
        return await runner.run(suite, run_id=run_id)
    finally:
        await cleanup()


def _overrides(base_url: str | None, attach: bool | None) -> dict[str, object]:
    flags: dict[str, object] = {}
    if base_url is not None:
        flags["runtime.base_url"] = base_url
    if attach is not None:
        flags["runtime.attach"] = attach
    return flags


def _run_command(
    kind: str,
    *,
    suite_override: Path | None,
    out: Path,
    run_id: str | None,
    base_url: str | None,
    attach: bool | None,
    config_file: Path | None,
    profile: Path | None,
    case_ids: list[str] | None,
    as_json: bool,
) -> None:
    """The shared body of ``capability``, ``flags`` and ``tasks``."""
    try:
        if profile is not None:
            resolved = load_config(
                config_file=config_file,
                overrides=_overrides(base_url, attach),
                profile_file=profile,
            )
        else:
            resolved = load_config(
                config_file=config_file, overrides=_overrides(base_url, attach)
            )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG) from exc

    suite_path = suite_override or SUITE_PATHS[kind]
    try:
        suite = load_suite(suite_path)
        if case_ids:
            # ``select`` refuses an unknown id rather than returning an empty,
            # successful-looking run — the same policy the loader follows.
            suite = suite.select(case_ids)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc

    for warning in resolved.warnings:
        err_console.print(f"[yellow]![/yellow] {warning}")

    try:
        run = asyncio.run(_run_suite(resolved, suite, run_id=run_id))
    except _RUNTIME_FAILURES as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        raise typer.Exit(EXIT_ERROR) from None

    path = write_run(run, out)
    if as_json:
        # stdout is the data, stderr is the chatter: a caller parsing stdout
        # gets exactly the run document and nothing that breaks json.loads.
        console.print_json(run.model_dump_json())
    else:
        console.print(render_summary(run))
    err_console.print(f"[dim]artefact: {path}[/dim]")

    # An invalid run printed its INVALID banner above; the exit code keeps a
    # caller that treats 0 as "trustworthy measurement" from buying one.
    raise typer.Exit(EXIT_OK if run.valid else EXIT_INVALID)


# -- subcommands -----------------------------------------------------------


@benchmark_app.command("capability")
def capability(
    suite: SuiteOpt = None,
    out: OutOpt = DEFAULT_OUT,
    run_id: RunIdOpt = None,
    base_url: BaseUrlOpt = None,
    attach: AttachOpt = None,
    config: ConfigOpt = None,
    profile: ProfileOpt = None,
    case: CaseOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Run the model-capability suite (structured output, tool calls, cache)."""
    _run_command(
        "capability",
        suite_override=suite, out=out, run_id=run_id, base_url=base_url,
        attach=attach, config_file=config, profile=profile,
        case_ids=case, as_json=json_output,
    )


@benchmark_app.command("flags")
def flags(
    suite: SuiteOpt = None,
    out: OutOpt = DEFAULT_OUT,
    run_id: RunIdOpt = None,
    base_url: BaseUrlOpt = None,
    attach: AttachOpt = None,
    config: ConfigOpt = None,
    profile: ProfileOpt = None,
    case: CaseOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Run the launch-flag sweep suite (canonical suite lands with #20)."""
    _run_command(
        "flags",
        suite_override=suite, out=out, run_id=run_id, base_url=base_url,
        attach=attach, config_file=config, profile=profile,
        case_ids=case, as_json=json_output,
    )


@benchmark_app.command("tasks")
def tasks(
    suite: SuiteOpt = None,
    out: OutOpt = DEFAULT_OUT,
    run_id: RunIdOpt = None,
    base_url: BaseUrlOpt = None,
    attach: AttachOpt = None,
    config: ConfigOpt = None,
    profile: ProfileOpt = None,
    case: CaseOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Run the task-comparison suite (canonical suite lands with #21)."""
    _run_command(
        "tasks",
        suite_override=suite, out=out, run_id=run_id, base_url=base_url,
        attach=attach, config_file=config, profile=profile,
        case_ids=case, as_json=json_output,
    )


@benchmark_app.command("compare")
def compare(
    a: Annotated[Path, typer.Argument(help="First run artefact (.json).")],
    b: Annotated[Path, typer.Argument(help="Second run artefact (.json).")],
    json_output: JsonOpt = False,
) -> None:
    """Set two runs side by side and state whether they are comparable.

    Exit codes: 0 both runs valid, 3 at least one run was invalidated, 1 an
    artefact could not be read. A fingerprint mismatch (different host, model
    or runtime) is printed as a loud NOT COMPARABLE banner but does not change
    the exit code — both runs are valid individually, the comparison is shown
    so a reader can see why the numbers diverge.
    """
    try:
        run_a = read_run(a)
        run_b = read_run(b)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc

    if json_output:
        console.print_json(
            json.dumps(
                {
                    "a": run_a.model_dump(mode="json"),
                    "b": run_b.model_dump(mode="json"),
                    "valid": [run_a.valid, run_b.valid],
                }
            )
        )
    else:
        console.print(render_comparison(run_a, run_b))

    raise typer.Exit(EXIT_OK if run_a.valid and run_b.valid else EXIT_INVALID)


__all__ = [
    "EXIT_CONFIG",
    "EXIT_ERROR",
    "EXIT_INVALID",
    "EXIT_OK",
    "benchmark_app",
    "build_runner",
]