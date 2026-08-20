"""Command line interface.

``harness doctor`` is the entry point that matters: it reports what the machine
actually is, what runtime is actually serving, and what the model actually
contains — then names what is still unknown instead of filling the gap with a
plausible number.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from harness import __version__
from harness.config import ConfigError, load_config
from harness.diagnostics import Diagnosis, Severity, diagnose
from harness.discovery import build_profile, save_profile
from harness.discovery.models import HardwareProfile

app = typer.Typer(
    name="harness",
    help="A hardware-aware local agent harness for Qwen3.6-35B-A3B.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

GIB = 1024 ** 3
MIB = 1024 ** 2
DEFAULT_PROFILE_PATH = Path("config/hardware-profile.json")

UNKNOWN = "[dim]unknown[/dim]"


def _gib(value: int | None) -> str:
    return f"{value / GIB:.1f} GiB" if value else UNKNOWN


def _section(title: str) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    console.print(f"\n[bold]{title}[/bold]")
    return table


def _render_hardware(p: HardwareProfile) -> None:
    t = _section("Hardware")
    cpu = p.cpu
    simd = ", ".join(filter(None, [
        "AVX2" if cpu.has_avx2 else None,
        "AVX-512" if cpu.has_avx512 else None,
    ])) or "no AVX2"
    cores = (f"{cpu.physical_cores}C / {cpu.logical_cores}T"
             if cpu.physical_cores else UNKNOWN)
    t.add_row("CPU", f"{cpu.model_name or UNKNOWN}")
    t.add_row("Cores", f"{cores}   [dim]{simd}[/dim]")
    t.add_row("RAM", f"{_gib(p.memory.total_bytes)} total, "
                     f"{_gib(p.memory.available_bytes)} available")

    if p.memory.swap_total_bytes:
        kind = "zram (compressed RAM)" if p.memory.swap_is_zram else "disk"
        used = p.memory.swap_used_bytes
        style = "yellow" if used > GIB else "dim"
        t.add_row("Swap", f"[{style}]{_gib(used)} used of "
                          f"{_gib(p.memory.swap_total_bytes)} — {kind}[/{style}]")

    for gpu in p.gpus:
        free = ((gpu.vram_total_bytes or 0) - (gpu.vram_used_bytes or 0))
        t.add_row("GPU", f"{gpu.name or UNKNOWN}")
        t.add_row("VRAM", f"{_gib(gpu.vram_total_bytes)} total, {_gib(free)} free")
        t.add_row("Driver", f"{gpu.driver_version or UNKNOWN}"
                            + (f"   [dim]CUDA {gpu.cuda_version}[/dim]"
                               if gpu.cuda_version else ""))
    if not p.gpus:
        t.add_row("GPU", f"{UNKNOWN} [dim](no vendor tool found — CPU inference)[/dim]")

    t.add_row("OS", f"{p.os_release or UNKNOWN}   [dim]{p.kernel or ''}[/dim]")

    if p.sandbox.available:
        t.add_row("Sandbox", "bwrap  [dim]network isolated by default[/dim]")
    else:
        t.add_row("Sandbox", "[red]bwrap missing[/red]")
    console.print(t)


def _render_runtime(p: HardwareProfile) -> None:
    t = _section("Runtime")
    if not p.runtimes:
        t.add_row("Status", "[red]no local inference endpoint answered[/red]")
        console.print(t)
        return

    primary = p.primary_runtime
    for rt in p.runtimes:
        marker = "[green]→[/green]" if rt is primary else " "
        detail = rt.kind
        if rt.requires_auth:
            detail += " [yellow](auth required)[/yellow]"
        elif rt.model_ids:
            detail += f" [dim]{', '.join(rt.model_ids[:3])}[/dim]"
        t.add_row(f"{marker} {rt.base_url}", detail)

    if primary:
        if primary.build_info:
            t.add_row("Build", primary.build_info)
        if primary.n_ctx:
            t.add_row("Context", f"{primary.n_ctx:,} tokens"
                                 + (f" across {primary.total_slots} slots "
                                    f"({primary.n_ctx // primary.total_slots:,} each)"
                                    if primary.total_slots and primary.total_slots > 1
                                    else ""))
        if primary.server_pid:
            t.add_row("Process", f"pid {primary.server_pid}")
        caps = [k.removeprefix("supports_") for k, v in primary.chat_template_caps.items()
                if v and k.startswith("supports_")]
        if caps:
            t.add_row("Template", ", ".join(sorted(caps)))
    console.print(t)


def _render_model(p: HardwareProfile) -> None:
    t = _section("Model")
    m = p.model
    if m is None:
        t.add_row("Status", "[yellow]no model file located[/yellow]")
        console.print(t)
        return

    t.add_row("File", str(m.path or UNKNOWN))
    if m.file_size_bytes:
        t.add_row("Size", _gib(m.file_size_bytes))
    if m.name and m.path and m.name not in m.path.name:
        # The filename and the embedded name disagreeing is worth surfacing:
        # a rebranded merge is easy to mistake for a different model family.
        t.add_row("Name", f"{m.name}  [yellow](differs from filename)[/yellow]")
    elif m.name:
        t.add_row("Name", m.name)
    t.add_row("Architecture", f"{m.architecture or UNKNOWN}   "
                              f"[dim]{m.quantization or ''}[/dim]")

    if m.n_layers:
        layers = f"{m.n_layers} trunk"
        if m.is_hybrid and m.n_dense_attention_layers is not None:
            layers += (f"  [dim]({m.n_dense_attention_layers} dense attention, "
                       f"{m.n_recurrent_layers} recurrent)[/dim]")
        if m.has_mtp:
            layers += f"  [yellow]+{m.n_mtp_layers} MTP[/yellow]"
        t.add_row("Layers", layers)

    if m.is_moe:
        t.add_row("Experts", f"{m.expert_count} total, {m.expert_used_count} active")
    if m.context_length_train:
        t.add_row("Trained ctx", f"{m.context_length_train:,} tokens")
    if m.embedded_sampling:
        t.add_row("Sampling", "  ".join(f"{k}={v:g}"
                                        for k, v in sorted(m.embedded_sampling.items()))
                              + "  [dim](embedded by author)[/dim]")
    console.print(t)


def _render_recommendations(p: HardwareProfile) -> None:
    t = _section("Recommendations")
    r = p.recommended
    values = {
        "Parallel requests": str(r.parallel_agents),
        "Threads": str(r.threads) if r.threads else None,
        "RAM reserve": _gib(r.ram_reserve_bytes) if r.ram_reserve_bytes else None,
        "Context length": f"{r.context_length:,}" if r.context_length else None,
        "Max output": f"{r.max_output_tokens:,}" if r.max_output_tokens else None,
        "GPU layers": str(r.gpu_layers) if r.gpu_layers is not None else None,
        "Batch size": str(r.batch_size) if r.batch_size is not None else None,
    }
    for label, value in values.items():
        t.add_row(label, value if value else "[dim]not measured yet[/dim]")
    console.print(t)

    if r.rationale:
        console.print("\n[bold]Why[/bold]")
        for key, reason in sorted(r.rationale.items()):
            console.print(f"  [cyan]{key}[/cyan]\n    [dim]{reason}[/dim]")


def _render_warnings(p: HardwareProfile) -> None:
    warnings: list[str] = []

    if not p.sandbox.available:
        warnings.append(
            "Bubblewrap (bwrap) not found. The shell sandbox is fail-closed: "
            "without it, untrusted commands are denied rather than run "
            "unsandboxed. Install bwrap before any agent run that may issue "
            "such commands."
        )

    if p.memory.swap_is_zram and p.memory.swap_used_bytes > GIB:
        warnings.append(
            f"{_gib(p.memory.swap_used_bytes)} of zram in use. Swap here is "
            "compressed RAM, so this costs CPU time the model already needs. "
            "Benchmarks taken in this state are not trustworthy."
        )

    primary = p.primary_runtime
    if primary and primary.total_slots and primary.total_slots > 1:
        warnings.append(
            f"Server offers {primary.total_slots} slots. A second concurrent "
            "request roughly halves generation throughput on a single GPU; "
            "the agent loop uses one."
        )

    m = p.model
    if m and m.has_mtp:
        warnings.append(
            f"Model ships {m.n_mtp_layers} multi-token-prediction block(s). Unless "
            "speculative decoding is enabled explicitly, they are loaded and "
            "discarded — memory spent for nothing."
        )

    if not warnings:
        return
    console.print("\n[bold yellow]Attention[/bold yellow]")
    for w in warnings:
        console.print(f"  [yellow]•[/yellow] {w}")


_SEVERITY_STYLE = {
    Severity.OK: ("[green]ok[/green]  ", ""),
    Severity.WARN: ("[yellow]warn[/yellow]", "yellow"),
    Severity.FAIL: ("[red]fail[/red]", "red"),
    Severity.SKIP: ("[dim]skip[/dim]", "dim"),
}


def _render_checks(diagnosis: Diagnosis) -> None:
    console.print("\n[bold]Readiness[/bold]")
    for check in diagnosis.checks:
        marker, style = _SEVERITY_STYLE[check.severity]
        body = f"[{style}]{check.detail}[/{style}]" if style else check.detail
        console.print(f"  {marker}  [cyan]{check.name}[/cyan]  {body}")
        if check.hint:
            console.print(f"        [dim]{check.hint}[/dim]")


@app.command()
def doctor(
    model: Annotated[
        Path | None,
        typer.Option("--model", "-m", help="GGUF path; defaults to the running server."),
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Write config/hardware-profile.json."),
    ] = True,
    profile_path: Annotated[
        Path,
        typer.Option("--profile-path", help="Where to write the profile."),
    ] = DEFAULT_PROFILE_PATH,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the readiness checks as JSON and exit."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero on warnings too."),
    ] = False,
) -> None:
    """Probe hardware, runtime and model, then report what is known and what is not.

    Exit codes: 0 usable, 1 something would stop a run, 2 warnings under
    ``--strict``. Nothing here changes the system apart from writing the
    hardware profile, which ``--no-save`` turns off.
    """
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    diagnosis = asyncio.run(diagnose(config))

    if as_json:
        payload = {
            "checks": [check.model_dump(mode="json") for check in diagnosis.checks],
            "exit_code": diagnosis.exit_code(strict=strict),
        }
        console.print_json(json.dumps(payload))
        raise typer.Exit(diagnosis.exit_code(strict=strict))

    profile = asyncio.run(build_profile(model_path=model))

    console.print(f"[bold]harness-qwen[/bold] [dim]{__version__}[/dim]")
    _render_hardware(profile)
    _render_runtime(profile)
    _render_model(profile)
    _render_recommendations(profile)
    _render_checks(diagnosis)
    _render_warnings(profile)

    for warning in config.warnings:
        console.print(f"  [yellow]•[/yellow] {warning}")

    if save:
        written = save_profile(profile, profile_path)
        console.print(f"\n[dim]profile written to {written}[/dim]")

    raise typer.Exit(diagnosis.exit_code(strict=strict))


@app.command("model-info")
def model_info(
    path: Annotated[Path, typer.Argument(help="Path to a GGUF file.")],
) -> None:
    """Print metadata read from a GGUF file, including the hybrid layer split."""
    from harness.discovery import read_gguf_metadata
    from harness.discovery.gguf import GgufError

    try:
        meta = read_gguf_metadata(path)
    except GgufError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    t.add_column(style="cyan", no_wrap=True)
    t.add_column()
    t.add_row("Architecture", meta.architecture or UNKNOWN)
    t.add_row("Name", meta.name or UNKNOWN)
    t.add_row("Quantisation", meta.quantization or UNKNOWN)
    t.add_row("Tensors", str(meta.tensor_count))
    t.add_row("Trunk layers", str(meta.n_trunk_layers or UNKNOWN))
    if split := meta.attention_layer_split():
        t.add_row("Layer split", f"{split[0]} dense attention, {split[1]} recurrent")
    if meta.n_mtp_layers:
        t.add_row("MTP blocks", str(meta.n_mtp_layers))
    console.print(t)

    console.print("\n[bold]KV cache per context size[/bold]")
    kv = Table(box=None, padding=(0, 2, 0, 0))
    kv.add_column("cache type", style="cyan")
    kv.add_column("per token", justify="right")
    for size in (8192, 16384, 32768, 65536):
        kv.add_column(f"{size // 1024}k", justify="right")
    for ctype in ("f16", "q8_0", "q4_0"):
        per_token = meta.kv_cache_bytes_per_token(ctype, ctype)
        if per_token is None:
            continue
        row = [ctype, f"{per_token:,.0f} B"]
        row += [f"{per_token * s / MIB:,.0f} MiB" for s in (8192, 16384, 32768, 65536)]
        kv.add_row(*row)
    console.print(kv)

    if (state := meta.recurrent_state_bytes()) is not None:
        console.print(f"\n[dim]recurrent state: {state / MIB:,.0f} MiB, "
                      f"independent of context length[/dim]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
