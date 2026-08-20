"""Detection of local inference runtimes.

Ports are probed, never assumed. In addition to network probing, running
processes are inspected so the harness learns the exact launch flags of a
``llama-server`` it did not start itself — those flags are what a performance
profile has to compare against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from harness.discovery.models import RuntimeInfo

# Ports commonly used by local inference servers. The process scan below finds
# anything listening elsewhere, so this list is a fast path, not a limit.
DEFAULT_PORTS: tuple[int, ...] = (1234, 8080, 8000, 11434, 18080, 5001, 4891, 9997)

_PROC = Path("/proc")
_SERVER_BINARIES = ("llama-server", "llama-cli", "ollama", "vllm", "mlx_lm")


def _classify(base_url: str, payload: dict[str, Any] | None) -> str:
    if payload and payload.get("data") and isinstance(payload["data"], list):
        first = payload["data"][0] if payload["data"] else {}
        if isinstance(first, dict) and first.get("owned_by") == "llamacpp":
            return "llama-server"
    if ":11434" in base_url:
        return "ollama"
    if ":1234" in base_url:
        return "lmstudio"
    return "openai-compatible" if payload else "unknown"


def _scan_processes() -> dict[int, tuple[int, list[str]]]:
    """Map port -> (pid, argv) for running inference servers.

    Reads ``/proc/<pid>/cmdline`` and extracts the ``--port`` argument. Only
    processes owned by the current user are readable, which is the intended
    scope.
    """
    found: dict[int, tuple[int, list[str]]] = {}
    try:
        pids = [int(p.name) for p in _PROC.iterdir() if p.name.isdigit()]
    except OSError:
        return found

    for pid in pids:
        try:
            raw = (_PROC / str(pid) / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if not raw:
            continue
        argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
        if not argv:
            continue
        exe = Path(argv[0]).name
        if not any(exe.startswith(b) for b in _SERVER_BINARIES):
            continue
        port: int | None = None
        for i, arg in enumerate(argv):
            if arg == "--port" and i + 1 < len(argv) and argv[i + 1].isdigit():
                port = int(argv[i + 1])
            elif m := re.fullmatch(r"--port=(\d+)", arg):
                port = int(m.group(1))
        if port is not None:
            found[port] = (pid, argv)
    return found


async def _probe_endpoint(client: httpx.AsyncClient, port: int,
                          host: str = "127.0.0.1") -> RuntimeInfo | None:
    """Probe one port. Returns None when nothing answered."""
    base_url = f"http://{host}:{port}"
    payload: dict[str, Any] | None = None
    requires_auth = False

    try:
        response = await client.get(f"{base_url}/v1/models", timeout=2.0)
    except (httpx.HTTPError, OSError):
        return None

    if response.status_code in (401, 403):
        requires_auth = True
    elif response.status_code == 200:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            payload = None
    elif response.status_code >= 500:
        # A server that is up but still loading answers 503; still worth reporting.
        pass
    else:
        return None

    info = RuntimeInfo(
        kind=_classify(base_url, payload),
        base_url=base_url,
        reachable=True,
        requires_auth=requires_auth,
    )

    if payload and isinstance(payload.get("data"), list):
        info.model_ids = [
            m["id"] for m in payload["data"]
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        ]

    # /props is llama-server specific and carries the facts we actually need.
    try:
        props = await client.get(f"{base_url}/props", timeout=2.0)
        if props.status_code == 200:
            data = props.json()
            info.build_info = data.get("build_info")
            info.model_path = data.get("model_path")
            info.total_slots = data.get("total_slots")
            caps = data.get("chat_template_caps")
            if isinstance(caps, dict):
                info.chat_template_caps = {
                    k: v for k, v in caps.items() if isinstance(v, bool)
                }
            gen = data.get("default_generation_settings")
            if isinstance(gen, dict) and isinstance(gen.get("n_ctx"), int):
                info.n_ctx = gen["n_ctx"]
            if info.kind == "unknown":
                info.kind = "llama-server"
    except (httpx.HTTPError, OSError, json.JSONDecodeError, ValueError):
        pass

    return info


async def probe_runtimes(ports: tuple[int, ...] = DEFAULT_PORTS,
                         host: str = "127.0.0.1") -> list[RuntimeInfo]:
    """Probe local inference endpoints and attach process information.

    Ports discovered by scanning running processes are probed in addition to
    the defaults, so a server on an unusual port is still found.
    """
    processes = _scan_processes()
    all_ports = tuple(dict.fromkeys((*ports, *processes.keys())))

    results: list[RuntimeInfo] = []
    async with httpx.AsyncClient() as client:
        for port in all_ports:
            info = await _probe_endpoint(client, port, host)
            if info is None:
                continue
            if port in processes:
                pid, argv = processes[port]
                info.server_pid = pid
                info.server_argv = argv
            results.append(info)
    return results
