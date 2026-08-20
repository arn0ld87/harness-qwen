"""Typed configuration for runtime, model, context, sandbox and budgets.

Deliberately small. The harness has one runtime, one model and one workspace,
so this is a handful of typed fields with validation — not a settings
framework with layers, plugins and profiles.

Every default here *points at* the definition that already owns the value
instead of restating it. Two owners of the same number is how a tuned soft
ceiling silently stops applying: someone raises it in ``budget.py`` and the
config keeps handing out the old one.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.context.budget import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_GENERATION_RESERVE_RATIO,
    DEFAULT_SOFT_CEILING,
)
from harness.core import Budget, NetworkMode
from harness.models.llamacpp import DEFAULT_BASE_URL, DEFAULT_HOST, DEFAULT_PORT

SECRET_FIELDS: frozenset[str] = frozenset({"api_key"})
"""Fields never printed in the clear. Kept as names rather than a pattern so a
new secret has to be declared, not merely hoped to match."""


class _Strict(BaseModel):
    """Reject unknown keys everywhere.

    A misspelled key that is silently ignored looks exactly like a setting
    that took effect, and the run it misconfigures is the one nobody debugs.
    """

    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(_Strict):
    """How to reach — or start — the inference server."""

    base_url: str = DEFAULT_BASE_URL
    host: str = DEFAULT_HOST
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    server_binary: Path | None = None
    """``llama-server`` to launch. None means attach only."""
    attach: bool = False
    """Use a server this harness did not start. Explicit on purpose: an
    accidental attach measures somebody else's process (#11)."""
    startup_timeout_s: float = Field(default=180.0, gt=0)
    api_key: str | None = None

    @model_validator(mode="after")
    def _url_follows_host_and_port(self) -> RuntimeConfig:
        """Derive ``base_url`` from host/port unless it was set explicitly.

        Two independent ways to say where the server is means one of them is
        wrong half the time — and the half that is wrong is the one nobody
        looks at. Setting only the port therefore moves the URL too; setting
        the URL explicitly wins, because that is the attach case where host
        and port describe a server this harness did not start.
        """
        derived = f"http://{self.host}:{self.port}"
        if self.base_url == DEFAULT_BASE_URL and derived != DEFAULT_BASE_URL:
            self.base_url = derived
        return self


class ModelConfig(_Strict):
    """Which weights to serve and how to serve them."""

    path: Path | None = None
    alias: str | None = None
    n_ctx: int | None = Field(default=None, ge=512)
    n_gpu_layers: int | None = Field(default=None, ge=0)
    threads: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    extra_flags: list[str] = Field(default_factory=list)
    """Passed to llama-server verbatim, after the typed flags."""


class ContextConfig(_Strict):
    """Context accounting. Owned by ``context.budget``; mirrored here."""

    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, ge=512)
    soft_ceiling: int = Field(default=DEFAULT_SOFT_CEILING, ge=256)
    generation_reserve_ratio: float = Field(
        default=DEFAULT_GENERATION_RESERVE_RATIO, gt=0.0, lt=1.0
    )

    @model_validator(mode="after")
    def _ceiling_fits_the_window(self) -> ContextConfig:
        if self.soft_ceiling > self.context_window:
            raise ValueError(
                f"soft_ceiling {self.soft_ceiling} exceeds context_window "
                f"{self.context_window}: the advisory limit cannot sit above "
                "the hard one"
            )
        return self


class SandboxConfig(_Strict):
    """Shell sandbox policy. Fail-closed defaults, matching `tools.shell`."""

    network: NetworkMode = NetworkMode.ISOLATED
    require_sandbox: bool = True
    """Refuse to run untrusted commands without bubblewrap rather than
    running them unsandboxed."""


class HarnessConfig(_Strict):
    """The whole effective configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    budget: Budget = Field(default_factory=Budget)
    """The same ``core.Budget`` the loop enforces — not a parallel model that
    would have to be kept in step with it."""
    workspace: Path = Field(default_factory=Path.cwd)
    database: Path = Path(".harness/memory.sqlite")


__all__ = [
    "SECRET_FIELDS",
    "ContextConfig",
    "HarnessConfig",
    "ModelConfig",
    "RuntimeConfig",
    "SandboxConfig",
]
