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

SECRET_FLAG_SUFFIXES: tuple[str, ...] = ("-key", "-token", "-password", "-secret")
"""Server flags whose *next* argument is a credential.

``extra_flags`` is passed to llama-server verbatim, and the real form is
``--api-key VALUE`` — two list entries, no ``=``. The text scrubber only ever
matched ``name=value``, so this shape went straight to the terminal through
``config show``. Matching on the suffix rather than a fixed list means a flag
this harness has not heard of still gets its value hidden."""


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

    adjustments: list[str] = Field(default_factory=list, exclude=True)
    """Corrections applied during validation, surfaced as warnings."""

    @model_validator(mode="after")
    def _ceiling_fits_the_window(self) -> ContextConfig:
        """Lower the advisory limit to the hard one instead of refusing.

        Refusing looked principled until a hardware profile measuring a
        smaller context blocked every command on the machine it was measured
        on: the layer meant to be advisory had become a gate. The pair still
        cannot be left inconsistent — a soft ceiling above the window would
        let the budget grow the prompt past what the server accepts — so it is
        clamped, and the adjustment is reported rather than applied quietly.
        """
        if self.soft_ceiling > self.context_window:
            self.adjustments.append(
                f"soft_ceiling lowered from {self.soft_ceiling} to "
                f"{self.context_window} to fit the context window"
            )
            self.soft_ceiling = self.context_window
        return self


class SandboxConfig(_Strict):
    """Shell sandbox policy. Fail-closed defaults, matching `tools.shell`."""

    network: NetworkMode = NetworkMode.ISOLATED
    require_sandbox: bool = True
    """Refuse to run untrusted commands without bubblewrap rather than
    running them unsandboxed."""


class StrictBudget(Budget):
    """``core.Budget`` with unknown keys refused.

    A subclass rather than a change to ``Budget`` itself: the loop takes the
    base type and is unaffected, while a ``max_stpes`` in a config file stops
    being silently dropped. Configuration is the one place where a typo is
    indistinguishable from a decision.
    """

    model_config = ConfigDict(extra="forbid")


class HarnessConfig(_Strict):
    """The whole effective configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    budget: StrictBudget = Field(default_factory=StrictBudget)
    """The bounds the loop enforces, typed as ``core.Budget`` — not a parallel
    model that would have to be kept in step with it."""
    workspace: Path = Field(default_factory=Path.cwd)
    database: Path = Path(".harness/memory.sqlite")

    @model_validator(mode="after")
    def _server_window_and_budget_window_agree(self) -> HarnessConfig:
        """Keep ``model.n_ctx`` and ``context.context_window`` from diverging.

        ``n_ctx`` is what the server is started with; ``context_window`` is
        what ``TokenBudget`` believes it may fill. Two numbers for one window
        means the budget can grow a prompt past what the server accepts, and
        the overflow arrives as a runtime error rather than a config one. The
        launched value wins where it is set and the other is untouched.
        """
        if self.model.n_ctx is None:
            return self
        if self.context.context_window != self.model.n_ctx:
            self.context.adjustments.append(
                f"context_window set to {self.model.n_ctx} to match the "
                "context size the server is launched with (model.n_ctx)"
            )
            self.context.context_window = self.model.n_ctx
            if self.context.soft_ceiling > self.context.context_window:
                self.context.adjustments.append(
                    f"soft_ceiling lowered to {self.context.context_window} "
                    "to fit that window"
                )
                self.context.soft_ceiling = self.context.context_window
        return self


__all__ = [
    "SECRET_FIELDS",
    "ContextConfig",
    "HarnessConfig",
    "ModelConfig",
    "RuntimeConfig",
    "SandboxConfig",
    "StrictBudget",
]
