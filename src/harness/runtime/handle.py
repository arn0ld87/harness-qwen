"""What the harness knows about the server it is talking to.

The distinction that matters is ownership. A server this process started can
be stopped; one that was already running belongs to somebody else, and taking
it down destroys a model load that costs minutes to rebuild. That is a
different *state*, not a flag to remember to check, so it is on the handle and
``stop`` refuses rather than trusting the caller.
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Ownership(enum.StrEnum):
    """Whether this harness started the server it is using."""

    OWNED = "owned"
    ATTACHED = "attached"


class RuntimeHandle(BaseModel):
    """Identity of one inference server, as far as this process can tell.

    ``pid`` and ``started_at`` are what a later check uses to prove the
    endpoint answering on the port is still the process that was started, and
    not a stale one that inherited it (#11).
    """

    base_url: str
    ownership: Ownership
    pid: int | None = None
    started_at: datetime | None = None
    log_path: Path | None = None

    @property
    def owned(self) -> bool:
        return self.ownership is Ownership.OWNED


class RuntimeError_(RuntimeError):
    """Base class for supervisor failures."""


class RuntimeStartError(RuntimeError_):
    """The server did not come up."""


class RuntimeCrashed(RuntimeStartError):
    """The process exited. Carries its status and last output.

    Separate from a timeout because the two call for opposite responses:
    a crash is a configuration or model problem to read the log for, a
    timeout may just be a slow load on a slow disk.
    """


class RuntimeStartTimeout(RuntimeStartError):
    """The deadline passed while the server was still not answering."""


class RuntimeNotOwned(RuntimeError_):
    """A stop was requested for a server this harness did not start."""


__all__ = [
    "Ownership",
    "RuntimeCrashed",
    "RuntimeError_",
    "RuntimeHandle",
    "RuntimeNotOwned",
    "RuntimeStartError",
    "RuntimeStartTimeout",
]
