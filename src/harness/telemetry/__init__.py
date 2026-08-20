"""Telemetry subsystem for run tracking.

Provides secure, structured event logging with automatic secret redaction.
Designed to be incapable of recording model reasoning or any sensitive content.
"""

from __future__ import annotations

from .journal import RunJournal
from .redact import redact, redact_data

__all__ = [
    "RunJournal",
    "redact",
    "redact_data",
]
