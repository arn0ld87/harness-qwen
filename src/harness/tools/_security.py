"""Adapter onto :mod:`harness.security`.

The tool layer never decides whether a command may run or whether a path is
inside the workspace — that decision belongs to the security package and is
deterministic code, not a model judgment (ARCHITECTURE.md, Security boundary).

Two things justify the indirection instead of a plain import. The import is
resolved at call time, so every tool here is unit-testable against an injected
policy without the security package being loaded. And when the security package
cannot be imported at all, both entry points fail closed — an unclassifiable
command lands in CONFIRM (which denies without an approver) and an
unresolvable path is refused outright, rather than running unchecked.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from harness.core import Risk, ToolError

Classifier = Callable[[str], Risk]
"""Maps a full command line to its risk class."""

PathResolver = Callable[[Path, str], Path]
"""Maps ``(workspace, path)`` to an absolute path proven to be inside it."""


def default_classifier(command: str, workspace: Path | None = None) -> Risk:
    """Classify ``command`` with the workspace-aware security boundary."""
    try:
        from harness.security.classifier import classify_command
    except ImportError:
        return Risk.CONFIRM
    risk, _ = classify_command(command, workspace=workspace)
    return _coerce_risk(risk)


def default_resolver(workspace: Path, path: str) -> Path:
    """Resolve ``path`` inside ``workspace`` via ``harness.security.workspace``.

    Any failure is reported as ``denied``: a path the security layer refused to
    resolve is a path the tool layer must not touch, whatever the reason was.
    """
    try:
        from harness.security.workspace import resolve_in_workspace
    except ImportError as exc:
        raise ToolError(
            f"workspace containment is unavailable: {exc}", kind="denied"
        ) from exc
    try:
        return Path(resolve_in_workspace(path, workspace))
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"path refused: {path} ({exc})", kind="denied") from exc


def _coerce_risk(value: object) -> Risk:
    """Accept a Risk, its string value, or a decision object wrapping one.

    Anything else resolves to CONFIRM, matching the documented rule that an
    unrecognised classification is not a safe classification.
    """
    if isinstance(value, Risk):
        return value
    if isinstance(value, str):
        try:
            return Risk(value)
        except ValueError:
            return Risk.CONFIRM
    inner = getattr(value, "risk", None)
    if inner is not None and not isinstance(inner, type(value)):
        return _coerce_risk(inner)
    return Risk.CONFIRM
