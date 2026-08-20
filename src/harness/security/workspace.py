"""Workspace confinement: resolve and validate file paths stay within bounds."""

from __future__ import annotations

from pathlib import Path

from harness.core import ToolError


def resolve_in_workspace(path: str | Path, workspace_root: Path) -> Path:
    """Resolve a path within a workspace, enforcing confinement.

    Resolves symlinks before checking containment. Raises ToolError(kind="denied")
    if the resolved path escapes the workspace root. Non-existent paths whose
    parent directories are within the workspace are allowed (e.g., files about
    to be created).

    Args:
        path: File or directory path, relative or absolute.
        workspace_root: The workspace boundary; must be absolute.

    Returns:
        The resolved path, guaranteed to be within workspace_root.

    Raises:
        ToolError: If the resolved path escapes the workspace or is invalid.
    """
    path = Path(path)
    workspace_root = workspace_root.resolve()

    try:
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (workspace_root / path).resolve()
    except (OSError, RuntimeError) as e:
        raise ToolError(f"Cannot resolve path: {e}", kind="denied")

    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise ToolError(
            f"Path {resolved} is outside workspace {workspace_root}",
            kind="denied",
        )

    return resolved
