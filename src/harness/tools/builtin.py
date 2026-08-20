"""The tool set an agent run is given, declared once.

The tool functions existed before this module; what was missing was the
declaration that lets a model see them and the registry call them. Both live
here so that adding a tool means adding one entry, and so the two facts that
decide how a tool is treated — its security risk and whether repeating it is
safe — are stated next to each other rather than inferred at three call sites.

Every parameter schema sets ``additionalProperties: false``. A model that
invents an argument gets a validation error it can correct on the next step,
which is cheaper than a tool doing something unintended with it.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from harness.core import NetworkMode, Risk, SideEffect, ToolSpec
from harness.tools.filesystem import list_files, read_file, search_files, write_file
from harness.tools.git import git_diff, git_log, git_status
from harness.tools.registry import ToolRegistry
from harness.tools.shell import ConfirmCallback, run_command


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file from the workspace. Returns numbered lines.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object(
        {
            "path": {"type": "string", "description": "Path relative to the workspace."},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
        },
        ["path"],
    ),
)

WRITE_FILE = ToolSpec(
    name="write_file",
    description="Write a file in the workspace, creating parent directories.",
    risk=Risk.ALLOW,
    # Writing the same content twice leaves the same file: the effect is on
    # disk either way, and a resumed run may safely repeat it.
    side_effect=SideEffect.IDEMPOTENT,
    parameters=_object(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    ),
)

LIST_FILES = ToolSpec(
    name="list_files",
    description="List files in the workspace, honouring .gitignore.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object(
        {
            "path": {"type": "string"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 10},
        }
    ),
)

SEARCH_FILES = ToolSpec(
    name="search_files",
    description="Search the workspace for a pattern, with optional regex.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "regex": {"type": "boolean"},
        },
        ["pattern"],
    ),
)

RUN_COMMAND = ToolSpec(
    name="run_command",
    description=(
        "Run a shell command in the workspace sandbox. Commands are classified "
        "before execution; anything unrecognised needs approval."
    ),
    # CONFIRM at the tool level would ask about `ls` too. The per-command
    # classifier inside run_command is the real boundary, and it is stricter
    # than a single risk level on the tool could be.
    risk=Risk.ALLOW,
    side_effect=SideEffect.MUTATING,
    timeout_s=180.0,
    parameters=_object(
        {
            "command": {"type": "string"},
            "timeout": {"type": "number", "minimum": 1, "maximum": 600},
        },
        ["command"],
    ),
)

GIT_STATUS = ToolSpec(
    name="git_status",
    description="Show the working tree status.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object({}),
)

GIT_DIFF = ToolSpec(
    name="git_diff",
    description="Show the diff, optionally for one path or revision.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object({"path": {"type": "string"}, "revision": {"type": "string"}}),
)

GIT_LOG = ToolSpec(
    name="git_log",
    description="Show recent commits.",
    risk=Risk.ALLOW,
    side_effect=SideEffect.NONE,
    parameters=_object(
        {
            "max_count": {"type": "integer", "minimum": 1, "maximum": 100},
            "follow": {"type": "boolean"},
        }
    ),
)

BUILTIN_SPECS: tuple[ToolSpec, ...] = (
    GIT_DIFF,
    GIT_LOG,
    GIT_STATUS,
    LIST_FILES,
    READ_FILE,
    RUN_COMMAND,
    SEARCH_FILES,
    WRITE_FILE,
)


def build_registry(
    workspace: Path,
    *,
    confirm: ConfirmCallback | None = None,
    network: NetworkMode = NetworkMode.ISOLATED,
    read_only: bool = False,
) -> ToolRegistry:
    """Register the built-in tools against one workspace.

    ``workspace`` is bound here rather than passed by the model: it is the
    boundary every path is resolved against, and a tool argument the model
    controls would not be a boundary at all.

    ``read_only`` leaves out everything that can change the world. Useful for
    a run that is only meant to look, where the cheapest way to guarantee that
    is not to offer the tools.
    """
    registry = ToolRegistry()
    bind = partial(_bind, workspace)

    registry.register(READ_FILE, bind(read_file))
    registry.register(LIST_FILES, bind(list_files))
    registry.register(SEARCH_FILES, bind(search_files))
    registry.register(GIT_STATUS, bind(git_status))
    registry.register(GIT_DIFF, bind(git_diff))
    registry.register(GIT_LOG, bind(git_log))

    if not read_only:
        registry.register(WRITE_FILE, bind(write_file))
        registry.register(
            RUN_COMMAND,
            partial(run_command, workspace, confirm_callback=confirm, network=network),
        )
    return registry


def _bind(workspace: Path, function: Any) -> Any:
    return partial(function, workspace)


__all__ = ["BUILTIN_SPECS", "build_registry"]
