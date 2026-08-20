"""Deterministic allow/confirm/deny classification of shell commands.

The model's willingness to run a command is not an input to this decision.
A command string is split into every segment a shell would actually execute —
including command substitutions — and each segment is classified on its own.
The most severe verdict wins, so a denied command cannot be smuggled past the
gate behind ``;``, ``&&``, a pipe, a backtick or ``$(...)``.

Anything unrecognised resolves to CONFIRM. An unknown command is not a safe
command.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from harness.core import Risk
from harness.security.shellsplit import TOO_DEEP, split_segments

_SEVERITY: dict[Risk, int] = {Risk.ALLOW: 0, Risk.CONFIRM: 1, Risk.DENY: 2}

# Substitutions are lifted out into their own segments; the outer segment keeps
# a placeholder so token positions (and therefore the head command) survive.

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_NUMERIC_ARG_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")
_BLOCK_DEVICE_RE = re.compile(
    r"^/dev/(sd[a-z]+|hd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|mmcblk\d+"
    r"|disk\d+|nbd\d+|loop\d+|md\d+|sr\d+|dm-\d+)(p?\d+)?$"
)
# Redirect token, either standalone (">", "2>>") or with the target attached
# (">/dev/sda"). Quoted text containing ">" carries spaces and so never matches.
_REDIRECT_RE = re.compile(r"^(\d*|&)(>{1,2})(\S*)$")
_HARMLESS_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})

# A function that pipes itself into itself in the background and then calls
# itself. Checked against the raw string because splitting on ";" and "|" is
# exactly what a fork bomb's syntax abuses.
_FORK_BOMB_RE = re.compile(
    r"(?P<name>[\w:.\-]+)\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*(?P=name)"
)
_DROP_DATABASE_RE = re.compile(r"\bdrop\s+database\b", re.IGNORECASE)

# Prefixes that wrap another command. Stripping them is what stops
# "sudo rm -rf /" or "env FOO=1 rm -rf /" from reading as an unknown command.
_WRAPPERS = frozenset({
    "env", "time", "nohup", "sudo", "doas", "command", "exec", "nice",
    "ionice", "timeout", "stdbuf", "setsid", "xargs", "builtin",
})
_PRIVILEGE = frozenset({"sudo", "doas"})

_ALLOW_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"), ("git", "diff"), ("git", "log"), ("git", "show"),
    ("git", "branch"),
    ("ls",), ("cat",), ("head",), ("tail",), ("rg",), ("grep",), ("find",),
    ("wc",), ("tree",), ("pwd",), ("which",), ("echo",), ("pytest",),
    ("npm", "test"), ("uv", "run", "pytest"), ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"), ("make", "test"), ("cargo", "test"),
)

_CONFIRM_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("git", "push"), "git push publishes to a remote"),
    (("git", "commit"), "git commit rewrites repository history"),
    (("git", "reset"), "git reset can discard committed work"),
    (("git", "clean"), "git clean deletes untracked files"),
    (("git", "rebase"), "git rebase rewrites history"),
    (("pip", "install"), "package installation fetches and executes code"),
    (("pip3", "install"), "package installation fetches and executes code"),
    (("uv", "pip", "install"), "package installation fetches and executes code"),
    (("uv", "add"), "package installation fetches and executes code"),
    (("npm", "install"), "package installation fetches and executes code"),
    (("npm", "i"), "package installation fetches and executes code"),
    (("yarn", "add"), "package installation fetches and executes code"),
    (("pnpm", "add"), "package installation fetches and executes code"),
    (("cargo", "add"), "package installation fetches and executes code"),
)

_CONFIRM_COMMANDS: dict[str, str] = {
    "curl": "network access", "wget": "network access", "nc": "network access",
    "ncat": "network access", "telnet": "network access", "ssh": "network access",
    "scp": "network access", "sftp": "network access", "rsync": "network access",
    "chmod": "changes file permissions", "chown": "changes file ownership",
    "systemctl": "manages system services", "docker": "controls the container runtime",
    "podman": "controls the container runtime", "apt": "package installation",
    "apt-get": "package installation", "pacman": "package installation",
    "paru": "package installation", "yay": "package installation",
    "mv": "moves files, possibly outside the workspace",
    "cp": "copies files, possibly outside the workspace",
    "rm": "deletes files",
    "kill": "signals other processes", "pkill": "signals other processes",
    "mount": "changes the filesystem tree", "umount": "changes the filesystem tree",
}

_DENY_COMMANDS: dict[str, str] = {
    "shutdown": "powers the machine down", "reboot": "reboots the machine",
    "halt": "halts the machine", "poweroff": "powers the machine down",
    "wipefs": "erases filesystem signatures",
    "mkswap": "formats a device",
    "mke2fs": "formats a filesystem",
}

# Recursively removing any of these destroys the system or the user's data.
_SYSTEM_DIR_RE = re.compile(
    r"^/(bin|boot|dev|etc|home|lib|lib32|lib64|media|mnt|opt|proc|root|run"
    r"|sbin|srv|sys|usr|var)(/\*)?$"
)
_ROOT_TARGETS = frozenset({"/", "/*", "/.", "/..", "~", "~/*", "$HOME", "${HOME}", "$HOME/*"})


def classify_command(
    command: str, *, workspace: str | Path | None = None
) -> tuple[Risk, str]:
    """Classify a shell command string.

    Returns the risk and a human-readable reason. Every executable segment is
    classified; the returned pair belongs to the most severe one.
    """
    if not command or not command.strip():
        return Risk.CONFIRM, "empty command"

    if _FORK_BOMB_RE.search(command):
        return Risk.DENY, "fork bomb"
    if _DROP_DATABASE_RE.search(command):
        return Risk.DENY, "drops a database"

    worst: tuple[Risk, str] | None = None
    for segment in split_segments(command):
        if not segment.strip():
            continue
        if segment == TOO_DEEP:
            # Part of the command was never read. Reporting the part that was
            # would be a verdict on something other than what runs.
            return Risk.CONFIRM, "substitution nested too deeply to classify"
        verdict = _classify_segment(segment, workspace=workspace)
        # Ties keep the first verdict, so a command reports the rule that
        # actually matched it rather than a placeholder — the reason is what
        # ends up in the journal.
        if worst is None or _SEVERITY[verdict[0]] > _SEVERITY[worst[0]]:
            worst = verdict
    if worst is None:
        return Risk.CONFIRM, "command contains no executable segment"
    return worst




# --------------------------------------------------------------------------
# Segment classification
# --------------------------------------------------------------------------


def _basename(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def _is_prefix_like(token: str) -> bool:
    return bool(
        _ENV_ASSIGN_RE.match(token)
        or _basename(token) in _WRAPPERS
        or (token.startswith("-") and len(token) > 1)
        or _NUMERIC_ARG_RE.match(token)
    )


def _command_starts(tokens: list[str]) -> list[int]:
    """Indices where a real command could begin after wrapper prefixes.

    An option may or may not consume the next token (``sudo -u root cmd`` vs
    ``sudo -n cmd``), so both successors are treated as candidates rather than
    guessing per option and letting a wrong guess hide the command.

    ``tokens`` is never empty: the caller returns before this point when
    tokenisation yields nothing, so every index on the frontier is valid.
    """
    candidates = {0}
    frontier = [0]
    while frontier:
        i = frontier.pop()
        if not _is_prefix_like(tokens[i]):
            continue
        nxt = [i + 1]
        if tokens[i].startswith("-") and len(tokens[i]) > 1:
            nxt.append(i + 2)
        for j in nxt:
            if j < len(tokens) and j not in candidates:
                candidates.add(j)
                frontier.append(j)
    return sorted(candidates)


def _head(tokens: list[str]) -> tuple[list[str], bool]:
    """Strip wrapper prefixes. Returns the remaining tokens and a privilege flag."""
    i = 0
    privileged = False
    while i < len(tokens):
        token = tokens[i]
        if _ENV_ASSIGN_RE.match(token):
            i += 1
            continue
        name = _basename(token)
        if name in _WRAPPERS:
            privileged = privileged or name in _PRIVILEGE
            i += 1
            while i < len(tokens) and (
                _ENV_ASSIGN_RE.match(tokens[i])
                or (tokens[i].startswith("-") and len(tokens[i]) > 1)
                or _NUMERIC_ARG_RE.match(tokens[i])
            ):
                i += 1
            continue
        break
    return tokens[i:], privileged


def _classify_segment(
    segment: str, *, workspace: str | Path | None
) -> tuple[Risk, str]:
    try:
        tokens = shlex.split(segment)
    except ValueError as exc:
        return Risk.CONFIRM, f"command could not be parsed ({exc})"
    # Unreachable from classify_command, which already skips blank segments —
    # shlex treats a subset of the whitespace that str.strip() does, so a
    # segment that survives that skip always tokenises to something. Kept, and
    # kept here rather than repeated downstream, because it is the single point
    # that guarantees tokens[0] exists for every helper below.
    if not tokens:
        return Risk.ALLOW, "empty segment"

    for index in _command_starts(tokens):
        denial = _deny_rule(tokens[index:])
        if denial is not None:
            return Risk.DENY, denial

    redirect = _redirect_risk(tokens)
    if redirect is not None and redirect[0] is Risk.DENY:
        return redirect

    rest, privileged = _head(tokens)
    verdict = _confirm_or_allow(rest)
    if verdict[0] is Risk.ALLOW:
        workspace_risk = _workspace_operand_risk(tokens, workspace=workspace)
        if workspace_risk is not None:
            verdict = workspace_risk
    if privileged and _SEVERITY[verdict[0]] < _SEVERITY[Risk.CONFIRM]:
        verdict = (Risk.CONFIRM, "runs with elevated privileges")
    if redirect is not None and _SEVERITY[redirect[0]] > _SEVERITY[verdict[0]]:
        verdict = redirect
    return verdict


def _workspace_operand_risk(
    tokens: list[str], *, workspace: str | Path | None
) -> tuple[Risk, str] | None:
    """Reject operands that a nominally read-only command resolves outside."""
    root = Path(workspace).resolve() if workspace is not None else None
    for token in tokens[1:]:
        if token == "--":
            continue
        candidate = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
        candidate = re.sub(r"^\d*<+", "", candidate)
        if candidate.startswith("-") or not candidate:
            continue
        if "$" in candidate or candidate.startswith("~"):
            return Risk.DENY, f"path expansion is outside the workspace boundary: {candidate}"
        path = Path(candidate)
        if root is None:
            if path.is_absolute() or ".." in path.parts:
                return Risk.DENY, f"path is outside the workspace boundary: {candidate}"
            continue
        try:
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return Risk.DENY, f"path is outside the workspace boundary: {candidate}"
    return None


def _confirm_or_allow(tokens: list[str]) -> tuple[Risk, str]:
    if not tokens:
        return Risk.CONFIRM, "no command after prefixes"
    name = _basename(tokens[0])
    normalised = [name, *tokens[1:]]

    for prefix, reason in _CONFIRM_PREFIXES:
        if tuple(normalised[:len(prefix)]) == prefix:
            return Risk.CONFIRM, reason
    if name in _CONFIRM_COMMANDS:
        return Risk.CONFIRM, _CONFIRM_COMMANDS[name]

    for prefix in _ALLOW_PREFIXES:
        if tuple(normalised[:len(prefix)]) == prefix:
            escalation = _allow_escalation(prefix, normalised)
            if escalation is not None:
                return Risk.CONFIRM, escalation
            return Risk.ALLOW, f"read-only development command: {' '.join(prefix)}"
    return Risk.CONFIRM, f"unrecognised command: {name}"


# Flags that turn a read-only command into a writing one.
_ALLOW_ESCALATIONS: dict[tuple[str, ...], tuple[frozenset[str], str]] = {
    ("find",): (frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"}),
                "find executes or deletes rather than only listing"),
    ("git", "branch"): (frozenset({"-d", "-D", "-m", "-M", "--delete", "--move",
                                   "--force", "-f"}),
                        "git branch deletes or renames a branch"),
}


def _allow_escalation(prefix: tuple[str, ...], tokens: list[str]) -> str | None:
    entry = _ALLOW_ESCALATIONS.get(prefix)
    if entry is None:
        return None
    flags, reason = entry
    return reason if any(t in flags for t in tokens[len(prefix):]) else None


def _redirect_risk(tokens: list[str]) -> tuple[Risk, str] | None:
    """Redirects write files, so they lift a read-only command to CONFIRM."""
    worst: tuple[Risk, str] | None = None
    for index, token in enumerate(tokens):
        match = _REDIRECT_RE.match(token)
        if match is None:
            continue
        target = match.group(3)
        if not target:
            target = tokens[index + 1] if index + 1 < len(tokens) else ""
        if target.startswith("&") or not target:
            continue  # file descriptor duplication, e.g. 2>&1
        if _BLOCK_DEVICE_RE.match(target):
            return Risk.DENY, f"writes directly to the block device {target}"
        if target in _HARMLESS_SINKS:
            continue
        worst = (Risk.CONFIRM, f"redirects output into {target}")
    return worst


# --------------------------------------------------------------------------
# Deny rules
# --------------------------------------------------------------------------


def _is_catastrophic_path(target: str) -> bool:
    t = target.strip()
    if not t:
        return False
    t = t.rstrip("/") or "/"
    return t in _ROOT_TARGETS or bool(_SYSTEM_DIR_RE.match(t))


def _split_flags(args: list[str]) -> tuple[set[str], list[str]]:
    """Return long/short flag names and the operands, honouring ``--``."""
    flags: set[str] = set()
    operands: list[str] = []
    end_of_flags = False
    for arg in args:
        if not end_of_flags and arg == "--":
            end_of_flags = True
            continue
        if not end_of_flags and arg.startswith("--") and len(arg) > 2:
            flags.add(arg[2:].split("=", 1)[0])
            continue
        if not end_of_flags and arg.startswith("-") and len(arg) > 1:
            flags.update(arg[1:])
            continue
        operands.append(arg)
    return flags, operands


def _deny_rule(tokens: list[str]) -> str | None:
    """Return a denial reason if these tokens start a catastrophic command."""
    name = _basename(tokens[0])
    args = tokens[1:]

    if name in _DENY_COMMANDS:
        return _DENY_COMMANDS[name]
    if name == "mkfs" or name.startswith("mkfs."):
        return "formats a filesystem"
    if name in {"init", "telinit"} and args and args[0] in {"0", "6"}:
        return "changes the runlevel to halt or reboot"
    if name == "systemctl" and any(
        a in {"poweroff", "reboot", "halt", "kexec", "emergency"} for a in args
    ):
        return "powers the machine down or reboots it"
    if name == "dd":
        for arg in args:
            if arg.startswith("of=") and _BLOCK_DEVICE_RE.match(arg[3:]):
                return f"writes a raw image to the block device {arg[3:]}"
    if name == "rm":
        flags, operands = _split_flags(args)
        recursive = bool(flags & {"r", "R", "recursive"})
        if "no-preserve-root" in flags and operands:
            return "removes the filesystem root with --no-preserve-root"
        hits = [t for t in operands if _is_catastrophic_path(t)]
        if recursive and hits:
            return f"recursively deletes {hits[0]}"
    if name in {"chmod", "chown"}:
        flags, operands = _split_flags(args)
        if flags & {"R", "recursive"}:
            hits = [t for t in operands if _is_catastrophic_path(t)]
            if hits:
                return f"recursively changes permissions on {hits[0]}"
    return None
