"""The command tables and the deny rules they feed.

Kept apart from :mod:`harness.security.classifier` so the policy — which
commands are read-only, which need a human, which are refused outright — can be
read and audited without wading through the shell parsing that finds them.
"""

from __future__ import annotations

import re

_BLOCK_DEVICE_RE = re.compile(
    r"^/dev/(sd[a-z]+|hd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|mmcblk\d+"
    r"|disk\d+|nbd\d+|loop\d+|md\d+|sr\d+|dm-\d+)(p?\d+)?$"
)

# A function that pipes itself into itself in the background and then calls
# itself. Matched against the raw command because splitting on ";" and "|" is
# precisely what a fork bomb's syntax abuses.
FORK_BOMB_RE = re.compile(
    r"(?P<name>[\w:.\-]+)\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*(?P=name)"
)
DROP_DATABASE_RE = re.compile(r"\bdrop\s+database\b", re.IGNORECASE)

ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
NUMERIC_ARG_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")

# Redirect token, either standalone (">", "2>>") or with the target attached
# (">/dev/sda"). Quoted text containing ">" carries spaces and never matches.
REDIRECT_RE = re.compile(r"^(\d*|&)(>{1,2})(\S*)$")
HARMLESS_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})

# Prefixes that wrap another command. Stripping them is what stops
# "sudo rm -rf /" or "env FOO=1 rm -rf /" from reading as an unknown command.
WRAPPERS = frozenset({
    "env", "time", "nohup", "sudo", "doas", "command", "exec", "nice",
    "ionice", "timeout", "stdbuf", "setsid", "xargs", "builtin",
})
PRIVILEGE_WRAPPERS = frozenset({"sudo", "doas"})

ALLOW_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"), ("git", "diff"), ("git", "log"), ("git", "show"),
    ("git", "branch"),
    ("ls",), ("cat",), ("head",), ("tail",), ("rg",), ("grep",), ("find",),
    ("wc",), ("tree",), ("pwd",), ("which",), ("echo",), ("pytest",),
    ("npm", "test"), ("uv", "run", "pytest"), ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"), ("make", "test"), ("cargo", "test"),
)

# Flags that turn one of the read-only commands above into a writing one.
ALLOW_ESCALATIONS: dict[tuple[str, ...], tuple[frozenset[str], str]] = {
    ("find",): (
        frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"}),
        "find executes or deletes rather than only listing",
    ),
    ("git", "branch"): (
        frozenset({"-d", "-D", "-m", "-M", "-f", "--delete", "--move", "--force"}),
        "git branch deletes or renames a branch",
    ),
}

_INSTALL = "package installation fetches and executes third-party code"

CONFIRM_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("git", "push"), "git push publishes to a remote"),
    (("git", "commit"), "git commit writes repository history"),
    (("git", "reset"), "git reset can discard committed work"),
    (("git", "clean"), "git clean deletes untracked files"),
    (("git", "rebase"), "git rebase rewrites history"),
    (("pip", "install"), _INSTALL),
    (("pip3", "install"), _INSTALL),
    (("uv", "pip", "install"), _INSTALL),
    (("uv", "add"), _INSTALL),
    (("npm", "install"), _INSTALL),
    (("npm", "i"), _INSTALL),
    (("yarn", "add"), _INSTALL),
    (("pnpm", "add"), _INSTALL),
    (("cargo", "add"), _INSTALL),
)

CONFIRM_COMMANDS: dict[str, str] = {
    "curl": "network access",
    "wget": "network access",
    "nc": "network access",
    "ncat": "network access",
    "telnet": "network access",
    "ssh": "network access",
    "scp": "network access",
    "sftp": "network access",
    "rsync": "network access",
    "chmod": "changes file permissions",
    "chown": "changes file ownership",
    "systemctl": "manages system services",
    "docker": "controls the container runtime",
    "podman": "controls the container runtime",
    "apt": _INSTALL,
    "apt-get": _INSTALL,
    "pacman": _INSTALL,
    "paru": _INSTALL,
    "yay": _INSTALL,
    "rm": "deletes files",
    "mv": "moves files, possibly outside the workspace",
    "cp": "copies files, possibly outside the workspace",
    "kill": "signals other processes",
    "pkill": "signals other processes",
    "mount": "changes the filesystem tree",
    "umount": "changes the filesystem tree",
}

DENY_COMMANDS: dict[str, str] = {
    "shutdown": "powers the machine down",
    "reboot": "reboots the machine",
    "halt": "halts the machine",
    "poweroff": "powers the machine down",
    "wipefs": "erases filesystem signatures",
    "mkswap": "formats a device as swap",
    "mke2fs": "formats a filesystem",
}

# Recursively deleting any of these destroys the system or the user's data.
_SYSTEM_DIR_RE = re.compile(
    r"^/(bin|boot|dev|etc|home|lib|lib32|lib64|media|mnt|opt|proc|root|run"
    r"|sbin|srv|sys|usr|var)(/\*)?$"
)
_ROOT_TARGETS = frozenset(
    {"/", "/*", "/.", "/..", "~", "~/*", "$HOME", "${HOME}", "$HOME/*"}
)


def is_block_device(target: str) -> bool:
    return bool(_BLOCK_DEVICE_RE.match(target))


def is_catastrophic_path(target: str) -> bool:
    """True for paths whose recursive destruction is not recoverable."""
    t = target.strip()
    if not t:
        return False
    t = t.rstrip("/") or "/"
    return t in _ROOT_TARGETS or bool(_SYSTEM_DIR_RE.match(t))


def split_flags(args: list[str]) -> tuple[set[str], list[str]]:
    """Return flag names (long names and individual short letters) and operands.

    ``--`` ends flag parsing, so ``rm -- -rf`` treats ``-rf`` as a filename.
    """
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


def deny_reason(tokens: list[str]) -> str | None:
    """Return why these tokens are refused outright, or None.

    ``tokens`` must start at a command word; the caller is responsible for
    having stripped wrappers such as ``sudo``.
    """
    name = tokens[0].rsplit("/", 1)[-1]
    args = tokens[1:]

    if name in DENY_COMMANDS:
        return DENY_COMMANDS[name]
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
            if arg.startswith("of=") and is_block_device(arg[3:]):
                return f"writes a raw image to the block device {arg[3:]}"
    if name == "rm":
        return _rm_deny_reason(args)
    if name in {"chmod", "chown"}:
        flags, operands = split_flags(args)
        if flags & {"R", "recursive"}:
            hits = [t for t in operands if is_catastrophic_path(t)]
            if hits:
                return f"recursively changes ownership or permissions on {hits[0]}"
    return None


def _rm_deny_reason(args: list[str]) -> str | None:
    flags, operands = split_flags(args)
    if "no-preserve-root" in flags and operands:
        return "removes the filesystem root with --no-preserve-root"
    if not flags & {"r", "R", "recursive"}:
        return None
    hits = [t for t in operands if is_catastrophic_path(t)]
    return f"recursively deletes {hits[0]}" if hits else None
