"""Effect tests for every rule in :mod:`harness.security.rules`.

Two things are being pinned down here, and they are not the same thing.

The predicates (:func:`deny_reason`, :func:`split_flags`,
:func:`is_block_device`, :func:`is_catastrophic_path`) are exercised directly,
because they are the exported policy API and their edge cases — ``--`` ending
flag parsing, a trailing slash on ``/``, a partition suffix on a block device —
are where a rule quietly stops matching.

The tables are exercised through :func:`classify_command`, because a table row
that never changes a verdict is not a rule, only a string. The parametrisations
iterate over the tables themselves, so a row added later is tested by the same
test rather than silently unasserted.

Note the seam this exposes: ``classifier.py`` carries its own private copies of
these tables and does not import ``rules.py``. ``test_the_two_policy_tables_do_not_drift``
is what keeps the tested module and the enforcing module in step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import Risk
from harness.security import classifier as _classifier
from harness.security import (
    classify_command,
    deny_reason,
    is_block_device,
    is_catastrophic_path,
    rules,
    split_flags,
)

# ---------------------------------------------------------------------------
# is_block_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "/dev/sda", "/dev/sda1", "/dev/hdb", "/dev/vda2", "/dev/xvdf",
        "/dev/nvme0n1", "/dev/nvme0n1p3", "/dev/mmcblk0", "/dev/mmcblk0p1",
        "/dev/disk0", "/dev/nbd0", "/dev/loop3", "/dev/md0", "/dev/sr0",
        "/dev/dm-2",
    ],
)
def test_every_block_device_family_is_recognised(target: str) -> None:
    assert is_block_device(target) is True


@pytest.mark.parametrize(
    "target",
    [
        "/dev/null",       # a harmless sink, not a disk
        "/dev/sda1x",      # the suffix must be numeric
        "dev/sda",         # relative, so not the device node
        "/dev/sd",         # no letter after "sd"
        "/dev/nvme0",      # missing the namespace part
        "",
    ],
)
def test_non_block_devices_are_not_recognised(target: str) -> None:
    assert is_block_device(target) is False


# ---------------------------------------------------------------------------
# is_catastrophic_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "/", "/*", "/.", "/..", "~", "~/*", "$HOME", "${HOME}", "$HOME/*",
        "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
        "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv",
        "/sys", "/usr", "/var",
        "/etc/*",          # the glob form is the same destruction
        "/etc/",           # a trailing slash must not defeat the match
        "  /etc  ",        # nor surrounding whitespace
        "//",              # rstrip("/") collapsing to "" falls back to "/"
    ],
)
def test_catastrophic_paths_are_recognised(target: str) -> None:
    assert is_catastrophic_path(target) is True


@pytest.mark.parametrize(
    "target",
    [
        "",                # nothing to delete
        "   ",
        "build",
        "./build",
        "/etc/hosts",      # a single file below /etc, not /etc itself
        "/usr/local",
        "/nonstandard",
    ],
)
def test_ordinary_paths_are_not_catastrophic(target: str) -> None:
    assert is_catastrophic_path(target) is False


# ---------------------------------------------------------------------------
# split_flags
# ---------------------------------------------------------------------------


def test_short_flags_are_split_into_individual_letters() -> None:
    """``-rf`` must count as both ``r`` and ``f``, or ``rm -rf /`` slips past."""
    flags, operands = split_flags(["-rf", "/"])

    assert flags == {"r", "f"}
    assert operands == ["/"]


def test_long_flags_keep_their_name_and_drop_their_value() -> None:
    flags, operands = split_flags(["--recursive", "--exclude=build", "src"])

    assert flags == {"recursive", "exclude"}
    assert operands == ["src"]


def test_double_dash_ends_flag_parsing() -> None:
    """``rm -- -rf`` deletes a file called ``-rf``; it is not recursive."""
    flags, operands = split_flags(["--", "-rf", "/"])

    assert flags == set()
    assert operands == ["-rf", "/"]


def test_a_bare_dash_is_an_operand_not_a_flag() -> None:
    """``-`` means stdin, and ``len(arg) > 1`` is what keeps it out of the flags."""
    flags, operands = split_flags(["-"])

    assert flags == set()
    assert operands == ["-"]


def test_a_bare_double_dash_after_the_terminator_is_an_operand() -> None:
    flags, operands = split_flags(["--", "--", "x"])

    assert flags == set()
    assert operands == ["--", "x"]


# ---------------------------------------------------------------------------
# deny_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "reason"), sorted(rules.DENY_COMMANDS.items()))
def test_every_deny_command_is_refused_by_name(name: str, reason: str) -> None:
    assert deny_reason([name]) == reason


def test_a_deny_command_is_recognised_through_its_absolute_path() -> None:
    """Only the basename is matched, so ``/sbin/shutdown`` is the same rule."""
    assert deny_reason(["/sbin/shutdown"]) == "powers the machine down"


@pytest.mark.parametrize("name", ["mkfs", "mkfs.ext4", "mkfs.xfs", "mkfs.btrfs"])
def test_any_mkfs_variant_formats_a_filesystem(name: str) -> None:
    assert deny_reason([name, "/dev/sda1"]) == "formats a filesystem"


@pytest.mark.parametrize(("name", "level"), [("init", "0"), ("init", "6"),
                                             ("telinit", "0"), ("telinit", "6")])
def test_init_to_halt_or_reboot_is_refused(name: str, level: str) -> None:
    assert deny_reason([name, level]) == "changes the runlevel to halt or reboot"


@pytest.mark.parametrize("args", [[], ["1"], ["3"], ["5"]])
def test_init_to_any_other_runlevel_is_not_a_denial(args: list[str]) -> None:
    """Only halt (0) and reboot (6) are refused; the rest is left to CONFIRM."""
    assert deny_reason(["init", *args]) is None


@pytest.mark.parametrize(
    "verb", ["poweroff", "reboot", "halt", "kexec", "emergency"]
)
def test_systemctl_power_verbs_are_refused(verb: str) -> None:
    assert deny_reason(["systemctl", verb]) == (
        "powers the machine down or reboots it"
    )


def test_systemctl_finds_the_verb_wherever_it_sits() -> None:
    """The verb is searched for in every argument, not only the first."""
    assert deny_reason(["systemctl", "--no-block", "reboot"]) is not None


def test_systemctl_without_a_power_verb_is_not_a_denial() -> None:
    assert deny_reason(["systemctl", "status", "sshd"]) is None


def test_dd_onto_a_block_device_is_refused() -> None:
    assert deny_reason(["dd", "if=/dev/zero", "of=/dev/sda"]) == (
        "writes a raw image to the block device /dev/sda"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["if=/dev/zero", "of=image.img"],  # a regular file, not a device
        ["if=/dev/sda", "of=backup.img"],  # reading a device is not the rule
        ["if=/dev/zero"],                  # no output operand at all
    ],
)
def test_dd_that_misses_a_block_device_is_not_a_denial(args: list[str]) -> None:
    assert deny_reason(["dd", *args]) is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-rf", "/"], "recursively deletes /"),
        (["-r", "/etc"], "recursively deletes /etc"),
        (["-R", "~"], "recursively deletes ~"),
        (["--recursive", "$HOME"], "recursively deletes $HOME"),
        (["-rf", "build", "/usr"], "recursively deletes /usr"),
        (["--no-preserve-root", "-rf", "/"],
         "removes the filesystem root with --no-preserve-root"),
    ],
)
def test_catastrophic_rm_is_refused(args: list[str], expected: str) -> None:
    assert deny_reason(["rm", *args]) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["-rf", "build"],       # recursive, but inside the workspace
        ["/etc/hosts"],         # catastrophic-looking, but not recursive
        ["-f", "/etc"],         # not recursive either
        ["--no-preserve-root"],  # the flag alone deletes nothing
        ["--", "-rf"],          # "-rf" is a filename here
    ],
)
def test_ordinary_rm_is_left_to_the_confirm_gate(args: list[str]) -> None:
    assert deny_reason(["rm", *args]) is None


@pytest.mark.parametrize("name", ["chmod", "chown"])
@pytest.mark.parametrize("flag", ["-R", "--recursive"])
def test_recursive_chmod_or_chown_on_a_system_path_is_refused(
    name: str, flag: str
) -> None:
    reason = deny_reason([name, flag, "777", "/etc"])

    assert reason == "recursively changes ownership or permissions on /etc"


@pytest.mark.parametrize("name", ["chmod", "chown"])
def test_chmod_or_chown_without_recursion_is_not_a_denial(name: str) -> None:
    assert deny_reason([name, "777", "/etc"]) is None


@pytest.mark.parametrize("name", ["chmod", "chown"])
def test_recursive_chmod_or_chown_inside_the_workspace_is_not_a_denial(
    name: str,
) -> None:
    assert deny_reason([name, "-R", "777", "build"]) is None


@pytest.mark.parametrize("tokens", [["ls"], ["git", "status"], ["frobnicate"]])
def test_commands_without_a_deny_rule_return_none(tokens: list[str]) -> None:
    """``deny_reason`` decides refusal only; everything else is the caller's job."""
    assert deny_reason(tokens) is None


# ---------------------------------------------------------------------------
# The tables, through the classifier that enforces them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(rules.DENY_COMMANDS))
def test_each_deny_command_reaches_a_deny_verdict(name: str, tmp_path: Path) -> None:
    risk, _ = classify_command(name, workspace=tmp_path)

    assert risk is Risk.DENY


@pytest.mark.parametrize("name", sorted(rules.CONFIRM_COMMANDS))
def test_each_confirm_command_reaches_a_confirm_verdict(
    name: str, tmp_path: Path
) -> None:
    risk, _ = classify_command(name, workspace=tmp_path)

    assert risk is Risk.CONFIRM


@pytest.mark.parametrize("prefix", [p for p, _ in rules.CONFIRM_PREFIXES])
def test_each_confirm_prefix_reaches_a_confirm_verdict(
    prefix: tuple[str, ...], tmp_path: Path
) -> None:
    risk, _ = classify_command(" ".join(prefix), workspace=tmp_path)

    assert risk is Risk.CONFIRM


@pytest.mark.parametrize("prefix", rules.ALLOW_PREFIXES)
def test_each_allow_prefix_reaches_an_allow_verdict(
    prefix: tuple[str, ...], tmp_path: Path
) -> None:
    risk, _ = classify_command(" ".join(prefix), workspace=tmp_path)

    assert risk is Risk.ALLOW


@pytest.mark.parametrize("wrapper", sorted(rules.WRAPPERS))
def test_no_wrapper_hides_the_command_it_wraps(wrapper: str, tmp_path: Path) -> None:
    """Stripping wrappers is what keeps ``sudo shutdown`` a denial."""
    risk, reason = classify_command(f"{wrapper} shutdown", workspace=tmp_path)

    assert risk is Risk.DENY
    assert reason == "powers the machine down"


@pytest.mark.parametrize("wrapper", sorted(rules.PRIVILEGE_WRAPPERS))
def test_a_privilege_wrapper_lifts_a_read_only_command_to_confirm(
    wrapper: str, tmp_path: Path
) -> None:
    risk, reason = classify_command(f"{wrapper} ls", workspace=tmp_path)

    assert risk is Risk.CONFIRM
    assert reason == "runs with elevated privileges"


@pytest.mark.parametrize(
    ("prefix", "flag"),
    [
        (prefix, flag)
        for prefix, (flags, _) in rules.ALLOW_ESCALATIONS.items()
        for flag in sorted(flags)
    ],
)
def test_each_escalation_flag_downgrades_its_allowed_command(
    prefix: tuple[str, ...], flag: str, tmp_path: Path
) -> None:
    command = f"{' '.join(prefix)} {flag}"
    expected = rules.ALLOW_ESCALATIONS[prefix][1]

    risk, reason = classify_command(command, workspace=tmp_path)

    assert risk is Risk.CONFIRM
    assert reason == expected


def test_env_assignments_are_stripped_before_the_command_is_matched() -> None:
    """``ENV_ASSIGN_RE``: without it, ``FOO=1 shutdown`` reads as an unknown word."""
    assert rules.ENV_ASSIGN_RE.match("FOO=1") is not None
    assert rules.ENV_ASSIGN_RE.match("_x9=") is not None
    assert rules.ENV_ASSIGN_RE.match("9FOO=1") is None
    assert rules.ENV_ASSIGN_RE.match("--opt=1") is None


def test_numeric_arguments_are_treated_as_wrapper_operands() -> None:
    """``NUMERIC_ARG_RE``: ``timeout 5 shutdown`` must not stop at the ``5``."""
    for value in ("5", "0.5", "30s", "2m", "1h", "7d"):
        assert rules.NUMERIC_ARG_RE.match(value) is not None
    assert rules.NUMERIC_ARG_RE.match("5x") is None
    assert rules.NUMERIC_ARG_RE.match("abc") is None


@pytest.mark.parametrize(
    ("token", "target"),
    [(">", ""), (">>", ""), ("2>", ""), ("2>>", ""), ("&>", ""),
     (">/dev/sda", "/dev/sda"), ("2>>log", "log")],
)
def test_the_redirect_pattern_matches_both_token_shapes(
    token: str, target: str
) -> None:
    match = rules.REDIRECT_RE.match(token)

    assert match is not None
    assert match.group(3) == target


@pytest.mark.parametrize("token", ["a > b", "->", "file>name here", "x>y"])
def test_the_redirect_pattern_ignores_text_that_only_contains_an_angle_bracket(
    token: str,
) -> None:
    assert rules.REDIRECT_RE.match(token) is None


def test_the_fork_bomb_pattern_matches_the_classic_form() -> None:
    assert rules.FORK_BOMB_RE.search(":(){ :|:& };:") is not None
    assert rules.FORK_BOMB_RE.search("bomb() { bomb|bomb& }; bomb") is not None
    assert rules.FORK_BOMB_RE.search("greet() { echo hi; }; greet") is None


def test_the_drop_database_pattern_ignores_case_and_spacing() -> None:
    assert rules.DROP_DATABASE_RE.search("DROP DATABASE prod") is not None
    assert rules.DROP_DATABASE_RE.search("drop   database prod") is not None
    assert rules.DROP_DATABASE_RE.search("dropdatabase prod") is None


def test_harmless_sinks_are_the_only_redirect_targets_that_stay_allowed() -> None:
    assert frozenset(
        {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}
    ) == rules.HARMLESS_SINKS


# ---------------------------------------------------------------------------
# Drift between the policy tables and the classifier's private copies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_rules", "in_classifier"),
    [
        ("DENY_COMMANDS", "_DENY_COMMANDS"),
        ("CONFIRM_COMMANDS", "_CONFIRM_COMMANDS"),
        ("WRAPPERS", "_WRAPPERS"),
        ("PRIVILEGE_WRAPPERS", "_PRIVILEGE"),
        ("ALLOW_PREFIXES", "_ALLOW_PREFIXES"),
        ("HARMLESS_SINKS", "_HARMLESS_SINKS"),
        ("ALLOW_ESCALATIONS", "_ALLOW_ESCALATIONS"),
    ],
)
def test_the_two_policy_tables_do_not_drift(
    in_rules: str, in_classifier: str
) -> None:
    """``classifier.py`` duplicates every table instead of importing ``rules``.

    Testing ``rules.py`` therefore proves nothing about the enforcing code
    unless the two agree on which commands are listed. Membership is what a
    verdict depends on, so membership is what is pinned here; the reason
    strings have already drifted apart and are reported separately on #32.
    """
    mine = getattr(rules, in_rules)
    theirs = getattr(_classifier, in_classifier)

    assert set(mine) == set(theirs)


def test_the_confirm_prefix_tables_list_the_same_commands() -> None:
    assert tuple(prefix for prefix, _ in rules.CONFIRM_PREFIXES) == tuple(
        prefix for prefix, _ in _classifier._CONFIRM_PREFIXES
    )


@pytest.mark.parametrize(
    ("in_rules", "in_classifier"),
    [
        ("_BLOCK_DEVICE_RE", "_BLOCK_DEVICE_RE"),
        ("_SYSTEM_DIR_RE", "_SYSTEM_DIR_RE"),
        ("FORK_BOMB_RE", "_FORK_BOMB_RE"),
        ("DROP_DATABASE_RE", "_DROP_DATABASE_RE"),
        ("REDIRECT_RE", "_REDIRECT_RE"),
        ("ENV_ASSIGN_RE", "_ENV_ASSIGN_RE"),
        ("NUMERIC_ARG_RE", "_NUMERIC_ARG_RE"),
    ],
)
def test_the_duplicated_patterns_do_not_drift(
    in_rules: str, in_classifier: str
) -> None:
    assert getattr(rules, in_rules).pattern == getattr(
        _classifier, in_classifier
    ).pattern


def test_the_root_target_sets_do_not_drift() -> None:
    assert rules._ROOT_TARGETS == _classifier._ROOT_TARGETS


@pytest.mark.parametrize(
    "tokens",
    [
        ["shutdown"], ["mkfs.ext4", "/dev/sda"], ["init", "0"],
        ["systemctl", "reboot"], ["dd", "of=/dev/sda"], ["rm", "-rf", "/"],
        ["chmod", "-R", "777", "/etc"], ["ls"],
    ],
)
def test_both_deny_implementations_agree_on_whether_a_command_is_refused(
    tokens: list[str],
) -> None:
    """Reason wording differs between the two copies; the verdict must not."""
    assert (deny_reason(tokens) is None) == (
        _classifier._deny_rule(tokens) is None
    )
