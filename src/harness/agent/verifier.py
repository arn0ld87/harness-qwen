"""Evidence-gated verification of an ``AnswerAction`` (ARCHITECTURE.md,
"Verification").

Claims of the form *implemented*, *fixed*, *tests pass* are rejected unless
matching evidence exists. This module operationalises the evidence table:
each claim keyword maps to one check, and every check is independent of what
the model asserts — it reads the workspace directly (file existence, mtime,
``git diff``) or audits the run's own executed steps (a captured tool result
already on record), never anything the model merely says happened.

Detection is a narrow phrase match, deliberately. A false negative here just
means an unverified claim slips through undetected — no worse than the
verifier not existing for that sentence. A false positive would flag a
legitimate no-op answer ("nothing needed to change") as an unbacked claim,
which is the wrong direction to err in, so the phrase list stays specific
rather than a loose bag of words.
"""

from __future__ import annotations

import enum
import hashlib
import os
import re
import shlex
import stat
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from harness.core import (
    AnswerAction,
    CommandEvidence,
    ExecutedToolStep,
    FileEvidence,
    WorkspaceBaseline,
)
from harness.security import resolve_in_workspace
from harness.security.shellsplit import split_segments


class Claim(enum.StrEnum):
    FILE_WRITTEN = "file_written"
    PATCH_APPLIED = "patch_applied"
    TESTS_PASS = "tests_pass"
    BUILDS = "builds"
    LINT_CLEAN = "lint_clean"
    TYPECHECK_CLEAN = "typecheck_clean"


CLAIM_PHRASES: dict[Claim, tuple[str, ...]] = {
    Claim.FILE_WRITTEN: (
        "wrote the file", "written the file", "created the file",
        "created file", "added the file",
    ),
    Claim.PATCH_APPLIED: (
        "patch applied", "applied the patch", "fixed the", "patched the",
    ),
    Claim.TESTS_PASS: (
        "tests pass", "test passes", "tests passing", "all tests pass",
        "tests are passing",
    ),
    Claim.BUILDS: (
        "build succeeds", "build passes", "builds cleanly", "build is clean", "it builds",
    ),
    Claim.LINT_CLEAN: (
        "lint is clean", "lint passes", "no lint errors",
    ),
    Claim.TYPECHECK_CLEAN: (
        "typecheck is clean", "typecheck passes", "type check passes",
    ),
}
"""Claim -> the specific phrasing that triggers it. Table-driven so a missing
check reads as a missing dict entry, not a buried conditional."""


ExecutedStep = ExecutedToolStep


class VerificationOutcome(BaseModel):
    verified: bool
    notes: list[str] = Field(default_factory=list)
    claims_checked: list[str] = Field(default_factory=list)


def detect_claims(content: str) -> list[Claim]:
    """Which evidence-requiring claims ``content`` makes, in table order."""
    lowered = content.lower()
    return [
        claim for claim, phrases in CLAIM_PHRASES.items()
        if any(phrase in lowered for phrase in phrases)
    ]


class Verifier:
    """Checks an ``AnswerAction`` against the evidence it cites.

    An answer that makes no detected claim passes trivially — there is
    nothing to check, and refusing an answer for a claim it never made would
    be false diligence, not rigour. An answer that does make a claim but
    cites no evidence, or whose evidence does not hold up, comes back
    ``verified=False`` with a note naming exactly what was missing — never
    silently accepted.
    """

    def verify(
        self,
        answer: AnswerAction,
        *,
        history: Sequence[ExecutedStep],
        run_id: str,
        workspace: Path,
        baseline: WorkspaceBaseline | None = None,
    ) -> VerificationOutcome:
        claims = detect_claims(answer.content)
        if not claims:
            return VerificationOutcome(verified=True, notes=[], claims_checked=[])

        if not answer.evidence:
            return VerificationOutcome(
                verified=False,
                notes=[f"claimed {c.value} but cited no evidence" for c in claims],
                claims_checked=[c.value for c in claims],
            )

        notes: list[str] = []
        all_ok = True
        for claim in claims:
            ok, note = self._check(
                claim,
                answer.evidence,
                history=history,
                run_id=run_id,
                workspace=workspace,
                baseline=baseline,
            )
            all_ok = all_ok and ok
            notes.append(f"{claim.value}: {note}")
        checked = [c.value for c in claims]
        return VerificationOutcome(verified=all_ok, notes=notes, claims_checked=checked)

    def _check(
        self,
        claim: Claim,
        evidence: Sequence[CommandEvidence | FileEvidence],
        *,
        history: Sequence[ExecutedStep],
        run_id: str,
        workspace: Path,
        baseline: WorkspaceBaseline | None,
    ) -> tuple[bool, str]:
        if claim is Claim.FILE_WRITTEN:
            return _check_file_written(evidence, workspace=workspace, baseline=baseline)
        if claim is Claim.PATCH_APPLIED:
            return _check_patch_applied(
                evidence, workspace=workspace, baseline=baseline
            )
        return _check_command_evidence(
            claim, evidence, history=history, run_id=run_id
        )


def _file_changed_since_baseline(
    item: FileEvidence,
    *,
    workspace: Path,
    baseline: WorkspaceBaseline,
) -> tuple[bool, str] | None:
    """Compare one evidence path against its run-start fingerprint.

    Returns ``None`` when the path is unresolvable or not fingerprintable
    (e.g. deleted, or a directory) so the caller can move to the next evidence
    item. Otherwise returns ``(changed, note)`` — ``changed`` is True when the
    content fingerprint differs from the baseline, including the case of a file
    that did not exist at run start (``before is None``).
    """
    try:
        resolved = resolve_in_workspace(item.path, workspace)
        relative = resolved.relative_to(workspace.resolve()).as_posix()
    except Exception:
        return None
    after = _fingerprint(resolved)
    if after is None:
        return None
    before = baseline.files.get(relative)
    if before != after:
        return True, f"{item.path}: content differs from run-start baseline"
    return False, f"{item.path}: unchanged since run-start baseline"


def _check_file_written(
    evidence: Sequence[CommandEvidence | FileEvidence],
    *,
    workspace: Path,
    baseline: WorkspaceBaseline | None,
) -> tuple[bool, str]:
    if baseline is None:
        return False, "run has no workspace baseline"
    for item in evidence:
        if not isinstance(item, FileEvidence) or item.kind != "file":
            continue
        result = _file_changed_since_baseline(item, workspace=workspace, baseline=baseline)
        if result is not None:
            return result
    return False, "no evidence path changed since the run-start baseline"


def _check_patch_applied(
    evidence: Sequence[CommandEvidence | FileEvidence],
    *,
    workspace: Path,
    baseline: WorkspaceBaseline | None,
) -> tuple[bool, str]:
    if baseline is None:
        return False, "run has no workspace baseline"
    for item in evidence:
        if not isinstance(item, FileEvidence) or item.kind != "patch":
            continue
        result = _file_changed_since_baseline(item, workspace=workspace, baseline=baseline)
        if result is not None:
            return result
    return False, "no evidence path changed since the run-start baseline"


def _git_output(workspace: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else b""


def _fingerprint(path: Path) -> str | None:
    try:
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            payload = f"symlink\0{os.readlink(path)}".encode()
        elif path.is_file():
            payload = b"file\0" + path.read_bytes()
        else:
            return None
    except OSError:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    return f"{mode:o}:{digest}"


def capture_workspace_baseline(workspace: Path) -> WorkspaceBaseline:
    """Capture content fingerprints of every relevant file before tool execution.

    Git-independent: the workspace tree is walked directly with ``os.walk``, so
    non-Git workspaces and ``.gitignore``'d files are included — an unchanged
    existing file must never read as a run-time change. ``.git/`` internals are
    skipped (constant churn, never an evidence target). Git-derived summary
    fields (``head_sha``, ``status_sha256``, ``diff_sha256``) remain
    best-effort and are ``None`` / empty-hash outside a Git repo, kept for audit
    continuity; only ``files`` is consulted for evidence checks.
    """
    root = workspace.resolve()
    head = _git_output(root, "rev-parse", "HEAD").decode("utf-8", "replace").strip()
    status = _git_output(root, "status", "--porcelain=v1", "-z")
    diff = _git_output(root, "diff", "--binary", "HEAD") if head else b""

    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune .git in place so os.walk does not descend into it.
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            full = Path(dirpath, name)
            try:
                relative = full.relative_to(root).as_posix()
            except ValueError:
                continue
            try:
                resolved = resolve_in_workspace(relative, root)
            except Exception:
                # Escaping symlinks and out-of-workspace paths are skipped,
                # not trusted — confinement is enforced on the baseline too.
                continue
            fingerprint = _fingerprint(resolved)
            if fingerprint is not None:
                files[relative] = fingerprint
    return WorkspaceBaseline(
        head_sha=head or None,
        status_sha256=hashlib.sha256(status).hexdigest(),
        diff_sha256=hashlib.sha256(diff).hexdigest(),
        files=files,
        captured_at=datetime.now(UTC),
    )


def _check_command_evidence(
    claim: Claim,
    evidence: Sequence[CommandEvidence | FileEvidence],
    *,
    history: Sequence[ExecutedStep],
    run_id: str,
) -> tuple[bool, str]:
    required_kind = {
        Claim.TESTS_PASS: "test",
        Claim.BUILDS: "build",
        Claim.LINT_CLEAN: "lint",
        Claim.TYPECHECK_CLEAN: "typecheck",
    }[claim]
    for item in evidence:
        if not isinstance(item, CommandEvidence) or item.kind != required_kind:
            continue
        step = next(
            (
                candidate
                for candidate in history
                if candidate.id == item.step_id and candidate.run_id == run_id
            ),
            None,
        )
        if step is None or step.tool != "run_command":
            continue
        if not step.result.ok or step.result.exit_code != 0:
            continue
        command = str(step.arguments.get("command", ""))
        if classify_command_kind(command) != required_kind:
            continue
        return True, f"step {step.id} ran `{command}` as {required_kind} and exited 0"
    return False, (
        f"no cited step in run {run_id!r} is a successful {required_kind} command"
    )


_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_COMMAND_KINDS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "test",
        (
            ("pytest",),
            ("uv", "run", "pytest"),
            ("python", "-m", "pytest"),
            ("python3", "-m", "pytest"),
            ("pnpm", "test"),
            ("npm", "test"),
            ("cargo", "test"),
        ),
    ),
    (
        "lint",
        (("ruff", "check"), ("uv", "run", "ruff", "check")),
    ),
    (
        "typecheck",
        (
            ("mypy",),
            ("uv", "run", "mypy"),
            ("pyright",),
            ("pnpm", "exec", "tsc"),
            ("tsc",),
        ),
    ),
    (
        "build",
        (
            ("uv", "build"),
            ("python", "-m", "build"),
            ("python3", "-m", "build"),
            ("pnpm", "build"),
            ("pnpm", "run", "build"),
            ("npm", "run", "build"),
            ("cargo", "build"),
            ("make", "build"),
        ),
    ),
)


def classify_command_kind(command: str) -> str | None:
    """Classify one non-compound command for verification purposes."""
    segments = [segment for segment in split_segments(command) if segment.strip()]
    if len(segments) != 1:
        return None
    try:
        tokens = shlex.split(segments[0])
    except ValueError:
        return None
    while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
        tokens.pop(0)
    if tokens and tokens[0] == "env":
        tokens.pop(0)
        while tokens and (_ENV_ASSIGNMENT.match(tokens[0]) or tokens[0].startswith("-")):
            tokens.pop(0)
    for kind, prefixes in _COMMAND_KINDS:
        if any(tuple(tokens[: len(prefix)]) == prefix for prefix in prefixes):
            return kind
    return None
