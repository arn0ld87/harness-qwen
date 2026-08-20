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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness.core import AnswerAction, ToolResult
from harness.security import resolve_in_workspace
from harness.tools.git import git_diff


class Claim(enum.StrEnum):
    FILE_WRITTEN = "file_written"
    PATCH_APPLIED = "patch_applied"
    TESTS_PASS = "tests_pass"
    BUILDS = "builds"
    LINT_CLEAN = "lint_clean"


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
        "typecheck is clean", "typecheck passes", "type check passes",
    ),
}
"""Claim -> the specific phrasing that triggers it. Table-driven so a missing
check reads as a missing dict entry, not a buried conditional."""


@dataclass(frozen=True, slots=True)
class ExecutedStep:
    """One tool call the run actually made — what :class:`Verifier` audits.

    Built by the agent loop as it executes, never by the verifier: this
    module does not re-run anything the model claims it ran, it reads what
    already happened.
    """

    step_index: int
    tool: str
    arguments: dict[str, Any]
    result: ToolResult


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
        workspace: Path,
        since: datetime,
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
                claim, answer.evidence, history=history, workspace=workspace, since=since,
            )
            all_ok = all_ok and ok
            notes.append(f"{claim.value}: {note}")
        checked = [c.value for c in claims]
        return VerificationOutcome(verified=all_ok, notes=notes, claims_checked=checked)

    def _check(
        self,
        claim: Claim,
        evidence: Sequence[str],
        *,
        history: Sequence[ExecutedStep],
        workspace: Path,
        since: datetime,
    ) -> tuple[bool, str]:
        if claim is Claim.FILE_WRITTEN:
            return _check_file_written(evidence, workspace=workspace, since=since)
        if claim is Claim.PATCH_APPLIED:
            return _check_patch_applied(evidence, workspace=workspace)
        # tests_pass, builds and lint_clean all reduce to the same shape of
        # evidence per the table: a captured exit code 0 from something the
        # run actually ran, not a rerun triggered at verify time.
        return _check_command_evidence(claim, evidence, history=history)


def _check_file_written(
    evidence: Sequence[str], *, workspace: Path, since: datetime,
) -> tuple[bool, str]:
    for item in evidence:
        try:
            # Argument order is (path, workspace_root) — see
            # harness.security.workspace.resolve_in_workspace. Confinement is
            # still enforced; an evidence path outside the workspace is
            # skipped, not trusted.
            resolved = resolve_in_workspace(item, workspace)
        except Exception:
            continue
        if not resolved.exists():
            continue
        mtime = datetime.fromtimestamp(resolved.stat().st_mtime, tz=since.tzinfo)
        if mtime >= since:
            note = f"{item} exists, modified {mtime.isoformat()} (run started {since.isoformat()})"
            return True, note
    return False, (
        "no evidence path resolves to a file that exists with an mtime "
        "advanced since the run started"
    )


def _check_patch_applied(evidence: Sequence[str], *, workspace: Path) -> tuple[bool, str]:
    for item in evidence:
        result = git_diff(workspace, path=item)
        if result.ok and not result.content.rstrip().endswith("(empty)"):
            return True, f"{item}: `git diff` is non-empty"
    return False, "no evidence path has a non-empty `git diff` against the workspace"


def _check_command_evidence(
    claim: Claim, evidence: Sequence[str], *, history: Sequence[ExecutedStep],
) -> tuple[bool, str]:
    ran = [step for step in history if step.tool == "run_command" and step.result.ok]
    for item in evidence:
        needle = item.strip().lower()
        if not needle:
            continue
        for step in ran:
            command = str(step.arguments.get("command", "")).lower()
            if needle in command or needle in step.result.content.lower():
                ran_command = step.arguments.get("command")
                return True, f"step {step.step_index} ran `{ran_command}` and exited 0"
    return False, (
        f"no executed run_command step matching the cited evidence exited 0 "
        f"({claim.value} needs a captured, successful run, not a claim about one)"
    )
