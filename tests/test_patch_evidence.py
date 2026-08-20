import subprocess
from pathlib import Path
from typing import Literal

from harness.agent.verifier import Verifier, capture_workspace_baseline
from harness.core import AnswerAction, FileEvidence, WorkspaceBaseline


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "Harness Tests")
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("other = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def _verify(
    workspace: Path,
    baseline: WorkspaceBaseline | None,
    path: str,
    *,
    kind: Literal["file", "patch"] = "patch",
    phrase: str | None = None,
) -> bool:
    if phrase is None:
        phrase = "wrote the file" if kind == "file" else f"Fixed the {path}"
    outcome = Verifier().verify(
        AnswerAction(
            content=phrase,
            evidence=[FileEvidence(kind=kind, path=path)],
        ),
        history=[],
        run_id="run-1",
        workspace=workspace,
        baseline=baseline,
    )
    return outcome.verified


def _verify_patch(workspace: Path, baseline: WorkspaceBaseline | None, path: str) -> bool:
    return _verify(workspace, baseline, path, kind="patch")


def test_dirty_file_unchanged_during_run_is_not_patch_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    (workspace / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    assert _verify_patch(workspace, baseline, "tracked.py") is False


def test_clean_file_changed_during_run_is_patch_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    baseline = capture_workspace_baseline(workspace)

    (workspace / "tracked.py").write_text("value = 2\n", encoding="utf-8")

    assert _verify_patch(workspace, baseline, "tracked.py") is True


def test_additional_change_to_dirty_file_is_patch_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    (workspace / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    (workspace / "tracked.py").write_text("value = 3\n", encoding="utf-8")

    assert _verify_patch(workspace, baseline, "tracked.py") is True


def test_unrelated_dirty_file_does_not_supply_patch_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    (workspace / "other.py").write_text("other = 2\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    assert _verify_patch(workspace, baseline, "tracked.py") is False


# --- Issue #5: Git-independent baseline + content-fingerprint file evidence ---


def test_existing_file_in_non_git_workspace_is_not_patch_evidence(tmp_path: Path) -> None:
    workspace = tmp_path  # no `git init`
    (workspace / "plain.py").write_text("value = 1\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    assert _verify_patch(workspace, baseline, "plain.py") is False


def test_changed_file_in_non_git_workspace_is_patch_evidence(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / "plain.py").write_text("value = 1\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    (workspace / "plain.py").write_text("value = 2\n", encoding="utf-8")

    assert _verify_patch(workspace, baseline, "plain.py") is True


def test_new_file_in_non_git_workspace_is_patch_evidence(tmp_path: Path) -> None:
    workspace = tmp_path
    baseline = capture_workspace_baseline(workspace)

    (workspace / "created.py").write_text("value = 1\n", encoding="utf-8")

    assert _verify_patch(workspace, baseline, "created.py") is True


def test_gitignored_file_unchanged_is_not_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    (workspace / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (workspace / "ignored.py").write_text("ignored = 1\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    assert _verify_patch(workspace, baseline, "ignored.py") is False


def test_gitignored_file_changed_is_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    (workspace / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (workspace / "ignored.py").write_text("ignored = 1\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    (workspace / "ignored.py").write_text("ignored = 2\n", encoding="utf-8")

    assert _verify_patch(workspace, baseline, "ignored.py") is True


def test_deleted_file_is_not_patch_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    baseline = capture_workspace_baseline(workspace)

    (workspace / "tracked.py").unlink()

    assert _verify_patch(workspace, baseline, "tracked.py") is False


def test_file_evidence_uses_content_fingerprint_not_mtime(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    baseline = capture_workspace_baseline(workspace)

    # Same content, only mtime advanced (touch): must NOT count as written.
    (workspace / "tracked.py").write_text("value = 1\n", encoding="utf-8")

    assert _verify(workspace, baseline, "tracked.py", kind="file") is False


def test_file_evidence_changed_content_is_evidence(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    baseline = capture_workspace_baseline(workspace)

    (workspace / "tracked.py").write_text("value = 2\n", encoding="utf-8")

    assert _verify(workspace, baseline, "tracked.py", kind="file") is True


def test_file_evidence_new_file_in_non_git_is_evidence(tmp_path: Path) -> None:
    workspace = tmp_path
    baseline = capture_workspace_baseline(workspace)

    (workspace / "created.py").write_text("value = 1\n", encoding="utf-8")

    assert _verify(workspace, baseline, "created.py", kind="file") is True


def test_baseline_skips_git_internals(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    baseline = capture_workspace_baseline(workspace)

    # .git/ internals must never appear as tracked baseline files.
    assert not any(p.startswith(".git/") for p in baseline.files)
    # Tracked source files are present.
    assert "tracked.py" in baseline.files


def test_symlink_inside_workspace_is_fingerprinted(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / "target.py").write_text("value = 1\n", encoding="utf-8")
    (workspace / "link.py").symlink_to(workspace / "target.py")
    baseline = capture_workspace_baseline(workspace)

    assert "link.py" in baseline.files


def test_escaping_symlink_is_rejected_from_baseline(tmp_path: Path) -> None:
    workspace = tmp_path
    outside = tmp_path.parent / "outside_target.py"
    outside.write_text("secret = 1\n", encoding="utf-8")
    try:
        (workspace / "escape.py").symlink_to(outside)
        baseline = capture_workspace_baseline(workspace)

        # The escaping symlink is not a baseline entry we would honour.
        assert "escape.py" not in baseline.files
        # And it cannot serve as evidence (confinement rejects it).
        assert _verify_patch(workspace, baseline, "escape.py") is False
    finally:
        outside.unlink(missing_ok=True)


def test_escaping_symlink_patch_evidence_is_denied(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    outside = tmp_path.parent / "outside_escape.py"
    outside.write_text("secret = 1\n", encoding="utf-8")
    try:
        (workspace / "escape.py").symlink_to(outside)
        baseline = capture_workspace_baseline(workspace)

        assert _verify_patch(workspace, baseline, "escape.py") is False
    finally:
        outside.unlink(missing_ok=True)


def test_baseline_without_git_has_empty_summary_fields(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / "plain.py").write_text("value = 1\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)

    assert baseline.head_sha is None
    assert baseline.status_sha256 == hashlib_empty()
    assert "plain.py" in baseline.files


def hashlib_empty() -> str:
    import hashlib

    return hashlib.sha256(b"").hexdigest()


def test_file_evidence_without_baseline_is_unverified(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    outcome = Verifier().verify(
        AnswerAction(
            content="wrote the file tracked.py",
            evidence=[FileEvidence(kind="file", path="tracked.py")],
        ),
        history=[],
        run_id="run-1",
        workspace=workspace,
        baseline=None,
    )
    assert outcome.verified is False


