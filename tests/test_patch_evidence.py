import subprocess
from datetime import UTC, datetime
from pathlib import Path

from harness.agent.verifier import Verifier, capture_workspace_baseline
from harness.core import AnswerAction, FileEvidence


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


def _verify_patch(workspace: Path, baseline: object, path: str) -> bool:
    outcome = Verifier().verify(
        AnswerAction(
            content=f"Fixed the {path}",
            evidence=[FileEvidence(kind="patch", path=path)],
        ),
        history=[],
        run_id="run-1",
        workspace=workspace,
        since=datetime.now(UTC),
        baseline=baseline,
    )
    return outcome.verified


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
