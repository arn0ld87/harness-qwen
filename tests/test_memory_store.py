import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from harness.core import RunRuntimeState, StepStatus, TaskState
from harness.memory import MemoryStore


def _task_state(workspace: Path) -> TaskState:
    now = datetime.now(UTC)
    return TaskState(
        run_id="run-1",
        goal="test",
        workspace=str(workspace),
        created_at=now,
        updated_at=now,
    )


def test_task_and_runtime_state_round_trip(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        task = _task_state(tmp_path)
        runtime = RunRuntimeState(run_id=task.run_id, retries_used=2, tool_calls=3)

        store.save_task_state(task)
        store.save_runtime_state(runtime)

        assert store.load_task_state(task.run_id) == task
        assert store.load_runtime_state(task.run_id) == runtime
        assert store.resumable_runs() == [task.run_id]


def test_schema_version_one_migrates_runtime_state_table(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    with MemoryStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE runtime_state")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with MemoryStore(database) as migrated:
        assert migrated.schema_version == 2
        assert migrated.has_table("runtime_state")


def test_fact_search_handles_fts_syntax_as_plain_terms(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        store.put_fact("security-boundary", "workspace confinement", "architecture")

        matches = store.search_facts('"workspace')

        assert [fact.key for fact in matches] == ["security-boundary"]


def test_tool_checkpoint_updates_step_task_and_runtime_atomically(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        task = _task_state(tmp_path)
        store.save_task_state(task)
        step_id = store.append_step(task.run_id, step_index=0, action="tool_call")
        task.step_index = 1
        runtime = RunRuntimeState(run_id=task.run_id, tool_calls=1)

        store.save_tool_checkpoint(
            step_id=step_id,
            status=StepStatus.DONE,
            task_state=task,
            runtime_state=runtime,
            exit_code=0,
        )

        assert store.get_steps(task.run_id)[0].status is StepStatus.DONE
        assert store.load_task_state(task.run_id).step_index == 1
        assert store.load_runtime_state(task.run_id).tool_calls == 1
