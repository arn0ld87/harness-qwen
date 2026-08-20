"""JSONL event journal for run telemetry.

Logs events to disk with immediate flush. Designed to be incapable of recording
model reasoning content: there is no parameter for it anywhere in the public API.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redact import redact


class RunJournal:
    """Write-only JSONL journal of a run's events.

    Metadata is automatically added to each record: timestamp, run_id.
    All string values are passed through redact() before writing.
    """

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        """Initialize journal.

        Args:
            run_dir: Directory where journal will be written. Created if absent.
            run_id: Identifier for this run, included in every record.
        """
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.run_dir / "journal.jsonl"
        self._file = open(  # noqa: SIM115 - the journal owns this handle until close()
            self.journal_path, "a", encoding="utf-8"
        )

    def log_step(
        self,
        *,
        step: str,
        role: str | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        context_size: int | None = None,
        prefix_hash: str | None = None,
        tool: str | None = None,
        tool_ms: float | None = None,
        exit_code: int | None = None,
        retry: int | None = None,
        error_kind: str | None = None,
    ) -> None:
        """Log a step in the run.

        All string fields are redacted before writing. No reasoning content
        is permitted at all in the signature.

        Args:
            step: Name of the step. Required.
            role: Role executing this step (planner, coder, tester, reviewer).
            latency_ms: Wall-clock time in milliseconds.
            prompt_tokens: Tokens in the prompt.
            completion_tokens: Tokens in the completion.
            cached_tokens: Prompt tokens served from cache.
            context_size: Total context window used.
            prefix_hash: Hash of the prompt prefix for cache tracking.
            tool: Tool name if a tool was called.
            tool_ms: Tool execution time in milliseconds.
            exit_code: Exit code if a tool was executed.
            retry: Retry attempt number if this is a retry.
            error_kind: Kind of error, if one occurred.
        """
        record: dict[str, Any] = {
            "kind": "step",
            "step": step,
        }
        if role is not None:
            record["role"] = role
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
        if prompt_tokens is not None:
            record["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            record["completion_tokens"] = completion_tokens
        if cached_tokens is not None:
            record["cached_tokens"] = cached_tokens
        if context_size is not None:
            record["context_size"] = context_size
        if prefix_hash is not None:
            record["prefix_hash"] = prefix_hash
        if tool is not None:
            record["tool"] = tool
        if tool_ms is not None:
            record["tool_ms"] = tool_ms
        if exit_code is not None:
            record["exit_code"] = exit_code
        if retry is not None:
            record["retry"] = retry
        if error_kind is not None:
            record["error_kind"] = error_kind

        self._write_record(record)

    def log_event(self, kind: str, **fields: Any) -> None:
        """Log an arbitrary event with arbitrary fields.

        String values are redacted before writing. This method should be used
        for logging that does NOT involve model reasoning or output.

        Args:
            kind: Event kind/type string.
            **fields: Arbitrary key-value pairs.
        """
        record = {"kind": kind}
        record.update(fields)
        self._write_record(record)

    def _write_record(self, record: dict[str, Any]) -> None:
        """Write a single record with timestamp and run_id.

        All string values are redacted before serialization.
        """
        record["timestamp"] = datetime.now(UTC).isoformat()
        record["run_id"] = self.run_id

        # Redact all string values recursively
        redacted_record = self._redact_dict(record)

        line = json.dumps(redacted_record, ensure_ascii=False, separators=(",", ":"))
        self._file.write(line + "\n")
        self._file.flush()

    @staticmethod
    def _redact_dict(obj: Any) -> Any:
        """Recursively redact all string values in a data structure."""
        if isinstance(obj, dict):
            return {k: RunJournal._redact_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [RunJournal._redact_dict(item) for item in obj]
        if isinstance(obj, str):
            return redact(obj)
        return obj

    def close(self) -> None:
        """Close the journal file."""
        if self._file:
            self._file.close()

    def __enter__(self) -> RunJournal:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
