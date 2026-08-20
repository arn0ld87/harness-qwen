"""The prefix contract on the assembler (issue #22).

CONTEXT.md §4 makes byte-stability an assertion, not a convention: the stable
prefix is hashed on every build and compared to the previous hash. These tests
pin the contract directly on ``PromptAssembler`` — the append zone may grow,
roles may switch, tools may return and retrieval may fire, and none of that
moves the prefix hash. A legitimate prefix change is declared first, attributed
to a fixed reason, and recorded; an undeclared one raises.

The contract itself is implemented already; the gap these tests close is that
it had no direct unit tests, only indirect cover through the chat loop and the
retrieval tool.
"""

from __future__ import annotations

import pytest

from harness.context.assembler import (
    InvalidationReason,
    PrefixViolation,
    PromptAssembler,
)
from harness.context.budget import TokenBudget
from harness.core import Role, ToolResult, ToolSpec


def _assembler() -> PromptAssembler:
    return PromptAssembler(
        system="You are a careful agent.",
        task="Fix the failing test.",
        tools=(ToolSpec(name="run", description="run a command",
                        parameters={"type": "object", "properties": {}}),),
        repo_map="src/  main.py\ntests/  test_main.py",
        budget=TokenBudget(),
    )


# -- the append zone never moves the prefix hash ---------------------------


def test_the_first_build_records_an_initial_invalidation() -> None:
    asm = _assembler()
    asm.build()
    assert asm.invalidations[0].reasons == [InvalidationReason.INITIAL]
    assert asm.invalidations[0].applied is True


def test_role_directives_do_not_change_the_prefix_hash() -> None:
    asm = _assembler()
    h0 = asm.build().prefix_hash
    asm.append_role_directive(Role.PLANNER)
    assert asm.build().prefix_hash == h0
    asm.append_role_directive(Role.CODER)
    assert asm.build().prefix_hash == h0
    # Many handovers, one hash — that is the point of keeping roles in the
    # append zone (CONTEXT.md §4).
    asm.append_role_directive(Role.TESTER)
    asm.append_role_directive(Role.REVIEWER)
    assert asm.build().prefix_hash == h0


def test_a_tool_result_does_not_change_the_prefix_hash() -> None:
    asm = _assembler()
    h0 = asm.build().prefix_hash
    asm.append_tool_result(ToolResult(tool="run", ok=True, content="1 passed"))
    assert asm.build().prefix_hash == h0
    asm.append_tool_result(ToolResult(tool="run", ok=False, content="boom",
                                      truncated=True, full_output_ref="out/1.txt"))
    assert asm.build().prefix_hash == h0


def test_retrieval_does_not_change_the_prefix_hash() -> None:
    asm = _assembler()
    h0 = asm.build().prefix_hash
    asm.append_retrieved("fact: the cache is sacred", source="facts")
    assert asm.build().prefix_hash == h0


def test_an_append_zone_rewrite_does_not_change_the_prefix_hash() -> None:
    """Compression rewrites the append zone, not the prefix: the cached prefix
    still hits, even though everything after the edit point is reprocessed."""
    from harness.core import Message

    asm = _assembler()
    asm.build()
    h0 = asm.prefix_hash()
    asm.rewrite_append(
        [Message(role="user", content="[summary] prior steps condensed")],
        note="summarise old tool output",
    )
    assert asm.build().prefix_hash == h0
    assert asm.append_rewrites == ["summarise old tool output"]


# -- declared invalidation -------------------------------------------------


def test_a_declared_task_change_changes_the_hash_and_records_the_reason() -> None:
    asm = _assembler()
    h0 = asm.build().prefix_hash
    asm.invalidate(InvalidationReason.TASK_CHANGED, note="new goal from the user")
    asm.set_task("Fix a different test instead.")
    built = asm.build()
    assert built.prefix_hash != h0
    record = asm.invalidations[-1]
    assert record.applied is True
    assert InvalidationReason.TASK_CHANGED in record.reasons
    assert "new goal from the user" in record.notes
    assert record.previous_hash == h0
    assert record.new_hash == built.prefix_hash
    assert "task" in record.changed_segments


def test_a_declared_tools_change_records_the_reason() -> None:
    asm = _assembler()
    asm.build()
    asm.invalidate(InvalidationReason.TOOLS_CHANGED)
    asm.set_tools((ToolSpec(name="edit", description="edit a file",
                            parameters={"type": "object", "properties": {}}),))
    asm.build()
    record = asm.invalidations[-1]
    assert InvalidationReason.TOOLS_CHANGED in record.reasons
    assert "tools" in record.changed_segments
    assert record.applied is True


def test_two_declared_reasons_land_in_one_record() -> None:
    """A new goal that also registers a tool is one reprocess, two reasons."""
    asm = _assembler()
    asm.build()
    asm.invalidate(InvalidationReason.TASK_CHANGED)
    asm.invalidate(InvalidationReason.TOOLS_CHANGED)
    asm.set_task("New goal.")
    asm.set_tools((ToolSpec(name="edit", description="edit",
                            parameters={"type": "object", "properties": {}}),))
    asm.build()
    record = asm.invalidations[-1]
    assert InvalidationReason.TASK_CHANGED in record.reasons
    assert InvalidationReason.TOOLS_CHANGED in record.reasons


def test_a_declaration_without_a_following_change_is_recorded_not_carried() -> None:
    """A stale declaration must not authorise a later, genuinely unannounced
    change — so it is recorded as applied=False and dropped."""
    asm = _assembler()
    asm.build()
    asm.invalidate(InvalidationReason.TASK_CHANGED)
    asm.build()  # nothing changed
    record = asm.invalidations[-1]
    assert record.applied is False
    assert record.changed_segments == []

    # And it does not cover a later undeclared change:
    asm.set_task("Undeclared new goal.")
    with pytest.raises(PrefixViolation):
        asm.build()


# -- undeclared change -----------------------------------------------------


def test_an_undeclared_task_change_raises_prefix_violation() -> None:
    asm = _assembler()
    h0 = asm.build().prefix_hash
    asm.set_task("Undeclared new goal.")
    with pytest.raises(PrefixViolation) as exc:
        asm.build()
    assert exc.value.expected == h0
    assert exc.value.segments == ["task"]


def test_an_undeclared_repo_map_change_names_the_segment() -> None:
    asm = _assembler()
    asm.build()
    asm.set_repo_map("completely different map")
    with pytest.raises(PrefixViolation) as exc:
        asm.build()
    assert "repo_map" in exc.value.segments