"""Sequential role rotation for the agent loop (ARCHITECTURE.md, "Agent loop").

Planner, coder, tester and reviewer are personas the same model is asked to
adopt one step at a time, never concurrent processes — a second concurrent
request was measured to roughly halve generation throughput on this hardware,
which is exactly what sequential roles avoid. ``PromptAssembler`` already
appends the role directive as a user turn at the tail
(``append_role_directive``), so the prefix it guards is never touched by a
role switch. This module owns only the *sequencing*: which role a given step
number gets.
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.context.assembler import PromptAssembler
from harness.core import Role

DEFAULT_CYCLE: tuple[Role, ...] = (Role.CODER, Role.TESTER, Role.REVIEWER)
"""Steady-state rotation once planning is done: coder acts, tester checks the
action, reviewer judges the evidence before the next coder step. The same
three-way split ``Verifier`` re-applies after the run ends — this just runs
it live, one persona per step."""


class RoleSequencer:
    """Maps a step index to a role, deterministically and without side effects.

    A pure function of ``step_index``: a resumed run recomputes the same role
    for the step it left off on without needing any state of its own —
    ``TaskState.step_index`` is already the durable record.
    """

    def __init__(
        self,
        *,
        planning_steps: int = 1,
        cycle: Sequence[Role] = DEFAULT_CYCLE,
    ) -> None:
        if planning_steps < 0:
            raise ValueError("planning_steps must not be negative")
        if not cycle:
            raise ValueError("cycle must name at least one role")
        self.planning_steps = planning_steps
        self.cycle: tuple[Role, ...] = tuple(cycle)

    def role_for_step(self, step_index: int) -> Role:
        """The role active at ``step_index`` (0-based).

        The first ``planning_steps`` steps are the planner; afterwards the
        role cycles through ``self.cycle`` forever, so a run that never
        finishes still has a well-defined role for every step it takes.
        """
        if step_index < self.planning_steps:
            return Role.PLANNER
        offset = (step_index - self.planning_steps) % len(self.cycle)
        return self.cycle[offset]

    def apply(self, assembler: PromptAssembler, step_index: int) -> Role:
        """Append this step's role directive to ``assembler`` and return it.

        Always an append-zone write. No directive text is passed, so the
        assembler's own ``DEFAULT_ROLE_DIRECTIVES`` phrasing is used — this
        module decides *when* a role applies, not what it says, so the two
        cannot drift apart by being maintained in two places.
        """
        role = self.role_for_step(step_index)
        assembler.append_role_directive(role)
        return role
