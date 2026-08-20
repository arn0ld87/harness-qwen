"""The agent loop: role rotation, bounded retries, and evidence-gated verification."""

from harness.agent.loop import AgentLoop
from harness.agent.retry import ErrorCategory, RetryDecision, RetryPolicy
from harness.agent.roles import DEFAULT_CYCLE, RoleSequencer
from harness.agent.verifier import (
    Claim,
    ExecutedStep,
    VerificationOutcome,
    Verifier,
    detect_claims,
)

__all__ = [
    "DEFAULT_CYCLE",
    "AgentLoop",
    "Claim",
    "ErrorCategory",
    "ExecutedStep",
    "RetryDecision",
    "RetryPolicy",
    "RoleSequencer",
    "VerificationOutcome",
    "Verifier",
    "detect_claims",
]
