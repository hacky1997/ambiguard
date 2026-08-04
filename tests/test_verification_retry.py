"""Test verification retry bounding (AGENTS.md rule 12).

Verification retries MUST be bounded at 2, then escalate by downgrading
the response to CLARIFY rather than looping infinitely.
"""

from __future__ import annotations

from app.graph.agents.verification import verification_node
from app.graph.routing import route_by_verification
from app.graph.state import AgentState, VerificationResult


class TestVerificationRetryBounding:
    """Test rule 12: bounded retries at 2 -> escalate to CLARIFY."""

    def test_first_failure_increments_retry_count(self) -> None:
        """Initial verification failure produces retry_count=0, 1st retry produces retry_count=1."""
        state: AgentState = {
            "answer": "Completely hallucinated claim with zero backing",
            "evidence": [],
        }

        result = verification_node(state)
        ver = result["verification"]

        assert ver["passed"] is False
        assert ver["retry_count"] == 0

        # Now simulate 1st retry attempt
        state["verification"] = ver
        result_retry1 = verification_node(state)
        ver_retry1 = result_retry1["verification"]
        assert ver_retry1["retry_count"] == 1

    def test_retry_count_below_2_routes_to_research(self) -> None:
        """retry_count < 2 routes back to research node."""
        state: AgentState = {
            "verification": VerificationResult(
                passed=False,
                ungrounded_claims=["Low grounding"],
                confidence=0.4,
                retry_count=1,
            )
        }

        next_node = route_by_verification(state)
        assert next_node == "research"

    def test_retry_count_2_escalates_and_downgrades_to_clarify(self) -> None:
        """retry_count >= 2 escalates response to CLARIFY and terminates."""
        state: AgentState = {
            "answer": "Another hallucinated answer",
            "evidence": [],
            "verification": VerificationResult(
                passed=False,
                ungrounded_claims=["Low grounding"],
                confidence=0.4,
                retry_count=1,
            ),
        }

        # Execute 2nd retry attempt
        result = verification_node(state)
        ver = result["verification"]

        assert ver["passed"] is False
        assert ver["retry_count"] == 2
        # Answer must be cleared and replaced with clarifying question
        assert result["answer"] is None
        assert result.get("clarifying_question") is not None
        assert "Verification limit reached" in result["clarifying_question"]

        # Routing must terminate (end)
        state_after: AgentState = {**state, **result}
        next_node = route_by_verification(state_after)
        assert next_node == "end"
