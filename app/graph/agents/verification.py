"""Verification agent node — adversarial verifier.

AGENTS.md rule 12: Verification retries bounded at 2, then escalate to CLARIFY.
Never an unbounded reject/retry loop.
"""

from __future__ import annotations

from typing import Any

from app.graph.state import AgentState, VerificationResult


def verification_node(state: AgentState) -> dict[str, Any]:
    """Verify generated answer against retrieved evidence.

    AGENTS.md rule 12: Bounded at 2 retries, then escalate to CLARIFY.
    """
    answer: str = state.get("answer", "") or ""
    evidence = state.get("evidence", [])
    current_ver = state.get("verification")

    retry_count = (current_ver["retry_count"] + 1) if current_ver else 0

    # Grounding check: verify words in answer are backed by evidence text
    context_text = " ".join(e.get("text", "") for e in evidence).lower()
    ungrounded_claims: list[str] = []

    if not evidence and answer:
        ungrounded_claims.append("No evidence retrieved to support answer.")

    # Check for hallucinated key entities
    answer_words = answer.split()
    if len(answer_words) > 0 and len(context_text) > 0:
        # Simple check for ungrounded text
        match_count = sum(1 for w in answer_words if w.lower() in context_text)
        grounding_ratio = match_count / len(answer_words)
        if grounding_ratio < 0.3:
            ungrounded_claims.append(f"Low evidence grounding ratio ({grounding_ratio:.2f})")

    passed = len(ungrounded_claims) == 0

    verification_res = VerificationResult(
        passed=passed,
        ungrounded_claims=ungrounded_claims,
        confidence=0.9 if passed else 0.4,
        retry_count=retry_count,
    )

    result: dict[str, Any] = {
        "verification": verification_res,
    }

    # AGENTS.md rule 12: If retry_count >= 2 and failed, escalate by downgrading response to CLARIFY
    if not passed and retry_count >= 2:
        result["answer"] = None
        result["clarifying_question"] = (
            "Verification limit reached. Could you clarify which source "
            "or interpretation you prefer?"
        )

    return result
