"""Conditional routing functions for LangGraph execution."""

from __future__ import annotations

from typing import Literal

from app.graph.state import AgentState


def route_by_gate(state: AgentState) -> Literal["clarification", "research"]:
    """Route after supervisor gate based on behaviour decision."""
    gate = state.get("gate")
    if not gate:
        return "research"

    behaviour = gate.get("behaviour", "ANSWER")
    if behaviour == "CLARIFY":
        return "clarification"
    return "research"


def route_by_verification(state: AgentState) -> Literal["research", "end"]:
    """Route after verification node based on verification result.

    AGENTS.md rule 12: Bounded at 2 retries, then escalate to CLARIFY.
    """
    verification = state.get("verification")
    if not verification:
        return "end"

    if verification.get("passed", True):
        return "end"

    retry_count = verification.get("retry_count", 0)
    if retry_count < 2:
        return "research"  # Retry research loop

    # Retry limit reached (retry_count >= 2) -> escalated to CLARIFY, terminal state
    return "end"
