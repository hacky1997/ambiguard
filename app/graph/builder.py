"""LangGraph state graph builder for AmbiGuard multi-agent QA system.

Links:
    Supervisor (gate) → [CLARIFY → Clarification] | [ANSWER / ALTERNATIVES → Research]
    Research → Synthesis → Verification
    Verification → [Passed → END] | [Failed & retry < 2 → Research] | [Failed & retry >= 2 → Escalated CLARIFY → END]
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.agents.clarification import clarification_node
from app.graph.agents.research import research_node
from app.graph.agents.supervisor import supervisor_node
from app.graph.agents.synthesis import synthesis_node
from app.graph.agents.verification import verification_node
from app.graph.routing import route_by_gate, route_by_verification
from app.graph.state import AgentState


def build_graph() -> Any:
    """Construct and compile the LangGraph workflow for AmbiGuard."""
    builder = StateGraph(AgentState)

    # Add agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("verification", verification_node)

    # Set entry point
    builder.set_entry_point("supervisor")

    # Conditional routing after supervisor gate
    builder.add_conditional_edges(
        "supervisor",
        route_by_gate,
        {
            "clarification": "clarification",
            "research": "research",
        },
    )

    # Clarification is terminal node (waiting for user response / interrupt)
    builder.add_edge("clarification", END)

    # Research -> Synthesis -> Verification pipeline
    builder.add_edge("research", "synthesis")
    builder.add_edge("synthesis", "verification")

    # Conditional routing after verification
    builder.add_conditional_edges(
        "verification",
        route_by_verification,
        {
            "research": "research",
            "end": END,
        },
    )

    return builder.compile()
