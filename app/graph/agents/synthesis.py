"""Synthesis agent node — formats answer and citations.

Merges evidence into a final response payload with citations.
AGENTS.md rule 9: Returns partial dict.
"""

from __future__ import annotations

from typing import Any

from app.graph.state import AgentState


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Synthesize final answer and citations from retrieved evidence."""
    query: str = state.get("resolved_question") or state.get("question", "")
    evidence = state.get("evidence", [])
    gate = state.get("gate")

    behaviour = gate["behaviour"] if gate else "ANSWER"

    citations: list[str] = [e["doc_id"] for e in evidence if "doc_id" in e]

    if behaviour == "ALTERNATIVES":
        alternatives_list: list[dict[str, Any]] = []
        for idx, ev in enumerate(evidence, 1):
            alternatives_list.append(
                {
                    "interpretation": f"Option {idx} ({ev.get('source', 'ref')})",
                    "answer": ev.get("text", ""),
                    "citation": ev.get("doc_id", ""),
                }
            )
        answer_text = f"The query '{query}' has multiple interpretations:\n" + "\n".join(
            f"- {alt['interpretation']}: {alt['answer']}" for alt in alternatives_list
        )
        return {
            "answer": answer_text,
            "alternatives": alternatives_list,
            "citations": citations,
        }

    # Single ANSWER path
    if evidence:
        primary_evidence = evidence[0]["text"]
        answer_text = f"Based on {citations[0] if citations else 'reference'}: {primary_evidence}"
    else:
        answer_text = f"Direct response to query '{query}'."

    return {
        "answer": answer_text,
        "citations": citations,
    }
