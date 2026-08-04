"""Research agent node — executes multi-hop retrieval over corpus.

Returns typed Evidence records.
AGENTS.md rule 11: resolved_question takes precedence over question downstream when set.
AGENTS.md rule 9: return partial dicts.
"""

from __future__ import annotations

from typing import Any

from app.graph.state import AgentState, Evidence
from app.retrieval.in_memory import InMemoryRetriever


def research_node(state: AgentState) -> dict[str, Any]:
    """Execute research retrieval for the query.

    Uses resolved_question if set, otherwise question.
    """
    retriever = InMemoryRetriever()
    query: str = state.get("resolved_question") or state.get("question", "")

    evidence_list: list[Evidence] = retriever.search(query, top_k=3)

    return {
        "evidence": evidence_list,
    }
