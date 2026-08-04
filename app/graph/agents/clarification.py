"""Clarification agent node — generates targeted clarifying questions.

SPEC.md §4.3 requirement:
  - Generates EXACTLY ONE question, <= 25 words.
  - Names a concrete competing interpretation from context.
  - Generic clarifications ("Could you clarify?") are a scored failure.
"""

from __future__ import annotations

import re
from typing import Any

from app.graph.state import AgentState


def clarification_node(state: AgentState) -> dict[str, Any]:
    """Generate a targeted clarifying question based on evidence context."""
    question: str = state.get("question", "")
    evidence = state.get("evidence", [])

    # Extract distinct entity or topic names from evidence to construct concrete question
    competing_topics: list[str] = []
    for ev in evidence:
        text = ev.get("text", "")
        # Find capitalized phrases or key terms
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for m in matches:
            if m not in competing_topics and m.lower() not in question.lower():
                competing_topics.append(m)

    if len(competing_topics) >= 2:
        clarifying_q = f"Are you asking about {competing_topics[0]} or {competing_topics[1]}?"
    elif len(competing_topics) == 1:
        clarifying_q = f"Did you mean regarding {competing_topics[0]} or general usage?"
    else:
        clarifying_q = f"Which specific context for '{question[:20]}' would you like answered?"

    # Enforce word count limit: <= 25 words
    words = clarifying_q.split()
    if len(words) > 25:
        clarifying_q = " ".join(words[:25]) + "?"

    return {
        "clarifying_question": clarifying_q,
    }
