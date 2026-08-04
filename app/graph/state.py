"""AgentState and typed handoff contracts between agents.

Rules (AGENTS.md rules 8, 9, 10, 11):
  - Never invent state keys: extend AgentState first, then use it.
  - Nodes return partial dicts: NO in-place mutation of state.
  - `messages` is the ONLY reducer field (Annotated[list[dict], add]). Everything else is last-write-wins.
  - `resolved_question` takes precedence over `question` everywhere downstream when set.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from app.gate.base import GateDecision

Behaviour = Literal["ANSWER", "CLARIFY", "ALTERNATIVES"]


class Evidence(TypedDict):
    """Typed evidence record retrieved by the research agent."""

    doc_id: str
    text: str
    score: float
    source: str
    retrieved_by: str  # agent or query string that produced this evidence


class VerificationResult(TypedDict):
    """Typed verification result produced by the verification agent."""

    passed: bool
    ungrounded_claims: list[str]
    confidence: float
    retry_count: int


def _reduce_messages(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reducer for messages array field."""
    return left + right


class AgentState(TypedDict, total=False):
    """Shared state across the multi-agent graph.

    Only messages uses a reducer field. All other fields are last-write-wins.
    Nodes MUST return partial dicts matching this schema.
    """

    thread_id: str
    messages: Annotated[list[dict[str, Any]], _reduce_messages]  # ONLY reducer field
    question: str
    resolved_question: str | None  # Takes precedence downstream when set
    evidence: list[Evidence]
    gate: GateDecision | None
    answer: str | None
    clarifying_question: str | None
    alternatives: list[dict[str, Any]] | None
    verification: VerificationResult | None
    citations: list[str]
    trace_id: str | None
    error: str | None
