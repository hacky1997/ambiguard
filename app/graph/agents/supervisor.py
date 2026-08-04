"""Supervisor node — the learned gate.

Runs the CenterDistill classifier (or heuristic fallback) before any LLM token
is spent. Evaluates routing policy and sets state["gate"].

Rules:
  - resolved_question takes precedence over question downstream when set (AGENTS.md rule 11).
  - Return partial dict (AGENTS.md rule 9).
"""

from __future__ import annotations

from typing import Any

from app.gate.centerdistill import CenterDistillGate
from app.gate.thresholds import DEFAULT_THRESHOLDS, GateThresholds
from app.graph.state import AgentState
from app.settings import get_settings


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Execute the supervisor gate routing decision.

    AGENTS.md rule 11: resolved_question takes precedence over question everywhere downstream when set.
    """
    settings = get_settings()
    gate = CenterDistillGate(
        checkpoint_path=settings.gate_checkpoint_path,
        hf_repo=settings.gate_hf_repo,
        thresholds=DEFAULT_THRESHOLDS,
    )

    # Rule 11: resolved_question takes precedence
    query: str = state.get("resolved_question") or state.get("question", "")
    context: str = ""

    # If evidence is already present, combine text into context
    evidence = state.get("evidence", [])
    if evidence:
        context = " ".join(e["text"] for e in evidence)

    decision = gate(query, context)

    # Return partial dict — no in-place mutation (AGENTS.md rule 9)
    return {
        "gate": decision,
    }
