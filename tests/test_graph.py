"""Tests for LangGraph multi-agent system execution and state contracts."""

from __future__ import annotations

import pytest

from app.graph.builder import build_graph
from app.graph.state import AgentState, Evidence, VerificationResult


class TestGraphStateContracts:
    """Verify AGENTS.md state rules (rules 8, 9, 10, 11, 12)."""

    def test_graph_compiles(self) -> None:
        """Graph compiles without errors."""
        graph = build_graph()
        assert graph is not None

    def test_answer_path_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ANSWER path routes supervisor -> research -> synthesis -> verification -> END."""
        monkeypatch.setenv("GATE_CHECKPOINT_PATH", "")
        graph = build_graph()
        initial_state: AgentState = {
            "thread_id": "t1",
            "question": "What is the capital of France?",
            "messages": [],
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("gate") is not None
        assert final_state.get("answer") is not None
        assert final_state.get("verification") is not None
        assert final_state["verification"]["passed"] is True

    def test_clarify_path_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLARIFY path routes supervisor -> clarification -> END."""
        monkeypatch.setenv("GATE_CHECKPOINT_PATH", "")
        graph = build_graph()
        initial_state: AgentState = {
            "thread_id": "t2",
            "question": "What are the side effects of it?",
            "messages": [],
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("gate") is not None
        assert final_state["gate"]["behaviour"] == "CLARIFY"
        assert final_state.get("clarifying_question") is not None
        # Word limit <= 25
        words = final_state["clarifying_question"].split()
        assert len(words) <= 25

    def test_resolved_question_precedence(self) -> None:
        """AGENTS.md rule 11: resolved_question takes precedence over question."""
        graph = build_graph()
        initial_state: AgentState = {
            "thread_id": "t3",
            "question": "What are the side effects of it?",  # Would trigger CLARIFY
            "resolved_question": "What are the side effects of Medication Alpha?",  # Specific -> ANSWER
            "messages": [],
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("gate") is not None
        # With resolved_question specified, should route cleanly
        assert final_state.get("answer") is not None
