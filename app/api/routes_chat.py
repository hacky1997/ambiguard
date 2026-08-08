"""Chat API router — handles turn processing and interrupt/resume.

Endpoints:
  POST /api/chat — submit question to multi-agent graph
  POST /api/chat/resume — resume CLARIFY thread with user's clarification answer
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, ResumeRequest
from app.graph.builder import build_graph
from app.graph.state import AgentState
from app.observability.metrics import metrics

router = APIRouter(tags=["Chat"])

# Simple in-memory thread storage for session state
_THREAD_SESSIONS: dict[str, AgentState] = {}
_COMPILED_GRAPH = build_graph()


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Submit a question to the AmbiGuard multi-agent pipeline."""
    initial_state: AgentState = {
        "thread_id": request.thread_id,
        "question": request.question,
        "resolved_question": None,
        "messages": [],
    }

    try:
        final_state: AgentState = _COMPILED_GRAPH.invoke(initial_state)
        _THREAD_SESSIONS[request.thread_id] = final_state

        gate = final_state.get("gate")
        behaviour = gate.get("behaviour", "ANSWER") if gate else "ANSWER"

        # Record operational metrics
        if gate:
            metrics.record_gate_decision(
                behaviour=behaviour,
                latency_ms=gate.get("latency_ms", 0.0),
                fallback_used=gate.get("fallback_used", False),
            )

        ver = final_state.get("verification")
        if ver:
            metrics.record_verification(ver.get("passed", True))

        return ChatResponse(
            thread_id=request.thread_id,
            behaviour=behaviour,
            gate=gate,
            answer=final_state.get("answer"),
            clarifying_question=final_state.get("clarifying_question"),
            alternatives=final_state.get("alternatives"),
            citations=final_state.get("citations", []),
            verification_passed=ver.get("passed", True) if ver else True,
            error=final_state.get("error"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/resume", response_model=ChatResponse)
def resume_endpoint(request: ResumeRequest) -> ChatResponse:
    """Resume a CLARIFY thread after user provides their clarification choice.

    AGENTS.md rule 11: resolved_question takes precedence over question.
    The gate re-runs on the enriched question.
    """
    session = _THREAD_SESSIONS.get(request.thread_id)
    orig_q = session.get("question", "") if session else ""

    # Enrich question with user clarification
    enriched_question = f"{orig_q} ({request.user_clarification})"

    resume_state: AgentState = {
        "thread_id": request.thread_id,
        "question": orig_q,
        "resolved_question": enriched_question,  # AGENTS.md rule 11
        "messages": [],
    }

    try:
        final_state: AgentState = _COMPILED_GRAPH.invoke(resume_state)
        _THREAD_SESSIONS[request.thread_id] = final_state

        gate = final_state.get("gate")
        behaviour = gate.get("behaviour", "ANSWER") if gate else "ANSWER"

        # Record clarify resolution metric
        resolved_to_answer = (behaviour == "ANSWER")
        metrics.record_clarify_resume(resolved_to_answer=resolved_to_answer)

        ver = final_state.get("verification")

        return ChatResponse(
            thread_id=request.thread_id,
            behaviour=behaviour,
            gate=gate,
            answer=final_state.get("answer"),
            clarifying_question=final_state.get("clarifying_question"),
            alternatives=final_state.get("alternatives"),
            citations=final_state.get("citations", []),
            verification_passed=ver.get("passed", True) if ver else True,
            error=final_state.get("error"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
