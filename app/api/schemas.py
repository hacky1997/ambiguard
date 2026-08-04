"""Pydantic schemas for API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.gate.base import GateDecision


class ChatRequest(BaseModel):
    """Request payload for /api/chat endpoint."""

    thread_id: str = Field(default="default_thread", description="Session thread ID")
    question: str = Field(..., min_length=1, description="User question to process")


class ResumeRequest(BaseModel):
    """Request payload for /api/chat/resume endpoint (CLARIFY turn resume)."""

    thread_id: str = Field(..., description="Session thread ID to resume")
    user_clarification: str = Field(..., min_length=1, description="User response clarifying the query")


class ChatResponse(BaseModel):
    """Standardized response payload from AmbiGuard."""

    thread_id: str
    behaviour: str  # ANSWER, CLARIFY, or ALTERNATIVES
    gate: GateDecision | None = None
    answer: str | None = None
    clarifying_question: str | None = None
    alternatives: list[dict[str, Any]] | None = None
    citations: list[str] = Field(default_factory=list)
    verification_passed: bool = True
    error: str | None = None
