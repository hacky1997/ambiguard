"""Health and readiness check endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.observability.metrics import metrics
from app.settings import get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    provider: str
    checkpoint_loaded: bool
    verification_rejection_rate: float
    clarify_resolution_rate: float


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness & readiness endpoint."""
    settings = get_settings()
    has_cp = bool(settings.gate_checkpoint_path and settings.gate_checkpoint_path.exists())

    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        checkpoint_loaded=has_cp,
        verification_rejection_rate=metrics.verification_rejection_rate,
        clarify_resolution_rate=metrics.clarify_resolution_rate,
    )
