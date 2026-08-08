"""AmbiGuard FastAPI Application Lifespan entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.settings import get_settings

logger = logging.getLogger("ambiguard")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan startup & shutdown tasks.

    Loads settings once at startup (AGENTS.md rule 3).
    """
    settings = get_settings()
    logger.info(
        "Booting AmbiGuard (provider=%s, checkpoint=%s)",
        settings.llm_provider,
        settings.gate_checkpoint_path,
    )
    yield
    logger.info("Shutting down AmbiGuard app.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI app instance."""
    app = FastAPI(
        title="AmbiGuard API",
        description="Multi-agent QA system with learned ambiguity routing (CenterDistill)",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api/chat")
    app.include_router(chat_router, prefix="/v1/chat")

    return app


app = create_app()
