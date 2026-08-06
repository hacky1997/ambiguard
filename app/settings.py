"""Application settings — pydantic-settings, single source of truth.

No magic constants elsewhere in the codebase. All configuration lives here.
The app MUST boot with a completely empty .env file (AGENTS.md rule 3).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AmbiGuard configuration.

    Every field has a default that allows the app to boot without any
    environment variables set. External dependencies degrade gracefully:
      - No checkpoint → heuristic gate (fallback_used=True)
      - No API key → mock LLM provider
      - No Qdrant → in-memory retrieval (Phase 2)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AMBIGUARD_",
        extra="ignore",
    )

    # --- Gate checkpoint ---
    gate_checkpoint_path: Path | None = Field(
        default=None,
        description="Local path to CenterDistill checkpoint directory",
    )
    gate_hf_repo: str | None = Field(
        default=None,
        description="HF Hub repo ID for CenterDistill checkpoint",
    )

    # --- LLM provider ---
    llm_provider: str = Field(
        default="mock",
        description="LLM provider: 'openai', 'ollama', or 'mock'",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional — falls back to mock provider)",
    )
    openai_model: str = Field(
        default="gpt-4.1",
        description="OpenAI model for LLM judge arm",
    )

    # --- Eval ---
    eval_bootstrap_resamples: int = Field(
        default=10_000,
        description="Number of bootstrap resamples for CI computation",
    )
    eval_determinism_runs: int = Field(
        default=3,
        description="Number of identical runs for determinism checking",
    )


def get_settings() -> Settings:
    """Load settings, gracefully handling missing .env."""
    return Settings()
