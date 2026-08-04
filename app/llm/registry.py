"""LLM provider registry — instantiates the right provider from settings.

Falls back to MockProvider when the requested provider is unavailable,
ensuring the app always boots without credentials (AGENTS.md rule 3).
"""

from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LLMProvider, LLMResponse
from app.llm.mock_provider import MockProvider

logger = logging.getLogger(__name__)


def get_provider(provider_name: str, **kwargs: Any) -> MockProvider | Any:
    """Get an LLM provider by name.

    Args:
        provider_name: One of 'mock', 'openai'.
        **kwargs: Provider-specific arguments (api_key, model, etc.).

    Returns:
        An LLM provider satisfying the LLMProvider protocol.
        Falls back to MockProvider on any failure.
    """
    if provider_name == "mock":
        return MockProvider()

    if provider_name == "openai":
        api_key: str | None = kwargs.get("api_key")
        model: str = kwargs.get("model", "gpt-4.1")

        if not api_key:
            logger.warning("No OpenAI API key — falling back to mock provider")
            return MockProvider()

        try:
            from app.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(api_key=api_key, model=model)
        except ImportError:
            logger.warning(
                "openai package not installed — falling back to mock provider"
            )
            return MockProvider()

    logger.warning("Unknown provider '%s' — falling back to mock", provider_name)
    return MockProvider()
