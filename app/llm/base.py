"""LLM provider protocol and response types."""

from __future__ import annotations

from typing import Protocol, TypedDict


class LLMResponse(TypedDict):
    """Standard response from any LLM provider."""

    content: str
    model: str
    latency_ms: float
    cost_usd: float  # estimated cost for this single call
    deterministic: bool  # whether identical input → identical output


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Implementations: MockProvider (testing), OpenAIProvider (LLM judge arm).
    The mock provider is used when no API key is configured, ensuring
    the app boots without credentials (AGENTS.md rule 3).
    """

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Complete a prompt and return a structured response."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...
