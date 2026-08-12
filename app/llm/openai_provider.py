"""OpenAI LLM provider for the LLM judge arm.

Used by eval/arms/llm_judge_arm.py to ask an LLM directly whether a
question is ambiguous. Temperature 0, few-shot. This is the arm whose
cost and non-determinism the comparison table quantifies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.llm.base import LLMResponse

logger = logging.getLogger(__name__)

# Approximate costs per 1K tokens (USD) — updated for mid-2026 pricing
_COST_PER_1K_INPUT: dict[str, float] = {
    "gpt-4.1": 0.002,
    "gpt-4.1-mini": 0.0004,
    "gpt-4o": 0.005,
}

_COST_PER_1K_OUTPUT: dict[str, float] = {
    "gpt-4.1": 0.008,
    "gpt-4.1-mini": 0.0016,
    "gpt-4o": 0.015,
}

# Fallback cost if model not in table
_DEFAULT_INPUT_COST: float = 0.01
_DEFAULT_OUTPUT_COST: float = 0.03


class OpenAIProvider:
    """OpenAI provider for the LLM judge arm.

    Requires an API key. If the openai package is not installed, raises
    ImportError immediately rather than failing silently at call time.
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1") -> None:
        self._model: str = model
        try:
            import openai  # type: ignore[import-untyped]

            self._client: Any = openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

    @property
    def model_name(self) -> str:
        """Return the OpenAI model identifier."""
        return self._model

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Complete using OpenAI API.

        Default temperature 0 for the judge arm (spec §3.1).
        Max tokens capped at 50 since we only need a behaviour label.
        """
        start: float = time.perf_counter()

        response: Any = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=50,
        )

        content: str = response.choices[0].message.content or ""
        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        # Estimate cost from token usage
        usage: Any = response.usage
        input_cost: float = (
            usage.prompt_tokens / 1000 * _COST_PER_1K_INPUT.get(self._model, _DEFAULT_INPUT_COST)
        )
        output_cost: float = (
            usage.completion_tokens
            / 1000
            * _COST_PER_1K_OUTPUT.get(self._model, _DEFAULT_OUTPUT_COST)
        )

        return LLMResponse(
            content=content.strip(),
            model=self._model,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=round(input_cost + output_cost, 6),
            deterministic=False,
        )
