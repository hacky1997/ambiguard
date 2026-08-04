"""Mock LLM provider — deterministic, zero cost, no credentials needed.

Used as the default when no API key is configured, ensuring the app
always boots (AGENTS.md rule 3). Produces deterministic but varied
responses based on input hash so eval arms get exercised.
"""

from __future__ import annotations

import hashlib
import time

from app.llm.base import LLMResponse


# Behaviours the mock can return, indexed by hash bucket
_BEHAVIOURS: list[str] = ["ANSWER", "CLARIFY", "ALTERNATIVES"]


class MockProvider:
    """Deterministic mock LLM provider.

    Responses are varied (based on input hash) but fully reproducible.
    Cost is always $0. Satisfies the LLMProvider protocol.
    """

    def __init__(self) -> None:
        self._model: str = "mock-v1"

    @property
    def model_name(self) -> str:
        """Return the mock model identifier."""
        return self._model

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Return a deterministic response based on prompt hash.

        The hash ensures different inputs get different responses while
        remaining fully reproducible across runs.
        """
        start: float = time.perf_counter()

        # Deterministic bucket from MD5 hash
        h: str = hashlib.md5(prompt.encode(), usedforsecurity=False).hexdigest()
        bucket: int = int(h[:2], 16) % len(_BEHAVIOURS)
        content: str = _BEHAVIOURS[bucket]

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        return LLMResponse(
            content=content,
            model=self._model,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=0.0,
            deterministic=True,
        )
