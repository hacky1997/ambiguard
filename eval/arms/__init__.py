"""Comparison arm base types."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class ArmResult(TypedDict):
    """Standard output from a comparison arm's predict() method."""

    prediction: str  # Behaviour label: ANSWER, CLARIFY, or ALTERNATIVES
    latency_ms: float
    cost_usd: float
    metadata: dict[str, Any]  # arm-specific diagnostic info


class EvalArm(Protocol):
    """Protocol for comparison arms (spec §3.1)."""

    @property
    def name(self) -> str:
        """Human-readable arm identifier for the comparison table."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether identical inputs produce byte-identical outputs."""
        ...

    def predict(self, question: str, context: str) -> ArmResult:
        """Predict a routing behaviour for a question-context pair."""
        ...
