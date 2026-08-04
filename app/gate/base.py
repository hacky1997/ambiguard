"""AmbiguityGate protocol — the contract every gate implementation must satisfy.

Two implementations exist:
  - CenterDistillGate: learned policy from the EAAAI 2026 paper
  - HeuristicGate: rule-based fallback, zero ML deps, always fallback_used=True

When fallback_used=True, downstream surfaces MUST label results accordingly —
never as CenterDistill results (AGENTS.md rule 4).
"""

from __future__ import annotations

from typing import Literal, Protocol
from typing_extensions import TypedDict


Behaviour = Literal["ANSWER", "CLARIFY", "ALTERNATIVES"]


class GateDecision(TypedDict):
    """Output of any AmbiguityGate implementation.

    All fields are mandatory. The thresholds dict is echoed for auditability
    so that every decision can be reproduced from logged state.
    """

    behaviour: Behaviour
    center_distribution: list[float]  # length K=5
    max_prob: float
    entropy: float  # NATS — -(p * np.log(p)).sum(), NEVER log2
    second_mass: float
    thresholds: dict[str, float]  # echoed for auditability
    latency_ms: float
    fallback_used: bool


class AmbiguityGate(Protocol):
    """Protocol for ambiguity gate implementations.

    The gate classifies a question-context pair into one of three routing
    behaviours BEFORE any LLM token is spent. Evaluation order is fixed:

        if max_prob > tau_conf:        ANSWER
        elif second_mass > tau_multi:  ALTERNATIVES
        elif entropy > tau_ent:        CLARIFY
        else:                          ANSWER  (conservative default)

    Do NOT reorder — it changes results (AGENTS.md rule 2).
    """

    def __call__(self, question: str, context: str) -> GateDecision:
        """Classify a question-context pair into a routing behaviour.

        Args:
            question: The user's question.
            context: Retrieved or provided context.

        Returns:
            GateDecision with routing behaviour and all diagnostic fields.
        """
        ...
