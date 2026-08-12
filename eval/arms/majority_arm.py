"""Majority-class comparison arm — always predicts the majority label.

Non-negotiable baseline (spec §3.1): MLQA routes 75.9% to CLARIFY, so
any comparison table without this arm is dishonest. If a model can't
beat "always predict CLARIFY", it has no value.
"""

from __future__ import annotations

import time
from collections import Counter

from eval.arms import ArmResult

# Default majority class from MLQA dataset distribution
_DEFAULT_MAJORITY: str = "CLARIFY"


class MajorityArm:
    """Always predicts the majority class.

    Deterministic, zero cost, <1ms latency. The floor every other arm
    must beat to justify its existence.
    """

    def __init__(
        self,
        majority_label: str = _DEFAULT_MAJORITY,
    ) -> None:
        self._majority: str = majority_label

    @classmethod
    def from_dataset(cls, gold_labels: list[str]) -> MajorityArm:
        """Infer the majority class from a dataset's gold labels."""
        if not gold_labels:
            return cls()
        counter = Counter(gold_labels)
        majority, _count = counter.most_common(1)[0]
        return cls(majority_label=majority)

    @property
    def name(self) -> str:
        return f"Majority class ({self._majority})"

    @property
    def deterministic(self) -> bool:
        return True

    def predict(self, question: str, context: str) -> ArmResult:
        """Always return the majority label."""
        start: float = time.perf_counter()
        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        return ArmResult(
            prediction=self._majority,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=0.0,
            metadata={"majority_label": self._majority},
        )
