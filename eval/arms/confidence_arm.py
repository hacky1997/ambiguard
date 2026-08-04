"""Confidence-threshold comparison arm — the baseline the paper beats.

Uses span-softmax max probability as a proxy for answer confidence.
If max_prob > threshold → ANSWER, else → CLARIFY. This arm can only
produce binary outcomes — it cannot distinguish ALTERNATIVES from
CLARIFY, which is one of the ways CenterDistill outperforms it.

When no CenterDistill model is loaded, uses the heuristic gate's
synthetic distribution as a confidence proxy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.gate.centerdistill import CenterDistillGate
from app.gate.thresholds import DEFAULT_THRESHOLDS, GateThresholds
from app.settings import get_settings
from eval.arms import ArmResult

# DECISION: Default confidence threshold is tau_conf from the paper.
# The confidence arm uses only this single threshold — it cannot
# distinguish ALTERNATIVES from CLARIFY, which is the weakness
# CenterDistill addresses with its three-threshold system.
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.44


class ConfidenceArm:
    """Confidence-threshold baseline arm.

    Binary decision: high confidence → ANSWER, low → CLARIFY.
    Cannot predict ALTERNATIVES — this structural limitation is
    the key argument for CenterDistill's multi-threshold approach.

    Deterministic, zero cost.
    """

    def __init__(
        self,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        thresholds: GateThresholds | None = None,
    ) -> None:
        self._conf_threshold: float = confidence_threshold
        settings = get_settings()
        self._gate = CenterDistillGate(
            checkpoint_path=settings.gate_checkpoint_path,
            hf_repo=settings.gate_hf_repo,
            thresholds=thresholds,
        )

    @property
    def name(self) -> str:
        return f"Confidence threshold (τ={self._conf_threshold})"

    @property
    def deterministic(self) -> bool:
        return True

    def predict(self, question: str, context: str) -> ArmResult:
        """Binary classification: confident → ANSWER, else → CLARIFY."""
        start: float = time.perf_counter()

        decision = self._gate(question, context)
        max_prob: float = decision["max_prob"]

        # Binary decision — the key limitation vs CenterDistill
        prediction: str = "ANSWER" if max_prob > self._conf_threshold else "CLARIFY"

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        return ArmResult(
            prediction=prediction,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=0.0,
            metadata={
                "max_prob": max_prob,
                "confidence_threshold": self._conf_threshold,
                "fallback_used": decision["fallback_used"],
                "original_behaviour": decision["behaviour"],
            },
        )
