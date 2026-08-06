"""CenterDistill comparison arm — the learned gate (or heuristic fallback).

When fallback_used=True in results, downstream surfaces MUST label it
accordingly — never as CenterDistill results (AGENTS.md rule 4).
"""

from __future__ import annotations

from typing import Any

from app.gate.centerdistill import CenterDistillGate
from app.gate.thresholds import GateThresholds
from app.settings import get_settings
from eval.arms import ArmResult


class CenterDistillArm:
    """Comparison arm using the CenterDistill gate.

    Loads the checkpoint from settings. If unavailable, falls back to
    the heuristic gate — but the ArmResult metadata will include
    fallback_used=True so the comparison table labels it correctly.
    """

    def __init__(self, thresholds: GateThresholds | None = None, name_suffix: str = "") -> None:
        settings = get_settings()
        self._gate = CenterDistillGate(
            checkpoint_path=settings.gate_checkpoint_path,
            hf_repo=settings.gate_hf_repo,
            thresholds=thresholds,
        )
        self._suffix = name_suffix

    @property
    def name(self) -> str:
        if self._gate.using_fallback:
            return "CenterDistill (heuristic fallback)"
        if self._suffix:
            return f"CenterDistill ({self._suffix})"
        return "CenterDistill"

    @property
    def deterministic(self) -> bool:
        return True

    def predict(self, question: str, context: str) -> ArmResult:
        """Run the gate and wrap the result."""
        decision = self._gate(question, context)
        return ArmResult(
            prediction=decision["behaviour"],
            latency_ms=decision["latency_ms"],
            cost_usd=0.0,  # gate is ~free — one linear layer on CLS
            metadata={
                "center_distribution": decision["center_distribution"],
                "max_prob": decision["max_prob"],
                "entropy": decision["entropy"],
                "second_mass": decision["second_mass"],
                "fallback_used": decision["fallback_used"],
            },
        )
