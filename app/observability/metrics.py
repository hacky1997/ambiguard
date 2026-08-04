"""Observability metrics for AmbiGuard.

Prometheus counters, histograms, and custom metrics:
  - gate_latency_ms (histogram)
  - behaviour_total (counter)
  - verification_rejection_rate (gauge)
  - clarify_resolution_rate (headline product metric — fraction of CLARIFY threads reaching a confident answer on resume)
  - fallback_gate_used_total (counter)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Metrics collector for Prometheus / operational instrumentation."""

    def __init__(self) -> None:
        self.behaviour_counts: dict[str, int] = {
            "ANSWER": 0,
            "CLARIFY": 0,
            "ALTERNATIVES": 0,
        }
        self.fallback_used_count: int = 0
        self.verification_total: int = 0
        self.verification_rejections: int = 0
        self.clarify_threads_total: int = 0
        self.clarify_resolved_total: int = 0
        self.latency_samples: list[float] = []

    def record_gate_decision(self, behaviour: str, latency_ms: float, fallback_used: bool) -> None:
        """Record a gate routing decision."""
        if behaviour in self.behaviour_counts:
            self.behaviour_counts[behaviour] += 1
        if fallback_used:
            self.fallback_used_count += 1
        self.latency_samples.append(latency_ms)

    def record_verification(self, passed: bool) -> None:
        """Record verification outcome."""
        self.verification_total += 1
        if not passed:
            self.verification_rejections += 1

    def record_clarify_resume(self, resolved_to_answer: bool) -> None:
        """Record clarify thread resume outcome."""
        self.clarify_threads_total += 1
        if resolved_to_answer:
            self.clarify_resolved_total += 1

    @property
    def verification_rejection_rate(self) -> float:
        """Compute verification rejection rate."""
        if not self.verification_total:
            return 0.0
        return round(self.verification_rejections / self.verification_total, 3)

    @property
    def clarify_resolution_rate(self) -> float:
        """Headline product metric: fraction of CLARIFY threads reaching confident ANSWER on resume."""
        if not self.clarify_threads_total:
            return 0.0
        return round(self.clarify_resolved_total / self.clarify_threads_total, 3)


# Global metrics instance
metrics = MetricsRegistry()
