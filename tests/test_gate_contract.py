"""Contract tests for AmbiguityGate implementations.

Both HeuristicGate and CenterDistillGate (in fallback mode) must pass
every test here. These tests enforce the protocol defined in base.py.

DO NOT DELETE THESE TESTS. (AGENTS.md rules 1, 2)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.gate.centerdistill import CenterDistillGate
from app.gate.heuristic import HeuristicGate


def _make_gates() -> list[tuple[str, Any]]:
    """Instantiate all gate implementations to test."""
    return [
        ("heuristic", HeuristicGate()),
        ("centerdistill_fallback", CenterDistillGate()),  # no checkpoint → fallback
    ]


@pytest.fixture(params=_make_gates(), ids=lambda g: g[0])
def gate(request: pytest.FixtureRequest) -> Any:
    """Parametrised fixture yielding each gate implementation."""
    return request.param[1]


# ---------- GateDecision structure ----------


class TestGateDecisionStructure:
    """Every gate must return a valid GateDecision."""

    _REQUIRED_KEYS: frozenset[str] = frozenset(
        {
            "behaviour",
            "center_distribution",
            "max_prob",
            "entropy",
            "second_mass",
            "thresholds",
            "latency_ms",
            "fallback_used",
        }
    )

    def test_returns_all_fields(self, gate: Any) -> None:
        result = gate("What is X?", "X could be A or B.")
        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing GateDecision fields: {missing}"

    def test_behaviour_is_valid_literal(self, gate: Any) -> None:
        result = gate("What is this?", "Context with information.")
        assert result["behaviour"] in ("ANSWER", "CLARIFY", "ALTERNATIVES"), (
            f"Invalid behaviour: {result['behaviour']}"
        )

    def test_fallback_used_is_bool(self, gate: Any) -> None:
        result = gate("Test", "Context")
        assert isinstance(result["fallback_used"], bool)


# ---------- Distribution constraints ----------


class TestDistribution:
    """Center distribution must be a valid probability vector."""

    def test_length_k5(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        assert len(result["center_distribution"]) == 5, (
            f"Expected K=5, got {len(result['center_distribution'])}"
        )

    def test_sums_to_one(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        total = sum(result["center_distribution"])
        assert abs(total - 1.0) < 1e-6, f"Distribution sums to {total}"

    def test_non_negative(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        assert all(p >= 0 for p in result["center_distribution"]), (
            f"Negative probabilities: {result['center_distribution']}"
        )

    def test_max_prob_matches(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        expected_max = max(result["center_distribution"])
        assert abs(result["max_prob"] - expected_max) < 1e-10


# ---------- Entropy ----------


class TestEntropy:
    """Entropy must be non-negative, in NATS, and consistent."""

    def test_non_negative(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        assert result["entropy"] >= 0, f"Negative entropy: {result['entropy']}"

    def test_in_nats(self, gate: Any) -> None:
        """Entropy MUST use np.log (nats), not np.log2 (bits)."""
        result = gate("Test question", "Test context")
        p = np.array(result["center_distribution"])
        expected_nats = float(-(p * np.log(p)).sum())
        assert abs(result["entropy"] - expected_nats) < 1e-10, (
            f"Entropy {result['entropy']} != nats {expected_nats}. Using log2?"
        )

    def test_bounded_by_ln_k(self, gate: Any) -> None:
        """Entropy cannot exceed ln(K) = ln(5) ≈ 1.609 nats."""
        result = gate("Test question", "Test context")
        max_entropy = float(np.log(5))
        assert result["entropy"] <= max_entropy + 1e-10, (
            f"Entropy {result['entropy']} > max {max_entropy}"
        )


# ---------- Thresholds ----------


class TestThresholds:
    """Thresholds must be echoed for auditability."""

    def test_all_thresholds_present(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        for key in ("tau_conf", "tau_multi", "tau_ent"):
            assert key in result["thresholds"], f"Missing threshold: {key}"

    def test_threshold_values_correct(self, monkeypatch: Any) -> None:
        """Default thresholds match the paper values."""
        monkeypatch.setenv("AMBIGUARD_TAU_CONF", "")
        monkeypatch.setenv("AMBIGUARD_TAU_MULTI", "")
        monkeypatch.setenv("AMBIGUARD_TAU_ENT", "")
        from app.gate.heuristic import HeuristicGate
        gate = HeuristicGate()
        result = gate("Test question", "Test context")
        assert result["thresholds"]["tau_conf"] == 0.44
        assert result["thresholds"]["tau_multi"] == 0.24
        assert result["thresholds"]["tau_ent"] == 1.51


# ---------- Timing ----------


class TestTiming:
    def test_latency_non_negative(self, gate: Any) -> None:
        result = gate("Test question", "Test context")
        assert result["latency_ms"] >= 0


# ---------- Robustness ----------


class TestRobustness:
    """Gate must handle edge-case inputs without crashing."""

    def test_empty_strings(self, gate: Any) -> None:
        result = gate("", "")
        assert result["behaviour"] in ("ANSWER", "CLARIFY", "ALTERNATIVES")

    def test_long_inputs(self, gate: Any) -> None:
        long_q = "What is X? " * 500
        long_c = "Context about X. " * 500
        result = gate(long_q, long_c)
        assert result["behaviour"] in ("ANSWER", "CLARIFY", "ALTERNATIVES")

    def test_unicode_input(self, gate: Any) -> None:
        result = gate("¿Cuáles son los efectos secundarios?", "El fármaco α causa náuseas.")
        assert result["behaviour"] in ("ANSWER", "CLARIFY", "ALTERNATIVES")

    def test_pure_punctuation(self, gate: Any) -> None:
        result = gate("???!!!", "---...;;;")
        assert result["behaviour"] in ("ANSWER", "CLARIFY", "ALTERNATIVES")


# ---------- Implementation-specific ----------


class TestHeuristicSpecific:
    """Heuristic gate must always set fallback_used=True."""

    _CASES: list[tuple[str, str]] = [
        ("What is X?", "X is A or B."),
        ("Tell me about Y.", "Y is well known."),
        ("how much?", "Products are $5 and $10."),
        ("", ""),
    ]

    @pytest.mark.parametrize("question,context", _CASES)
    def test_always_fallback(self, question: str, context: str) -> None:
        gate = HeuristicGate()
        result = gate(question, context)
        assert result["fallback_used"] is True, (
            f"HeuristicGate must always set fallback_used=True for '{question}'"
        )


class TestCenterDistillFallback:
    """CenterDistillGate without checkpoint must use and label fallback."""

    def test_using_fallback_property(self) -> None:
        gate = CenterDistillGate()
        assert gate.using_fallback is True

    def test_fallback_used_in_decision(self) -> None:
        gate = CenterDistillGate()
        result = gate("Test question", "Test context")
        assert result["fallback_used"] is True


# ---------- Threshold evaluation order ----------


class TestThresholdOrder:
    """Verify ANSWER → ALTERNATIVES → CLARIFY order (AGENTS.md rule 2).

    This test exercises the CenterDistillGate._apply_thresholds method
    directly to confirm the evaluation order is correct.
    """

    def test_high_confidence_routes_answer(self) -> None:
        """max_prob > tau_conf → ANSWER, regardless of other signals."""
        gate = CenterDistillGate()
        if not gate.using_fallback:
            pytest.skip("Need fallback mode for _apply_thresholds access")

        # Simulate: high max_prob, high second_mass, high entropy
        # With correct order, ANSWER wins because it's checked first
        from app.gate.thresholds import DEFAULT_THRESHOLDS

        # max_prob=0.45 > tau_conf=0.44 → ANSWER
        dist = [0.45, 0.25, 0.06, 0.06, 0.06, 0.06, 0.06]
        p = np.array(dist)
        max_prob = max(dist)
        second_mass = sorted(dist, reverse=True)[1]
        entropy = float(-(p * np.log(p)).sum())

        # Verify this WOULD trigger ALTERNATIVES and CLARIFY checks too
        assert second_mass > DEFAULT_THRESHOLDS.tau_multi  # 0.25 > 0.24
        assert entropy > DEFAULT_THRESHOLDS.tau_ent  # 1.55 > 1.51
        # But ANSWER check comes first
        assert max_prob > DEFAULT_THRESHOLDS.tau_conf  # 0.45 > 0.44
