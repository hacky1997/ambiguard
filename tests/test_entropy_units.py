"""Assert entropy is computed in NATS, never bits.

This test exists because using log2 instead of np.log produces working code
with plausible output that silently misroutes every query by a factor of
ln(2) ≈ 0.693. There is no error anywhere in the pipeline.

DO NOT DELETE THIS TEST. (AGENTS.md rule 1)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.gate.heuristic import HeuristicGate
from app.gate.thresholds import DEFAULT_THRESHOLDS


class TestEntropyUnits:
    """Every test here guards against the silent log2-vs-ln bug."""

    def test_entropy_is_nats_not_bits(self) -> None:
        """Gate entropy must match np.log (nats), not np.log2 (bits)."""
        gate = HeuristicGate()
        result = gate(
            "What are the side effects of it?",
            "Drug Alpha causes nausea. Drug Beta causes headaches and dizziness.",
        )

        p = np.array(result["center_distribution"])
        expected_nats = float(-(p * np.log(p)).sum())
        wrong_bits = float(-(p * np.log2(p)).sum())

        # Must match nats
        assert abs(result["entropy"] - expected_nats) < 1e-10, (
            f"Entropy {result['entropy']} != nats {expected_nats}. "
            f"Suspected log2 usage: bits would be {wrong_bits}"
        )

        # Must NOT match bits — they differ by factor ln(2) ≈ 0.693
        assert abs(result["entropy"] - wrong_bits) > 0.1, (
            f"Entropy {result['entropy']} ≈ bits {wrong_bits} — WRONG UNIT"
        )

    def test_nats_vs_bits_differ_by_ln2(self) -> None:
        """Verify the magnitude of the nats/bits discrepancy.

        The factor is 1/ln(2) ≈ 1.4427. This is large enough to cross
        the tau_ent = 1.51 threshold in the wrong direction for many
        real distributions.
        """
        p = np.array([0.25, 0.22, 0.20, 0.18, 0.15])
        nats = float(-(p * np.log(p)).sum())
        bits = float(-(p * np.log2(p)).sum())

        ratio = bits / nats
        assert abs(ratio - 1.0 / np.log(2)) < 0.01, (
            f"Ratio {ratio:.4f} doesn't match expected 1/ln(2) ≈ {1/np.log(2):.4f}"
        )

    def test_threshold_routing_with_nats(self) -> None:
        """tau_ent = 1.51 nats is calibrated for np.log, not np.log2.

        An ANSWER-like distribution should have entropy BELOW tau_ent in nats.
        With log2, the same distribution's entropy would be ~1.44× higher,
        potentially crossing the threshold and wrongly routing to CLARIFY.
        """
        # ANSWER distribution: dominant peak, low entropy
        p_answer = np.array([0.70, 0.10, 0.08, 0.07, 0.05])
        entropy_nats = float(-(p_answer * np.log(p_answer)).sum())
        entropy_bits = float(-(p_answer * np.log2(p_answer)).sum())

        # In nats: entropy should be well below tau_ent
        assert entropy_nats < DEFAULT_THRESHOLDS.tau_ent, (
            f"ANSWER entropy {entropy_nats:.3f} nats >= tau_ent {DEFAULT_THRESHOLDS.tau_ent}"
        )

        # Bits value would be higher — might wrongly cross threshold
        assert entropy_bits > entropy_nats, (
            "log2 entropy must be > log entropy (by factor 1/ln2)"
        )

    def test_tau_ent_value_is_nats(self) -> None:
        """tau_ent = 1.51 is in the nats range, not the bits range.

        For K=5 uniform: max entropy = ln(5) ≈ 1.609 nats = log2(5) ≈ 2.322 bits.
        tau_ent = 1.51 is near the nats maximum, confirming it's calibrated for nats.
        """
        max_entropy_nats = np.log(5)  # ≈ 1.609
        max_entropy_bits = np.log2(5)  # ≈ 2.322

        # 1.51 makes sense as a nats threshold (close to max)
        assert DEFAULT_THRESHOLDS.tau_ent < max_entropy_nats, (
            f"tau_ent {DEFAULT_THRESHOLDS.tau_ent} >= max nats {max_entropy_nats:.3f}"
        )

        # 1.51 would NOT make sense as a bits threshold (too far below max)
        assert DEFAULT_THRESHOLDS.tau_ent < max_entropy_bits, (
            "Sanity: tau_ent < max bits (both true, but ratio check matters)"
        )

        # The ratio of tau_ent to max entropy should be near 1 for nats
        nats_ratio = DEFAULT_THRESHOLDS.tau_ent / max_entropy_nats
        bits_ratio = DEFAULT_THRESHOLDS.tau_ent / max_entropy_bits
        assert nats_ratio > 0.9, (
            f"tau_ent/max_nats = {nats_ratio:.3f} — tau_ent appears calibrated for nats ✓"
        )
        assert bits_ratio < 0.7, (
            f"tau_ent/max_bits = {bits_ratio:.3f} — would be suspiciously low for bits"
        )
