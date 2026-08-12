"""Behaviour accuracy — exact match vs gold routing labels.

Ported from the CenterDistill paper (spec §3.2, §7).
"""

from __future__ import annotations


def behaviour_accuracy(predictions: list[str], gold: list[str]) -> float:
    """Compute exact-match accuracy between predicted and gold behaviours.

    Args:
        predictions: List of predicted behaviour labels (ANSWER/CLARIFY/ALTERNATIVES).
        gold: List of gold-standard behaviour labels.

    Returns:
        Fraction of correct predictions in [0.0, 1.0].
    """
    if not predictions or not gold:
        return 0.0
    if len(predictions) != len(gold):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold")
    matches: int = sum(1 for p, g in zip(predictions, gold, strict=True) if p == g)
    return matches / len(gold)
