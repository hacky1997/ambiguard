"""Worst-cluster F1 — robustness lower bound ported from the paper.

Computes per-class F1 for each behaviour label, then reports the
minimum (scaled to 0–10). A model that aces the majority class but
fails minorities will score poorly here — that's the point.

See spec §3.2, §7.
"""

from __future__ import annotations

_BEHAVIOURS: list[str] = ["ANSWER", "CLARIFY", "ALTERNATIVES"]


def _per_class_f1(
    predictions: list[str], gold: list[str], label: str
) -> float:
    """Compute F1 score for a single class label."""
    tp: int = sum(
        1 for p, g in zip(predictions, gold, strict=True)
        if p == label and g == label
    )
    fp: int = sum(
        1 for p, g in zip(predictions, gold, strict=True)
        if p == label and g != label
    )
    fn: int = sum(
        1 for p, g in zip(predictions, gold, strict=True)
        if p != label and g == label
    )

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def worst_cluster_f1(
    predictions: list[str],
    gold: list[str],
    labels: list[str] | None = None,
) -> float:
    """Compute worst-case F1 across behaviour classes, scaled to 0–10.

    Args:
        predictions: Predicted behaviour labels.
        gold: Gold-standard behaviour labels.
        labels: Classes to evaluate. Defaults to all three behaviours.

    Returns:
        min(per-class F1) × 10, rounded to one decimal. Range [0.0, 10.0].
    """
    if not predictions or not gold:
        return 0.0
    if len(predictions) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold"
        )

    eval_labels: list[str] = labels if labels is not None else _BEHAVIOURS

    # Only evaluate labels that appear in gold — avoids penalising for
    # classes that were never expected
    present_labels: list[str] = [lb for lb in eval_labels if lb in set(gold)]
    if not present_labels:
        return 0.0

    min_f1: float = min(
        _per_class_f1(predictions, gold, lb) for lb in present_labels
    )
    return round(min_f1 * 10, 1)
