"""Balanced accuracy — mean recall across unique gold classes.

Useful for imbalanced multiclass evaluation.
"""

from __future__ import annotations


def balanced_accuracy(predictions: list[str], gold: list[str]) -> float:
    """Compute balanced accuracy (macro-averaged recall across classes).

    Args:
        predictions: List of predicted behaviour labels.
        gold: List of gold-standard behaviour labels.

    Returns:
        Macro-averaged recall in [0.0, 1.0].
    """
    if not predictions or not gold:
        return 0.0
    if len(predictions) != len(gold):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold")

    unique_classes = sorted(set(gold))
    if not unique_classes:
        return 0.0

    recalls: list[float] = []
    for cls in unique_classes:
        cls_gold_indices = [i for i, g in enumerate(gold) if g == cls]
        if not cls_gold_indices:
            continue
        cls_correct = sum(1 for i in cls_gold_indices if predictions[i] == gold[i])
        recalls.append(cls_correct / len(cls_gold_indices))

    return sum(recalls) / len(recalls) if recalls else 0.0
