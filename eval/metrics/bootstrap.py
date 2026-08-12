"""Bootstrap confidence intervals — 10,000 resamples, 95%.

Report intervals, never bare point estimates (spec §3.2).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    predictions: list[str],
    gold: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for any metric.

    Args:
        predictions: Predicted labels.
        gold: Gold-standard labels.
        metric_fn: A function (predictions, gold) → float.
        n_resamples: Number of bootstrap resamples (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, lower_bound, upper_bound).
    """
    if not predictions or not gold:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n: int = len(predictions)

    scores: np.ndarray = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices: np.ndarray = rng.integers(0, n, size=n)
        sampled_preds: list[str] = [predictions[int(j)] for j in indices]
        sampled_gold: list[str] = [gold[int(j)] for j in indices]
        scores[i] = metric_fn(sampled_preds, sampled_gold)

    alpha: float = 1.0 - confidence
    lower: float = round(float(np.percentile(scores, 100 * alpha / 2)), 3)
    upper: float = round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 3)
    point: float = round(metric_fn(predictions, gold), 3)

    return point, lower, upper
