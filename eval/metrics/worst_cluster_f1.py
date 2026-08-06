"""Worst-cluster F1 — minimum per-class performance.

WHY THIS FILE WAS REWRITTEN
---------------------------
The previous implementation reported 6.7 for a majority-class predictor that
never emits AMBIGUOUS. That is arithmetically impossible: a class the model
never predicts has recall 0, so the minimum across classes must be 0.

Two defects produced it:

  1. Iterating over PREDICTED classes rather than GOLD classes. A class the
     model never predicts simply vanished from the minimum instead of scoring 0.
  2. Assuming K=5 clusters from the original CenterDistill design, so the
     metric never adapted when the task became binary.

This metric exists to catch degenerate predictors. When it fails to do that it
is worse than useless — it launders a constant predictor as competitive, which
is the exact failure mode the comparison table is meant to expose.

DEFINITION
----------
For every class present in the GOLD labels, compute F1. Report the minimum,
scaled x10 to match the reporting convention in the CenterDistill paper.

    F1_c = 2 * P_c * R_c / (P_c + R_c)
    worst = 10 * min over c in gold_classes of F1_c

A class with no predictions has precision 0 by convention (0/0 -> 0), so its
F1 is 0 and the minimum is 0. That is the intended behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence


def per_class_prf(
    predictions: Sequence[str],
    gold: Sequence[str],
    label: str,
) -> tuple[float, float, float]:
    """Precision, recall, F1 for one class.

    Convention: 0/0 -> 0.0. A class that is never predicted scores 0 precision,
    and therefore 0 F1. This is what makes the metric able to detect collapse.
    """
    tp = sum(1 for p, g in zip(predictions, gold, strict=True) if p == label and g == label)
    fp = sum(1 for p, g in zip(predictions, gold, strict=True) if p == label and g != label)
    fn = sum(1 for p, g in zip(predictions, gold, strict=True) if p != label and g == label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def per_class_breakdown(
    predictions: Sequence[str],
    gold: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Full per-class table, keyed by GOLD class.

    Classes are taken from the gold labels, never from the predictions — a class
    the model ignores must still appear, scoring 0.
    """
    if len(predictions) != len(gold):
        raise ValueError(f"length mismatch: {len(predictions)} vs {len(gold)}")

    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(gold)):
        p, r, f1 = per_class_prf(predictions, gold, label)
        out[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }
    return out


def worst_cluster_f1(
    predictions: Sequence[str],
    gold: Sequence[str],
    scale: float = 10.0,
) -> float:
    """Minimum per-class F1 across all GOLD classes, scaled by `scale`.

    Returns 0.0 when any gold class is never predicted correctly — including the
    degenerate case of a constant predictor.
    """
    if not gold:
        raise ValueError("gold labels are empty")

    breakdown = per_class_breakdown(predictions, gold)
    return round(scale * min(v["f1"] for v in breakdown.values()), 2)


def macro_f1(predictions: Sequence[str], gold: Sequence[str]) -> float:
    """Mean per-class F1 over gold classes. Reported alongside the minimum."""
    breakdown = per_class_breakdown(predictions, gold)
    return round(sum(v["f1"] for v in breakdown.values()) / len(breakdown), 4)


# ── Self-check ────────────────────────────────────────────────────
# These are the cases the old implementation got wrong. Run this module
# directly to verify the fix before trusting any table that uses it.
if __name__ == "__main__":
    gold_bin = ["ANSWER"] * 300 + ["AMBIGUOUS"] * 300

    # 1. Constant predictor. MUST be 0.0 — this is the bug that motivated the rewrite.
    const = ["ANSWER"] * 600
    wc = worst_cluster_f1(const, gold_bin)
    print(f"constant ANSWER predictor : worst={wc}  macro={macro_f1(const, gold_bin)}")
    assert wc == 0.0, f"REGRESSION: constant predictor scored {wc}, expected 0.0"

    # 2. Perfect predictor -> 10.0
    wc = worst_cluster_f1(list(gold_bin), gold_bin)
    print(f"perfect predictor         : worst={wc}")
    assert wc == 10.0, f"perfect predictor scored {wc}"

    # 3. Skewed but not constant: catches most ANSWER, few AMBIGUOUS.
    skewed = ["ANSWER"] * 300 + ["ANSWER"] * 240 + ["AMBIGUOUS"] * 60
    print(f"skewed predictor          : worst={worst_cluster_f1(skewed, gold_bin)}  "
          f"macro={macro_f1(skewed, gold_bin)}")
    print(f"  breakdown: {per_class_breakdown(skewed, gold_bin)}")

    # 4. Three-class still works (the original CenterDistill setting).
    gold3 = ["ANSWER"] * 100 + ["CLARIFY"] * 100 + ["ALTERNATIVES"] * 100
    never_alt = ["ANSWER"] * 100 + ["CLARIFY"] * 100 + ["CLARIFY"] * 100
    wc = worst_cluster_f1(never_alt, gold3)
    print(f"3-class, never ALTERNATIVES: worst={wc}")
    assert wc == 0.0, f"REGRESSION: unpredicted class scored {wc}, expected 0.0"

    print("\nAll self-checks passed.")
