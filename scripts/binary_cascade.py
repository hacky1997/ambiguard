#!/usr/bin/env python3
"""
Binary reframing + class-gated cascade.

WHY
---
The per-behaviour breakdown from cascade_analysis.py:

    ANSWER        n=300   gate 88.3%   judge 98.3%
    ALTERNATIVES  n=150   gate  6.0%   judge 62.0%
    CLARIFY       n=150   gate 18.0%   judge 30.0%

The gate is not a weak 3-class router. It is a competent BINARY detector that
finds unambiguous questions well and is effectively blind to the two ambiguity
classes. 6% on a 3-class problem is below chance, which means systematic
confusion, not noise.

Two consequences, both testable against cached predictions:

  H1  Collapse ALTERNATIVES+CLARIFY into AMBIGUOUS. If the gate scores ~85% on
      ANSWER-vs-AMBIGUOUS, it is a usable free ambiguity detector even though it
      cannot resolve the ambiguity TYPE.

  H2  Class-gated cascade. Confidence-gating failed (corr = +0.005), but the
      gate's PREDICTED CLASS is informative: when it says ANSWER it is right
      88.3% of the time. Route those directly; escalate everything else to the
      judge. Gold is ~50% ANSWER, so roughly half of traffic never touches the API.

No new inference. No API calls.

USAGE
    python scripts/binary_cascade.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JUDGE_COST_PER_1K = 1.83
GATE_P50_MS = 500.0
JUDGE_P50_MS = 751.0
AMBIG = ("ALTERNATIVES", "CLARIFY")


def boot_ci(correct: np.ndarray, seed: int = 42, n_boot: int = 10_000):
    rng = np.random.default_rng(seed)
    b = np.array([correct[rng.integers(0, len(correct), len(correct))].mean()
                  for _ in range(n_boot)])
    return float(correct.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def to_binary(labels: np.ndarray) -> np.ndarray:
    return np.where(np.isin(labels, AMBIG), "AMBIGUOUS", "ANSWER")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/comparison.json")
    ap.add_argument("--out", default="eval/results/binary_cascade.json")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    prd = data.get("per_row_details")
    if not prd:
        print("ERROR: no per_row_details in results file.", file=sys.stderr)
        return 1

    gate_key = next(k for k in prd if "centerdistill" in k.lower())
    judge_key = next(k for k in prd if "judge" in k.lower())

    gate_rows, judge_rows = prd[gate_key], prd[judge_key]
    n = len(gate_rows)
    assert len(judge_rows) == n, "arm lengths differ"

    gold = np.array([r["gold"] for r in gate_rows])
    gate_pred = np.array([r["prediction"] for r in gate_rows])
    judge_pred = np.array([r["prediction"] for r in judge_rows])

    gate_ok = gate_pred == gold
    judge_ok = judge_pred == gold

    print("=" * 76)
    print(f"BINARY REFRAMING + CLASS-GATED CASCADE — n={n}")
    print("=" * 76)
    print(f"  3-class baseline:  gate {gate_ok.mean():.1%}   judge {judge_ok.mean():.1%}")

    # ══════════════════════════════════════════════════════════════
    # H1 — binary: ANSWER vs AMBIGUOUS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-" * 76)
    print("H1  Binary task: ANSWER vs AMBIGUOUS (ALTERNATIVES + CLARIFY collapsed)")
    print("-" * 76)

    gold_b = to_binary(gold)
    gate_b = to_binary(gate_pred)
    judge_b = to_binary(judge_pred)

    gate_b_ok = gate_b == gold_b
    judge_b_ok = judge_b == gold_b

    g_acc, g_lo, g_hi = boot_ci(gate_b_ok)
    j_acc, j_lo, j_hi = boot_ci(judge_b_ok)

    # Majority baseline on the binary task
    vals, counts = np.unique(gold_b, return_counts=True)
    maj_label = vals[counts.argmax()]
    maj_acc = counts.max() / n

    print(f"{'System':<28} {'Accuracy':>10} {'CI95':>18}")
    print(f"{'Majority (' + maj_label + ')':<28} {maj_acc:>9.1%} {'—':>18}")
    print(f"{'CenterDistill (free)':<28} {g_acc:>9.1%} {f'[{g_lo:.1%}, {g_hi:.1%}]':>18}")
    print(f"{'LLM judge ($1.83/1k)':<28} {j_acc:>9.1%} {f'[{j_lo:.1%}, {j_hi:.1%}]':>18}")

    # Confusion detail on the binary task
    tp = int(((gate_b == "ANSWER") & (gold_b == "ANSWER")).sum())
    fp = int(((gate_b == "ANSWER") & (gold_b == "AMBIGUOUS")).sum())
    fn = int(((gate_b == "AMBIGUOUS") & (gold_b == "ANSWER")).sum())
    tn = int(((gate_b == "AMBIGUOUS") & (gold_b == "AMBIGUOUS")).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\n  Gate on ANSWER class:  precision {prec:.1%}  recall {rec:.1%}  F1 {f1:.1%}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("    FP = called ANSWER but actually ambiguous — the costly error:")
    print("         the system answers confidently when it should have asked.")

    gap_b = j_acc - g_acc
    gap_3 = judge_ok.mean() - gate_ok.mean()
    print(f"\n  Judge advantage: {gap_3:+.1%} on 3-class  ->  {gap_b:+.1%} on binary")
    if gap_b < gap_3 - 0.05:
        print("  ✅ The gap narrows substantially. The gate's weakness is ambiguity TYPING,")
        print("     not ambiguity DETECTION.")
    else:
        print("  ❌ The gap persists. The gate is weak at detection too.")

    # ══════════════════════════════════════════════════════════════
    # H2 — class-gated cascade
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-" * 76)
    print("H2  Class-gated cascade: trust the gate's ANSWER calls, escalate the rest")
    print("-" * 76)

    says_answer = gate_pred == "ANSWER"
    kept = says_answer.mean()

    # Precision of the gate when it says ANSWER — this is what makes or breaks it.
    prec_answer = gate_ok[says_answer].mean() if says_answer.any() else 0.0
    print(f"  Gate predicts ANSWER on {kept:.1%} of queries")
    print(f"  Precision when it does: {prec_answer:.1%}")
    print(f"  (judge accuracy on those same rows: {judge_ok[says_answer].mean():.1%})")

    # Cascade: gate handles its ANSWER calls, judge handles everything else.
    cascade_ok = np.where(says_answer, gate_ok, judge_ok)
    c_acc, c_lo, c_hi = boot_ci(cascade_ok)

    escalated = 1.0 - kept
    cascade_cost = escalated * JUDGE_COST_PER_1K
    cascade_lat = GATE_P50_MS + escalated * JUDGE_P50_MS
    saving = (1 - cascade_cost / JUDGE_COST_PER_1K) * 100

    print(f"\n{'System':<28} {'Accuracy':>10} {'CI95':>18} {'$/1k':>8} {'p50 ms':>9}")
    print(f"{'Gate only':<28} {gate_ok.mean():>9.1%} {'—':>18} "
          f"{0.0:>8.2f} {GATE_P50_MS:>9.0f}")
    print(f"{'Judge only':<28} {judge_ok.mean():>9.1%} {'—':>18} "
          f"{JUDGE_COST_PER_1K:>8.2f} {JUDGE_P50_MS:>9.0f}")
    print(f"{'Class-gated cascade':<28} {c_acc:>9.1%} {f'[{c_lo:.1%}, {c_hi:.1%}]':>18} "
          f"{cascade_cost:>8.2f} {cascade_lat:>9.0f}")

    delta = c_acc - judge_ok.mean()
    print(f"\n  vs judge-only: {delta:+.1%} accuracy, {saving:.0f}% cheaper, "
          f"{cascade_lat - JUDGE_P50_MS:+.0f} ms")

    if delta >= -0.02 and saving >= 30:
        print("  ✅ VIABLE. Near-parity accuracy at materially lower cost.")
    elif delta >= -0.05 and saving >= 40:
        print("  ⚠ MARGINAL. Real savings, real accuracy cost. Defensible if stated.")
    else:
        print("  ❌ NOT VIABLE. The accuracy loss is not worth the saving.")

    # Same idea on the binary task, where the gate is strongest.
    cascade_b_ok = np.where(says_answer, gate_b_ok, judge_b_ok)
    cb_acc, cb_lo, cb_hi = boot_ci(cascade_b_ok)
    print(f"\n  Binary-task cascade: {cb_acc:.1%} [{cb_lo:.1%}, {cb_hi:.1%}]  "
          f"vs judge-only {j_acc:.1%}  ({cb_acc - j_acc:+.1%})")

    # ══════════════════════════════════════════════════════════════
    # Where the gate's ANSWER calls go wrong
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-" * 76)
    print("Error profile of the gate's ANSWER calls")
    print("-" * 76)
    wrong = says_answer & ~gate_ok
    if wrong.any():
        vals, counts = np.unique(gold[wrong], return_counts=True)
        for v, c in zip(vals, counts):
            print(f"  said ANSWER, actually {v:<14} {c:>4}  "
                  f"({c / says_answer.sum():.1%} of its ANSWER calls)")
    else:
        print("  none")

    out = {
        "n": n,
        "three_class": {
            "gate": round(float(gate_ok.mean()), 4),
            "judge": round(float(judge_ok.mean()), 4),
        },
        "binary": {
            "task": "ANSWER vs AMBIGUOUS (ALTERNATIVES+CLARIFY collapsed)",
            "majority_label": str(maj_label),
            "majority_acc": round(float(maj_acc), 4),
            "gate": {"acc": round(g_acc, 4), "ci95": [round(g_lo, 4), round(g_hi, 4)]},
            "judge": {"acc": round(j_acc, 4), "ci95": [round(j_lo, 4), round(j_hi, 4)]},
            "gate_answer_precision": round(float(prec), 4),
            "gate_answer_recall": round(float(rec), 4),
            "gate_answer_f1": round(float(f1), 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "gap_narrowed": bool(gap_b < gap_3 - 0.05),
        },
        "class_gated_cascade": {
            "kept_frac": round(float(kept), 4),
            "gate_precision_on_kept": round(float(prec_answer), 4),
            "accuracy": round(c_acc, 4),
            "ci95": [round(c_lo, 4), round(c_hi, 4)],
            "cost_per_1k": round(float(cascade_cost), 3),
            "p50_ms": round(float(cascade_lat), 1),
            "cost_saving_pct": round(float(saving), 1),
            "acc_vs_judge": round(float(delta), 4),
            "binary_accuracy": round(cb_acc, 4),
            "binary_acc_vs_judge": round(float(cb_acc - j_acc), 4),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
