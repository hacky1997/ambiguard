#!/usr/bin/env python3
"""
Ablation experiment: Removing the second_mass threshold branch.

HYPOTHESIS
----------
`second_mass` was shown to be anti-correlated with true ambiguity (AUC = 0.425,
paired delta CI [-0.163, -0.031] below 0).
The production rule evaluates:
    1. max_prob > tau_conf       → ANSWER
    2. second_mass > tau_multi   → ALTERNATIVES
    3. entropy > tau_ent         → CLARIFY
    4. else                      → CLARIFY

Bypassing step 2 (the ALTERNATIVES branch) removes the anti-correlated second_mass
statistic. We evaluate behaviour accuracy and macro-F1 on golden_gate.jsonl (n=600)
and golden_gate_repaired.jsonl (n=600).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.gate.centerdistill import CenterDistillGate
from app.settings import get_settings

logger = logging.getLogger(__name__)

_GOLDEN_ORIG = Path("eval/datasets/golden_gate.jsonl")
_GOLDEN_REPAIRED = Path("eval/datasets/golden_gate_repaired.jsonl")
_OUT = Path("eval/results/ablation_no_second_mass.json")
_SEED = 42


def boot_ci(correct: np.ndarray, seed: int = 42, n_boot: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = np.array([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)])
    return float(correct.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def route_standard(max_prob: float, second_m: float, ent: float, tc: float, tm: float, te: float) -> str:
    if max_prob > tc:
        return "ANSWER"
    elif second_m > tm:
        return "ALTERNATIVES"
    elif ent > te:
        return "CLARIFY"
    else:
        return "CLARIFY"


def route_no_second_mass(max_prob: float, second_m: float, ent: float, tc: float, tm: float, te: float) -> str:
    if max_prob > tc:
        return "ANSWER"
    elif ent > te:
        return "CLARIFY"
    else:
        return "CLARIFY"


def evaluate_dataset(dataset_path: Path, gate: CenterDistillGate) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    logger.info("Running forward passes for %d rows from %s...", len(rows), dataset_path.name)
    extracted = []
    for idx, row in enumerate(rows):
        dec = gate.decide(row["question"], row.get("context", ""))
        extracted.append({
            "gold": row["expected_behaviour"],
            "max_prob": float(dec["max_prob"]),
            "second_mass": float(dec["second_mass"]),
            "entropy": float(dec["entropy"]),
            "thresholds": dec["thresholds"]
        })

    tc = gate._thresholds.tau_conf
    tm = gate._thresholds.tau_multi
    te = gate._thresholds.tau_ent

    golds = np.array([item["gold"] for item in extracted])
    std_preds = np.array([route_standard(item["max_prob"], item["second_mass"], item["entropy"], tc, tm, te) for item in extracted])
    ablated_preds = np.array([route_no_second_mass(item["max_prob"], item["second_mass"], item["entropy"], tc, tm, te) for item in extracted])

    std_binary_preds = np.where(std_preds == "ANSWER", "ANSWER", "AMBIGUOUS")
    abl_binary_preds = np.where(ablated_preds == "ANSWER", "ANSWER", "AMBIGUOUS")
    gold_binary = np.where(golds == "ANSWER", "ANSWER", "AMBIGUOUS")

    std_3class_ok = std_preds == golds
    abl_3class_ok = ablated_preds == golds

    std_bin_ok = std_binary_preds == gold_binary
    abl_bin_ok = abl_binary_preds == gold_binary

    std_acc, std_lo, std_hi = boot_ci(std_3class_ok)
    abl_acc, abl_lo, abl_hi = boot_ci(abl_3class_ok)

    std_bacc, std_blo, std_bhi = boot_ci(std_bin_ok)
    abl_bacc, abl_blo, abl_bhi = boot_ci(abl_bin_ok)

    # Paired delta bootstrap for 3-class accuracy
    rng = np.random.default_rng(_SEED)
    deltas = []
    for _ in range(10_000):
        idx = rng.integers(0, len(rows), len(rows))
        deltas.append(abl_3class_ok[idx].mean() - std_3class_ok[idx].mean())
    d_mean = float(np.mean(deltas))
    d_lo = float(np.percentile(deltas, 2.5))
    d_hi = float(np.percentile(deltas, 97.5))

    return {
        "dataset": dataset_path.name,
        "n": len(rows),
        "thresholds_used": {"tau_conf": tc, "tau_multi": tm, "tau_ent": te},
        "standard_policy": {
            "accuracy_3class": round(std_acc, 4),
            "ci95_3class": [round(std_lo, 4), round(std_hi, 4)],
            "accuracy_binary": round(std_bacc, 4),
            "ci95_binary": [round(std_blo, 4), round(std_bhi, 4)],
            "confusion_3class": {
                "said_ANSWER": int((std_preds == "ANSWER").sum()),
                "said_ALTERNATIVES": int((std_preds == "ALTERNATIVES").sum()),
                "said_CLARIFY": int((std_preds == "CLARIFY").sum()),
            }
        },
        "ablated_policy_no_second_mass": {
            "accuracy_3class": round(abl_acc, 4),
            "ci95_3class": [round(abl_lo, 4), round(abl_hi, 4)],
            "accuracy_binary": round(abl_bacc, 4),
            "ci95_binary": [round(abl_blo, 4), round(abl_bhi, 4)],
            "confusion_3class": {
                "said_ANSWER": int((ablated_preds == "ANSWER").sum()),
                "said_ALTERNATIVES": int((ablated_preds == "ALTERNATIVES").sum()),
                "said_CLARIFY": int((ablated_preds == "CLARIFY").sum()),
            }
        },
        "paired_delta_3class": {
            "mean": round(d_mean, 4),
            "ci95": [round(d_lo, 4), round(d_hi, 4)],
            "statistically_improved": bool(d_lo > 0.0)
        }
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    gate = CenterDistillGate(settings)
    if gate.using_fallback:
        logger.error("Gate in fallback mode — cannot evaluate checkpoint ablation.")
        return 1

    orig_res = evaluate_dataset(_GOLDEN_ORIG, gate)
    repaired_res = evaluate_dataset(_GOLDEN_REPAIRED, gate)

    print("\n" + "=" * 76)
    print("SECOND_MASS ABLATION EXPERIMENT — REMOVING ALTERNATIVES BRANCH")
    print("=" * 76)

    for res in (orig_res, repaired_res):
        print(f"\n--- Dataset: {res['dataset']} (n={res['n']}) ---")
        std = res["standard_policy"]
        abl = res["ablated_policy_no_second_mass"]
        p_delta = res["paired_delta_3class"]

        print(f"  Standard Policy (3-class):  {std['accuracy_3class']:.1%}  [{std['ci95_3class'][0]:.1%}, {std['ci95_3class'][1]:.1%}]")
        print(f"  Ablated Policy (no 2nd_m):  {abl['accuracy_3class']:.1%}  [{abl['ci95_3class'][0]:.1%}, {abl['ci95_3class'][1]:.1%}]")
        print(f"  Paired 3-Class Δ Accuracy:   {p_delta['mean']:+.1%}  [{p_delta['ci95'][0]:+.1%}, {p_delta['ci95'][1]:+.1%}]")
        print(f"  Binary Task Accuracy (std): {std['accuracy_binary']:.1%}")
        print(f"  Binary Task Accuracy (abl): {abl['accuracy_binary']:.1%}")

        if p_delta["statistically_improved"]:
            print("  ✅ Removal of second_mass branch STATISTICALLY IMPROVES 3-class accuracy!")
        elif p_delta["mean"] > 0:
            print("  ✅ Directional accuracy improvement from removing second_mass branch.")
        else:
            print("  ❌ No accuracy improvement from removing second_mass branch.")

    out_data = {
        "golden_gate_orig": orig_res,
        "golden_gate_repaired": repaired_res
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n✅ Results written to {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
