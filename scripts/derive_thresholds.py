"""Derive optimal gate thresholds on a validation split of golden_gate.jsonl.

Per AGENTS.md rule 2, threshold evaluation order is fixed:
    if max_prob > tau_conf:        ANSWER
    elif second_mass > tau_multi:  ALTERNATIVES
    elif entropy > tau_ent:        CLARIFY
    else:                          CLARIFY  (safe default — asking is cheaper than answering wrongly)

Entropy MUST be in nats (natural log).

Objective: macro-F1 (not raw accuracy) to prevent the grid search from
collapsing to a degenerate majority-class solution.

Usage:
    python scripts/derive_thresholds.py [--dataset eval/datasets/golden_gate.jsonl] [--out eval/results/derived_thresholds.json]
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


CLASSES = ("ANSWER", "ALTERNATIVES", "CLARIFY")


def _predict(
    item: dict[str, Any],
    tau_conf: float,
    tau_multi: float,
    tau_ent: float,
) -> str:
    """Predict behaviour under the fixed threshold order (AGENTS.md rule 2)."""
    if item["max_prob"] > tau_conf:
        return "ANSWER"
    elif item["second_mass"] > tau_multi:
        return "ALTERNATIVES"
    elif item["entropy"] > tau_ent:
        return "CLARIFY"
    else:
        # DECISION: safe default — asking is cheaper than answering wrongly.
        return "CLARIFY"


def evaluate_thresholds(
    items: list[dict[str, Any]],
    tau_conf: float,
    tau_multi: float,
    tau_ent: float,
) -> float:
    """Evaluate macro-F1 of a threshold triple.

    Macro-F1 prevents the grid search from collapsing to a degenerate
    majority-class solution (always-ANSWER scores ~0.33 macro-F1).
    """
    if not items:
        return 0.0

    preds = [_predict(item, tau_conf, tau_multi, tau_ent) for item in items]
    golds = [item["expected_behaviour"] for item in items]

    f1_sum = 0.0
    for cls in CLASSES:
        tp = sum(1 for p, g in zip(preds, golds) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, golds) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, golds) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_sum += f1

    return f1_sum / len(CLASSES)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Derive optimal gate thresholds")
    parser.add_argument(
        "--dataset",
        default="eval/datasets/golden_gate.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--out",
        default="eval/results/derived_thresholds.json",
        help="Output JSON file for derived thresholds",
    )
    args = parser.parse_args()

    settings = get_settings()
    gate_path = settings.gate_checkpoint_path
    if not gate_path or not gate_path.exists():
        logger.error("No valid gate checkpoint found at %s", gate_path)
        return

    logger.info("Initializing CenterDistillGate with checkpoint: %s", gate_path)
    gate = CenterDistillGate(gate_path)
    if gate.using_fallback:
        logger.error("Gate loaded in fallback mode — cannot calibrate learned thresholds")
        return

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return

    with open(dataset_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    logger.info("Running forward passes for %d dataset rows...", len(rows))
    extracted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = row["question"]
        context = row.get("context", "")
        gold = row["expected_behaviour"]
        decision = gate(question, context)

        extracted.append(
            {
                "id": row.get("id", f"row_{idx}"),
                "max_prob": float(decision["max_prob"]),
                "second_mass": float(decision["second_mass"]),
                "entropy": float(decision["entropy"]),
                "expected_behaviour": gold,
            }
        )

    # 50/50 Train / Validation split for threshold derivation
    n_train = len(extracted) // 2
    train_items = extracted[:n_train]
    val_items = extracted[n_train:]

    logger.info("Grid searching optimal thresholds on %d calibration samples...", len(train_items))

    best_score = -1.0
    best_tau_conf = 0.44
    best_tau_multi = 0.24
    best_tau_ent = 1.51

    # Search space
    conf_grid = np.linspace(0.20, 0.60, 41)
    multi_grid = np.linspace(0.05, 0.40, 36)
    ent_grid = np.linspace(1.00, 1.60, 31)

    for tc in conf_grid:
        for tm in multi_grid:
            for te in ent_grid:
                score = evaluate_thresholds(train_items, float(tc), float(tm), float(te))
                if score > best_score:
                    best_score = score
                    best_tau_conf = float(tc)
                    best_tau_multi = float(tm)
                    best_tau_ent = float(te)

    val_score = evaluate_thresholds(val_items, best_tau_conf, best_tau_multi, best_tau_ent)
    paper_train_score = evaluate_thresholds(train_items, 0.44, 0.24, 1.51)
    paper_val_score = evaluate_thresholds(val_items, 0.44, 0.24, 1.51)

    result = {
        "derived_thresholds": {
            "tau_conf": round(best_tau_conf, 4),
            "tau_multi": round(best_tau_multi, 4),
            "tau_ent": round(best_tau_ent, 4),
        },
        "objective": "macro-F1",
        "fallthrough_default": "CLARIFY",
        "calibration_results": {
            "n_calibration_samples": len(train_items),
            "n_validation_samples": len(val_items),
            "calibrated_train_macro_f1": round(best_score, 4),
            "calibrated_val_macro_f1": round(val_score, 4),
            "paper_defaults_train_macro_f1": round(paper_train_score, 4),
            "paper_defaults_val_macro_f1": round(paper_val_score, 4),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Threshold optimization complete (objective=macro-F1, fallthrough=CLARIFY)!")
    logger.info("  Optimal thresholds: tau_conf=%.4f, tau_multi=%.4f, tau_ent=%.4f", best_tau_conf, best_tau_multi, best_tau_ent)
    logger.info("  Train macro-F1: %.4f (derived) vs %.4f (paper default)", best_score, paper_train_score)
    logger.info("  Val macro-F1:   %.4f (derived) vs %.4f (paper default)", val_score, paper_val_score)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
