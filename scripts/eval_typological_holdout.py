#!/usr/bin/env python3
"""
Typological Benchmark Validation: Holdout Generalization Test.

2:1 Dev/Holdout Split
---------------------
For each of the 14 categories (15 rows each), split into:
  - 10 Dev rows (pattern discovery)
  - 5 Held-out rows (unseen evaluation)

If accuracy on held-out rows holds near dev accuracy, rules generalize across
lexical/typological variations within defined categories. If it collapses,
rules overfit generator templates.

USAGE
    python scripts/eval_typological_holdout.py
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from scripts.regex_baseline_typological import is_ambiguous_regex

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("eval/datasets/typological_ambiguity.jsonl")
_OUT_PATH = Path("eval/results/typological_holdout_validation.json")
_SEED = 42


def evaluate_holdout_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Group by category
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    rng = random.Random(_SEED)

    dev_total_correct = 0
    dev_total_count = 0
    holdout_total_correct = 0
    holdout_total_count = 0

    per_cat_results: dict[str, Any] = {}

    for cat, items in by_cat.items():
        shuffled = list(items)
        rng.shuffle(shuffled)

        # 10 dev, 5 holdout
        dev_set = shuffled[:10]
        holdout_set = shuffled[10:]

        dev_correct = sum(1 for r in dev_set if is_ambiguous_regex(r)[0] == (r["expected_behaviour"] == "AMBIGUOUS"))
        holdout_correct = sum(1 for r in holdout_set if is_ambiguous_regex(r)[0] == (r["expected_behaviour"] == "AMBIGUOUS"))

        dev_acc = dev_correct / len(dev_set) if dev_set else 0.0
        holdout_acc = holdout_correct / len(holdout_set) if holdout_set else 0.0

        dev_total_correct += dev_correct
        dev_total_count += len(dev_set)
        holdout_total_correct += holdout_correct
        holdout_total_count += len(holdout_set)

        per_cat_results[cat] = {
            "dev_n": len(dev_set),
            "dev_acc": round(dev_acc, 4),
            "holdout_n": len(holdout_set),
            "holdout_acc": round(holdout_acc, 4),
            "acc_drop": round(dev_acc - holdout_acc, 4),
        }

    overall_dev_acc = dev_total_correct / dev_total_count
    overall_holdout_acc = holdout_total_correct / holdout_total_count

    return {
        "dev_overall_acc": round(overall_dev_acc, 4),
        "holdout_overall_acc": round(overall_holdout_acc, 4),
        "overall_acc_drop": round(overall_dev_acc - overall_holdout_acc, 4),
        "per_category": per_cat_results,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not _DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {_DATASET_PATH}")

    rows = [json.loads(line) for line in open(_DATASET_PATH, encoding="utf-8")]
    logger.info("Loaded %d rows for holdout validation...", len(rows))

    holdout_res = evaluate_holdout_split(rows)

    print("\n" + "=" * 78)
    print("TYPOLOGICAL BENCHMARK HOLDOUT VALIDATION (2:1 Dev/Test Split)")
    print("=" * 78)
    print(f"Overall Dev Accuracy (140 rows):     {holdout_res['dev_overall_acc']:.1%}")
    print(f"Overall Holdout Accuracy (70 rows):  {holdout_res['holdout_overall_acc']:.1%}")
    print(f"Accuracy Drop (Dev vs Holdout):     {holdout_res['overall_acc_drop']:+.1%}")
    print("-" * 78)
    print(f"{'Category':<22} {'Dev Acc (10)':>14} {'Holdout Acc (5)':>16} {'Drop':>10}")
    print("-" * 78)

    for cat, res in holdout_res["per_category"].items():
        print(f"{cat:<22} {res['dev_acc']:>14.1%} {res['holdout_acc']:>16.1%} {res['acc_drop']:>+10.1%}")

    print("-" * 78)

    out_data = {
        "n_total": len(rows),
        "dev_n": 140,
        "holdout_n": 70,
        "holdout_results": holdout_res,
        "limitation_note": "Dataset labels are single-author and unvalidated by independent human annotators.",
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n✅ Clean holdout results written to {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
