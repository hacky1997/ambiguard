"""Run the four-arm comparison harness on golden_gate.jsonl.

This is the central deliverable of Phase 1 (spec §3.4):
    eval/run_comparison.py → eval/results/comparison.json + markdown table

The table goes at the very top of the README, above the architecture diagram.
Every number must trace to a committed JSON file from an actual run —
never fabricated (AGENTS.md rule 5).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from eval.arms import ArmResult
from eval.arms.centerdistill_arm import CenterDistillArm
from eval.arms.confidence_arm import ConfidenceArm
from eval.arms.llm_judge_arm import LLMJudgeArm
from eval.arms.majority_arm import MajorityArm
from eval.metrics.balanced_accuracy import balanced_accuracy
from eval.metrics.behaviour_accuracy import behaviour_accuracy
from eval.metrics.bootstrap import bootstrap_ci
from eval.metrics.worst_cluster_f1 import worst_cluster_f1
from eval.report import format_comparison_table

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("eval/datasets/golden_gate.jsonl")
_RESULTS_DIR = Path("eval/results")


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset file."""
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        logger.error(
            "golden_gate.jsonl is HUMAN-AUTHORED and must not be generated. "
            "See AGENTS.md rule 6."
        )
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, exc)
    return rows


def _run_arm(
    arm: Any,
    dataset: list[dict[str, Any]],
    determinism_runs: int = 3,
) -> dict[str, Any]:
    """Run a single arm on the full dataset and compute all metrics."""
    gold: list[str] = [row["expected_behaviour"] for row in dataset]
    predictions: list[str] = []
    latencies: list[float] = []
    total_cost: float = 0.0
    per_row_results: list[dict[str, Any]] = []

    for row in dataset:
        result: ArmResult = arm.predict(row["question"], row["context"])
        predictions.append(result["prediction"])
        latencies.append(result["latency_ms"])
        total_cost += result["cost_usd"]
        per_row_results.append(
            {
                "id": row["id"],
                "prediction": result["prediction"],
                "gold": row["expected_behaviour"],
                "correct": result["prediction"] == row["expected_behaviour"],
                "latency_ms": result["latency_ms"],
                "cost_usd": result["cost_usd"],
                "metadata": result["metadata"],
            }
        )

    # Compute metrics
    acc_point, acc_lower, acc_upper = bootstrap_ci(
        predictions, gold, behaviour_accuracy
    )
    bal_acc = balanced_accuracy(predictions, gold)
    wc_f1 = worst_cluster_f1(predictions, gold)

    # Latency stats
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    p50 = sorted_latencies[n // 2] if n > 0 else 0.0
    p95 = sorted_latencies[int(n * 0.95)] if n > 0 else 0.0

    # Cost per 1000 decisions
    cost_per_1k = (total_cost / len(dataset)) * 1000 if dataset else 0.0

    # Determinism check
    is_deterministic = arm.deterministic
    if determinism_runs > 1 and len(dataset) > 0:
        first_row = dataset[0]
        first_results = [
            arm.predict(first_row["question"], first_row["context"])["prediction"]
            for _ in range(determinism_runs)
        ]
        is_deterministic = len(set(first_results)) == 1

    # Check for fallback
    any_fallback = any(
        r.get("metadata", {}).get("fallback_used", False) for r in per_row_results
    )

    return {
        "arm_name": arm.name,
        "behaviour_accuracy": acc_point,
        "balanced_accuracy": bal_acc,
        "ci_95_lower": acc_lower,
        "ci_95_upper": acc_upper,
        "worst_cluster_f1": wc_f1,
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "cost_per_1k_usd": round(cost_per_1k, 4),
        "deterministic": is_deterministic,
        "fallback_used": any_fallback,
        "n_samples": len(dataset),
        "per_row": per_row_results,
    }


def run_comparison(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full four-arm comparison and write results.

    Returns the comparison results dict.
    """
    ds_path = dataset_path or _DATASET_PATH
    out_dir = output_dir or _RESULTS_DIR

    dataset = _load_dataset(ds_path)
    logger.info("Loaded %d rows from %s", len(dataset), ds_path)

    # Instantiate arms (spec §3.1)
    arms: list[Any] = [
        CenterDistillArm(),
        LLMJudgeArm(),
        MajorityArm.from_dataset([r["expected_behaviour"] for r in dataset]),
        ConfidenceArm(),
    ]

    results: list[dict[str, Any]] = []
    for arm in arms:
        logger.info("Running arm: %s", arm.name)
        start = time.time()
        arm_result = _run_arm(arm, dataset)
        elapsed = time.time() - start
        logger.info(
            "  %s: acc=%.3f [%.3f, %.3f] in %.1fs",
            arm.name,
            arm_result["behaviour_accuracy"],
            arm_result["ci_95_lower"],
            arm_result["ci_95_upper"],
            elapsed,
        )
        results.append(arm_result)

    # Build comparison output
    comparison: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(ds_path),
        "n_samples": len(dataset),
        "arms": [
            {k: v for k, v in r.items() if k != "per_row"} for r in results
        ],
        "per_row_details": {r["arm_name"]: r["per_row"] for r in results},
    }

    # Write results
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "comparison.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", results_path)

    # Print markdown table
    table = format_comparison_table(comparison)
    print(table)

    # Also write markdown
    table_path = out_dir / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)

    return comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_comparison()
