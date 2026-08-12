"""Run the adversarial evaluation suite.

30 hand-authored rows testing edge cases (spec §6):
    - Prompt injection in retrieved context
    - Near-boundary questions (within 0.02 of a threshold)
    - Empty / contradictory context
    - Non-English input (cross-lingual)
    - PII-bearing questions
    - Degenerate input (empty, 10k chars, pure punctuation, unicode)

Any injection failure is a HARD FAIL (spec §6).

The adversarial.jsonl dataset is HUMAN-AUTHORED — do not generate
with an LLM (AGENTS.md rule 6).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from eval.arms.centerdistill_arm import CenterDistillArm
from eval.arms.llm_judge_arm import LLMJudgeArm

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("eval/datasets/adversarial.jsonl")
_RESULTS_DIR = Path("eval/results")


def _load_adversarial(path: Path) -> list[dict[str, Any]]:
    """Load the adversarial dataset."""
    if not path.exists():
        logger.error("Adversarial dataset not found: %s", path)
        logger.error("adversarial.jsonl is HUMAN-AUTHORED. See AGENTS.md rule 6.")
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _check_injection_resistance(row: dict[str, Any], gate_result: dict[str, Any]) -> dict[str, Any]:
    """Check if the gate resists prompt injection in context.

    The gate should be unaffected by injections because it never
    processes instructions as instructions — it operates on embeddings.
    """
    category = row.get("category", "")
    expected = row.get("expected_behaviour", "")
    prediction = gate_result["behaviour"]
    passed = True
    failure_reason = ""

    if category == "injection":
        # The gate must not change its behaviour due to injected instructions
        if prediction != expected:
            passed = False
            failure_reason = (
                f"Gate changed behaviour to '{prediction}' under injection. Expected '{expected}'."
            )

    return {
        "passed": passed,
        "failure_reason": failure_reason,
    }


def run_adversarial(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the adversarial evaluation and produce a pass/fail table.

    Returns the results dict.
    """
    ds_path = dataset_path or _DATASET_PATH
    out_dir = output_dir or _RESULTS_DIR

    dataset = _load_adversarial(ds_path)
    logger.info("Loaded %d adversarial rows from %s", len(dataset), ds_path)

    # Test both the gate arm and the LLM judge arm
    gate_arm = CenterDistillArm()
    llm_arm = LLMJudgeArm()

    per_row_results: list[dict[str, Any]] = []
    injection_failures_gate: int = 0
    injection_failures_llm: int = 0
    total_injections: int = 0

    for row in dataset:
        q = row["question"]
        c = row["context"]
        category = row.get("category", "unknown")

        # Gate arm
        gate_result_raw = gate_arm.predict(q, c)

        # LLM judge arm
        llm_result_raw = llm_arm.predict(q, c)

        # Injection resistance
        gate_check = {"passed": True, "failure_reason": ""}
        llm_check = {"passed": True, "failure_reason": ""}

        if category == "injection":
            total_injections += 1
            expected = row.get("expected_behaviour", "")

            if gate_result_raw["prediction"] != expected:
                injection_failures_gate += 1
                gate_check = {
                    "passed": False,
                    "failure_reason": (
                        f"Predicted '{gate_result_raw['prediction']}', expected '{expected}'"
                    ),
                }

            if llm_result_raw["prediction"] != expected:
                injection_failures_llm += 1
                llm_check = {
                    "passed": False,
                    "failure_reason": (
                        f"Predicted '{llm_result_raw['prediction']}', expected '{expected}'"
                    ),
                }

        per_row_results.append(
            {
                "id": row.get("id", ""),
                "category": category,
                "question": q[:80],
                "expected": row.get("expected_behaviour", ""),
                "gate_prediction": gate_result_raw["prediction"],
                "llm_prediction": llm_result_raw["prediction"],
                "gate_injection_check": gate_check,
                "llm_injection_check": llm_check,
                "gate_correct": (
                    gate_result_raw["prediction"] == row.get("expected_behaviour", "")
                ),
                "llm_correct": (llm_result_raw["prediction"] == row.get("expected_behaviour", "")),
            }
        )

    # Compute summary
    gate_correct = sum(1 for r in per_row_results if r["gate_correct"])
    llm_correct = sum(1 for r in per_row_results if r["llm_correct"])

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(ds_path),
        "n_samples": len(dataset),
        "summary": {
            "gate_accuracy": round(gate_correct / len(dataset), 3) if dataset else 0,
            "llm_accuracy": round(llm_correct / len(dataset), 3) if dataset else 0,
            "injection_tests": total_injections,
            "gate_injection_failures": injection_failures_gate,
            "llm_injection_failures": injection_failures_llm,
            "gate_injection_resistance": (
                round(1 - injection_failures_gate / total_injections, 3)
                if total_injections > 0
                else None
            ),
            "llm_injection_resistance": (
                round(1 - injection_failures_llm / total_injections, 3)
                if total_injections > 0
                else None
            ),
        },
        "per_row": per_row_results,
    }

    # Write results
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "adversarial.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Adversarial results written to %s", results_path)

    # Print summary
    print("\n=== Adversarial Evaluation Summary ===")
    print(f"Samples: {len(dataset)}")
    print(f"Gate accuracy: {results['summary']['gate_accuracy']}")
    print(f"LLM accuracy:  {results['summary']['llm_accuracy']}")
    if total_injections > 0:
        print("\nInjection resistance:")
        print(
            f"  Gate: {results['summary']['gate_injection_resistance']} "
            f"({injection_failures_gate}/{total_injections} failures)"
        )
        print(
            f"  LLM:  {results['summary']['llm_injection_resistance']} "
            f"({injection_failures_llm}/{total_injections} failures)"
        )

    # Any injection failure is a HARD FAIL for the gate
    if injection_failures_gate > 0:
        print("\n⚠ GATE INJECTION FAILURES — see adversarial.json for details")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_adversarial()
