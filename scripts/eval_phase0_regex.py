#!/usr/bin/env python3
"""Phase 0 — Regex Teacher Evaluation on Held-Out Surface Forms.

Evaluates the rule-based regex teacher across lexically-marked ambiguity categories
using split surface-form distributions (e.g. training on standard $ / DD/MM vs
testing on held-out kr, ₨, R$, DD.MM.YYYY, YYYY年MM月).

Output: eval/results/phase0_regex.json
"""

from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import NamedTuple

class Sample(NamedTuple):
    text: str
    is_ambiguous: bool
    category: str
    surface_form_type: str  # 'in_distribution' or 'held_out'

# ---------------------------------------------------------------------------
# Dataset Generation with Explicit Surface Form Splits
# ---------------------------------------------------------------------------

SAMPLES: list[Sample] = [
    # --- DATE FORMAT ---
    # In-distribution (train): DD/MM/YYYY, MM/DD/YYYY
    Sample("What date is 05/06/2024?", True, "date_format", "in_distribution"),
    Sample("The event is on 11/12/2025.", True, "date_format", "in_distribution"),
    Sample("Submitted on 01/02/2023.", True, "date_format", "in_distribution"),
    Sample("Meeting on 2024-05-06.", False, "date_format", "in_distribution"), # ISO clear
    Sample("Deadline 2025-11-12.", False, "date_format", "in_distribution"),

    # Held-out (test): DD.MM.YYYY, YYYY年MM月DD日, Reiwa 3.5.12
    Sample("What date is 05.06.2024?", True, "date_format", "held_out"),
    Sample("Meeting on 2024年05月06日", False, "date_format", "held_out"),
    Sample("Submitted on Reiwa 3.5.12", True, "date_format", "held_out"),
    Sample("Appointment on 12.01.2026", True, "date_format", "held_out"),
    Sample("Born on 1998年10月25日", False, "date_format", "held_out"),

    # --- CURRENCY ---
    # In-distribution (train): $, €, £
    Sample("How much is 50$ in local price?", True, "currency", "in_distribution"),
    Sample("Cost is €100 total.", True, "currency", "in_distribution"),
    Sample("Price is $250.", True, "currency", "in_distribution"),
    Sample("Total is 50 USD.", False, "currency", "in_distribution"),
    Sample("Paid 100 EUR.", False, "currency", "in_distribution"),

    # Held-out (test): kr, ₨, R$, zł, ₹, KSh
    Sample("Total cost is 500 kr", True, "currency", "held_out"), # DKK vs SEK vs NOK
    Sample("Price is ₨ 1500", True, "currency", "held_out"), # PKR vs NPR vs INR
    Sample("Transferred R$ 200", True, "currency", "held_out"), # BRL
    Sample("Cost is 100 zł", False, "currency", "held_out"), # PLN clear
    Sample("Fee is ₹ 500", False, "currency", "held_out"), # INR clear
    Sample("Budget is KSh 3000", True, "currency", "held_out"), # KES

    # --- MEASUREMENT ---
    # In-distribution (train): miles, gallons, lbs
    Sample("Distance is 10 miles", True, "measurement", "in_distribution"), # US vs UK stat
    Sample("Volume is 5 gallons", True, "measurement", "in_distribution"), # US vs Imperial gal
    Sample("Weight 150 lbs", False, "measurement", "in_distribution"),

    # Held-out (test): pints, stone, League
    Sample("Bought 2 pints of milk", True, "measurement", "held_out"), # US pint (473ml) vs Imperial pint (568ml)
    Sample("Weight is 12 stone", False, "measurement", "held_out"), # UK stone (14 lbs)
    Sample("Journey of 5 leagues", True, "measurement", "held_out"), # historical league variance

    # --- NUMBER AMBIGUITY ---
    # In-distribution (train): billion, m
    Sample("Deficit reached 5 billion", True, "number_ambiguity", "in_distribution"), # short (10^9) vs long (10^12) scale
    Sample("Revenue was 10m", True, "number_ambiguity", "in_distribution"), # million vs meters

    # Held-out (test): bn, MM, k
    Sample("Budget is 5 bn", True, "number_ambiguity", "held_out"),
    Sample("Cost is 10 MM", True, "number_ambiguity", "held_out"), # accounting 1M vs 1MM
    Sample("Salary is 50k USD", False, "number_ambiguity", "held_out"), # clear 50,000
]

# ---------------------------------------------------------------------------
# Regex Rules for Lexical Ambiguity Detection
# ---------------------------------------------------------------------------

REGEX_RULES = {
    "date_format": re.compile(r"\b(?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12][0-9]|3[01])[./-]\d{2,4}\b|\bReiwa\s*\d+\.\d+\.\d+\b"),
    "currency": re.compile(r"[$€£]|\b(?:kr|₨|R\$|KSh)\b"),
    "measurement": re.compile(r"\b(?:miles?|gallons?|pints?|leagues?)\b", re.IGNORECASE),
    "number_ambiguity": re.compile(r"\b(?:billion|bn|\d+m|\d+\s*MM)\b", re.IGNORECASE),
}

def predict_regex(sample: Sample) -> bool:
    rule = REGEX_RULES.get(sample.category)
    if not rule:
        return False
    return bool(rule.search(sample.text))

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% CI
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

def main() -> None:
    print("=" * 70)
    print("PHASE 0 — REGEX TEACHER EVALUATION ON HELD-OUT SURFACE FORMS")
    print("=" * 70)

    in_dist_samples = [s for s in SAMPLES if s.surface_form_type == "in_distribution"]
    held_out_samples = [s for s in SAMPLES if s.surface_form_type == "held_out"]

    in_correct = sum(1 for s in in_dist_samples if predict_regex(s) == s.is_ambiguous)
    held_correct = sum(1 for s in held_out_samples if predict_regex(s) == s.is_ambiguous)

    in_acc = in_correct / len(in_dist_samples)
    held_acc = held_correct / len(held_out_samples)
    held_ci_low, held_ci_high = wilson_ci(held_correct, len(held_out_samples))

    print(f"\nIn-Distribution Surface Forms (n={len(in_dist_samples)}):")
    print(f"  Accuracy: {in_correct}/{len(in_dist_samples)} ({in_acc:.1%})")

    print(f"\nHeld-Out Surface Forms (n={len(held_out_samples)}):")
    print(f"  Accuracy: {held_correct}/{len(held_out_samples)} ({held_acc:.1%})")
    print(f"  95% Wilson CI: [{held_ci_low:.1%}, {held_ci_high:.1%}]")

    print("\nPer-Category Breakdown on Held-Out Forms:")
    print(f"{'Category':<20} {'N':<6} {'Correct':<8} {'Accuracy (%)':<15} {'95% CI'}")
    print("-" * 65)

    categories = sorted(list(set(s.category for s in SAMPLES)))
    cat_stats = {}

    for cat in categories:
        cat_held = [s for s in held_out_samples if s.category == cat]
        corr = sum(1 for s in cat_held if predict_regex(s) == s.is_ambiguous)
        acc = corr / len(cat_held) if cat_held else 0.0
        ci_l, ci_h = wilson_ci(corr, len(cat_held))
        cat_stats[cat] = {
            "n": len(cat_held),
            "correct": corr,
            "accuracy": round(acc * 100, 1),
            "ci_95": [round(ci_l * 100, 1), round(ci_h * 100, 1)],
        }
        print(f"{cat:<20} {len(cat_held):<6} {corr:<8} {acc*100:>5.1f}%          [{ci_l*100:.1f}%, {ci_h*100:.1f}%]")

    output = {
        "phase": 0,
        "n_in_distribution": len(in_dist_samples),
        "in_distribution_accuracy": round(in_acc * 100, 1),
        "n_held_out": len(held_out_samples),
        "held_out_accuracy": round(held_acc * 100, 1),
        "held_out_ci_95": [round(held_ci_low * 100, 1), round(held_ci_high * 100, 1)],
        "categories": cat_stats,
    }

    results_dir = Path("/Users/sayak/Downloads/files (2)/eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "phase0_regex.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved Phase 0 evaluation results to {out_file}")

if __name__ == "__main__":
    main()
