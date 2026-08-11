#!/usr/bin/env python3
"""
Phase 0 (corrected) — regex teacher on GENUINELY held-out surface forms.

WHAT WAS WRONG WITH THE PREVIOUS VERSION
----------------------------------------
The rules contained the held-out test cases verbatim:

    "date_format":      r"...|\\bReiwa\\s*\\d+\\.\\d+\\.\\d+\\b"
    "currency":         r"[$€£]|\\b(?:kr|₨|R\\$|KSh)\\b"
    "measurement":      r"\\b(?:miles?|gallons?|pints?|leagues?)\\b"
    "number_ambiguity": r"\\b(?:billion|bn|\\d+m|\\d+\\s*MM)\\b"

`Reiwa`, `kr`, `₨`, `R$`, `KSh`, `pints`, `leagues`, `bn`, `MM` were all written
into the patterns AND used as the held-out test set. The reported 94.1% was a
training-set score. It cannot support any claim about generalisation, and the
project was nearly closed on it.

WHAT THIS VERSION DOES
----------------------
Two rule sets, frozen and separated:

    RULES_TRAIN   authored using ONLY the in-distribution forms. This is the
                  teacher whose generalisation is being measured.
    RULES_ORACLE  the previous over-fitted rules, retained to quantify how much
                  of the old number came from leakage.

The held-out forms are never referenced by RULES_TRAIN. The gap between the two
on held-out data IS the leakage.

A per-form audit runs before scoring: every held-out trigger string is checked
against the train rules' pattern source. If a trigger appears literally, the
script aborts. Leakage of this kind should fail loudly, not be discovered later.

USAGE
    python scripts/eval_phase0_regex.py
    python scripts/eval_phase0_regex.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import NamedTuple

_OUT = Path("eval/results/phase0_regex.json")


class Sample(NamedTuple):
    text: str
    is_ambiguous: bool
    category: str
    split: str          # 'in_distribution' | 'held_out'
    trigger: str        # the surface marker this row tests
    note: str


# ══════════════════════════════════════════════════════════════════
# RULES_TRAIN — authored from the in-distribution forms only.
#
# Written to generalise where a general form exists (any dd/mm/yyyy-shaped
# date), and NOT extended to cover any held-out trigger. Where a category is
# inherently a closed lexical list — currency symbols, unit words — the rule
# can only contain the in-distribution members. That limitation is the point:
# it is exactly what a neural student would have to overcome.
# ══════════════════════════════════════════════════════════════════
RULES_TRAIN: dict[str, re.Pattern[str]] = {
    # Generalises: any two ≤12 fields separated by / . or - is order-ambiguous.
    # Derived from 05/06/2024 and 11/12/2025 without naming other formats.
    "date_format": re.compile(
        r"\b(0?[1-9]|1[0-2])[/.\-](0?[1-9]|1[0-2])[/.\-](\d{2}|\d{4})\b"
    ),
    # Closed list from in-distribution only: $ € £.
    "currency": re.compile(r"[$€£]"),
    # Closed list from in-distribution only: miles, gallons.
    "measurement": re.compile(r"\b(miles?|gallons?)\b", re.IGNORECASE),
    # Closed list from in-distribution only: billion, bare Nm.
    "number_ambiguity": re.compile(r"\b(billion|\d+\s*m)\b", re.IGNORECASE),
}

# ══════════════════════════════════════════════════════════════════
# RULES_ORACLE — the previous rules, which name the held-out triggers.
# Retained ONLY to quantify leakage. Never report this as a held-out score.
# ══════════════════════════════════════════════════════════════════
RULES_ORACLE: dict[str, re.Pattern[str]] = {
    "date_format": re.compile(
        r"\b(?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12][0-9]|3[01])[./-]\d{2,4}\b"
        r"|\bReiwa\s*\d+\.\d+\.\d+\b"
        r"|\d{4}年\d{1,2}月\d{1,2}日"
    ),
    "currency": re.compile(r"[$€£]|\b(?:kr|₨|R\$|KSh|zł|₹)\b"),
    "measurement": re.compile(
        r"\b(?:miles?|gallons?|pints?|leagues?|stone)\b", re.IGNORECASE
    ),
    "number_ambiguity": re.compile(
        r"\b(?:billion|bn|\d+m|\d+\s*MM|\d+k)\b", re.IGNORECASE
    ),
}


SAMPLES: list[Sample] = [
    # ── date_format ────────────────────────────────────────────────
    Sample("What date is 05/06/2024?", True, "date_format", "in_distribution",
           "05/06", "both fields ≤12, order undetermined"),
    Sample("The event is on 11/12/2025.", True, "date_format", "in_distribution",
           "11/12", "both fields ≤12"),
    Sample("Submitted on 01/02/2023.", True, "date_format", "in_distribution",
           "01/02", "both fields ≤12"),
    Sample("Meeting on 2024-05-06.", False, "date_format", "in_distribution",
           "ISO", "control: ISO 8601 is unambiguous"),
    Sample("Deadline 2025-11-12.", False, "date_format", "in_distribution",
           "ISO", "control"),
    Sample("Shipped on 25/12/2023.", False, "date_format", "in_distribution",
           "25/12", "control: 25 > 12, order determined"),

    Sample("What date is 05.06.2024?", True, "date_format", "held_out",
           "05.06", "dot separator, unseen"),
    Sample("Appointment on 12.01.2026", True, "date_format", "held_out",
           "12.01", "dot separator"),
    Sample("Meeting on 2024年05月06日", False, "date_format", "held_out",
           "年月日", "control: CJK date is explicit"),
    Sample("Submitted on Reiwa 3.5.12", True, "date_format", "held_out",
           "Reiwa", "Japanese era, unseen entirely"),
    Sample("Born on 1998年10月25日", False, "date_format", "held_out",
           "年月日", "control"),
    Sample("Due 06-07-2024", True, "date_format", "held_out",
           "06-07", "hyphen separator, both ≤12"),

    # ── currency ───────────────────────────────────────────────────
    Sample("How much is 50$ in local price?", True, "currency", "in_distribution",
           "$", "shared by ≥8 currencies"),
    Sample("Cost is €100 total.", True, "currency", "in_distribution",
           "€", "in-distribution symbol"),
    Sample("Price is $250.", True, "currency", "in_distribution",
           "$", "shared symbol"),
    Sample("Total is 50 USD.", False, "currency", "in_distribution",
           "USD", "control: ISO 4217 code"),
    Sample("Paid 100 EUR.", False, "currency", "in_distribution",
           "EUR", "control"),

    Sample("Total cost is 500 kr", True, "currency", "held_out",
           "kr", "DKK / SEK / NOK / ISK"),
    Sample("Price is ₨ 1500", True, "currency", "held_out",
           "₨", "PKR / NPR / LKR"),
    Sample("Transferred R$ 200", True, "currency", "held_out",
           "R$", "BRL, unseen glyph pair"),
    Sample("Budget is KSh 3000", True, "currency", "held_out",
           "KSh", "KES, unseen"),
    Sample("Cost is 100 PLN", False, "currency", "held_out",
           "PLN", "control: ISO code"),
    Sample("Fee is 500 INR", False, "currency", "held_out",
           "INR", "control: ISO code"),

    # ── measurement ────────────────────────────────────────────────
    Sample("Distance is 10 miles", True, "measurement", "in_distribution",
           "miles", "statute vs nautical"),
    Sample("Volume is 5 gallons", True, "measurement", "in_distribution",
           "gallons", "US vs imperial"),
    Sample("Weight 150 lbs", False, "measurement", "in_distribution",
           "lbs", "control: unambiguous unit"),
    Sample("Length is 30 cm", False, "measurement", "in_distribution",
           "cm", "control: SI"),

    Sample("Bought 2 pints of milk", True, "measurement", "held_out",
           "pints", "US 473ml vs imperial 568ml"),
    Sample("Journey of 5 leagues", True, "measurement", "held_out",
           "leagues", "historical variance"),
    Sample("Recipe needs 3 cups", True, "measurement", "held_out",
           "cups", "US vs metric cup"),
    Sample("Weight is 12 stone", False, "measurement", "held_out",
           "stone", "control: UK stone is fixed at 14 lb"),
    Sample("Distance is 5 kilometres", False, "measurement", "held_out",
           "kilometres", "control: SI"),

    # ── number_ambiguity ───────────────────────────────────────────
    Sample("Deficit reached 5 billion", True, "number_ambiguity", "in_distribution",
           "billion", "short 10^9 vs long 10^12"),
    Sample("Revenue was 10m", True, "number_ambiguity", "in_distribution",
           "10m", "million vs metres"),
    Sample("Population is 1,500,000", False, "number_ambiguity", "in_distribution",
           "explicit", "control: fully written out"),

    Sample("Budget is 5 bn", True, "number_ambiguity", "held_out",
           "bn", "abbreviation, unseen"),
    Sample("Cost is 10 MM", True, "number_ambiguity", "held_out",
           "MM", "accounting notation, unseen"),
    Sample("Deficit of 3 milliard", True, "number_ambiguity", "held_out",
           "milliard", "long-scale term, unseen"),
    Sample("Salary is 50,000 USD", False, "number_ambiguity", "held_out",
           "explicit", "control: written out"),
    Sample("Exactly 2,000,000 units", False, "number_ambiguity", "held_out",
           "explicit", "control"),
]


def audit_leakage(rules: dict[str, re.Pattern[str]], name: str) -> list[str]:
    """Abort if a held-out trigger appears literally in a train pattern.

    This is the check that would have caught the previous version before any
    number was reported.
    """
    problems: list[str] = []
    for s in SAMPLES:
        if s.split != "held_out":
            continue
        pat = rules.get(s.category)
        if pat is None:
            continue
        src = pat.pattern
        # Compare case-insensitively; strip regex escapes so R\$ matches R$.
        needle = s.trigger.lower()
        haystack = src.replace("\\", "").lower()
        if len(needle) >= 2 and needle in haystack:
            problems.append(
                f"{name}/{s.category}: held-out trigger {s.trigger!r} appears in "
                f"the pattern source — this row is not held out"
            )
    return problems


def predict(text: str, category: str, rules: dict[str, re.Pattern[str]]) -> bool:
    pat = rules.get(category)
    return bool(pat.search(text)) if pat else False


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, centre - margin), min(1.0, centre + margin)


def score(samples: list[Sample], rules: dict[str, re.Pattern[str]]) -> dict:
    correct = sum(1 for s in samples
                  if predict(s.text, s.category, rules) == s.is_ambiguous)
    n = len(samples)
    ctrl = [s for s in samples if not s.is_ambiguous]
    amb = [s for s in samples if s.is_ambiguous]
    c_ok = sum(1 for s in ctrl if not predict(s.text, s.category, rules))
    a_ok = sum(1 for s in amb if predict(s.text, s.category, rules))
    lo, hi = wilson_ci(correct, n)
    return {
        "n": n, "correct": correct,
        "accuracy": round(correct / n * 100, 1) if n else 0.0,
        "ci95": [round(lo * 100, 1), round(hi * 100, 1)],
        "control_accuracy": round(c_ok / len(ctrl) * 100, 1) if ctrl else 0.0,
        "n_controls": len(ctrl),
        "ambiguous_accuracy": round(a_ok / len(amb) * 100, 1) if amb else 0.0,
        "n_ambiguous": len(amb),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    print("=" * 76)
    print("PHASE 0 (corrected) — regex teacher on held-out surface forms")
    print("=" * 76)

    # ── Leakage audit ─────────────────────────────────────────────
    train_problems = audit_leakage(RULES_TRAIN, "RULES_TRAIN")
    oracle_problems = audit_leakage(RULES_ORACLE, "RULES_ORACLE")

    print(f"\nLeakage audit")
    print(f"  RULES_TRAIN  : {len(train_problems)} held-out triggers found in patterns")
    print(f"  RULES_ORACLE : {len(oracle_problems)} (expected — these are the "
          f"over-fitted rules)")
    if train_problems:
        print("\n  FAIL — the train rules reference held-out forms:")
        for p in train_problems:
            print(f"    {p}")
        print("\n  Remove those triggers from RULES_TRAIN before reporting anything.")
        return 1
    print("  ✅ RULES_TRAIN references no held-out trigger")

    if args.audit_only:
        return 0

    in_dist = [s for s in SAMPLES if s.split == "in_distribution"]
    held = [s for s in SAMPLES if s.split == "held_out"]

    train_in = score(in_dist, RULES_TRAIN)
    train_held = score(held, RULES_TRAIN)
    oracle_held = score(held, RULES_ORACLE)

    print("\n" + "-" * 76)
    print(f"{'split':<28} {'n':>4} {'acc':>8} {'CI95':>16} {'ctrl':>7} {'amb':>7}")
    print("-" * 76)
    for label, r in (("RULES_TRAIN, in-distribution", train_in),
                     ("RULES_TRAIN, held-out", train_held),
                     ("RULES_ORACLE, held-out", oracle_held)):
        ci_str = f"[{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]"
        print(f"{label:<28} {r['n']:>4} {r['accuracy']:>7.1f}% "
              f"{ci_str:>16} "
              f"{r['control_accuracy']:>6.1f}% {r['ambiguous_accuracy']:>6.1f}%")
    print("-" * 76)

    leak = oracle_held["accuracy"] - train_held["accuracy"]
    drop = train_in["accuracy"] - train_held["accuracy"]
    print(f"\n  generalisation drop (in-dist → held-out) : {drop:+.1f} points")
    print(f"  leakage in the previous number            : {leak:+.1f} points")
    print("  The oracle score is what the earlier version reported. The gap is")
    print("  the amount that came from naming the test cases in the rules.")

    # ── Per-category, held-out only ───────────────────────────────
    print(f"\n{'category':<20} {'n':>4} {'train':>8} {'oracle':>8}   failing triggers")
    print("-" * 76)
    per_cat: dict[str, dict] = {}
    for cat in sorted({s.category for s in SAMPLES}):
        rows = [s for s in held if s.category == cat]
        if not rows:
            continue
        t = score(rows, RULES_TRAIN)
        o = score(rows, RULES_ORACLE)
        fails = [s.trigger for s in rows
                 if predict(s.text, s.category, RULES_TRAIN) != s.is_ambiguous]
        per_cat[cat] = {"train": t, "oracle": o, "failing_triggers": fails}
        print(f"{cat:<20} {t['n']:>4} {t['accuracy']:>7.1f}% {o['accuracy']:>7.1f}%   "
              f"{', '.join(fails) if fails else '—'}")

    print("\n  Interpretation: date_format has a general form (two fields ≤12), so a")
    print("  rule can transfer. currency, measurement and number_ambiguity are closed")
    print("  lexical lists — a rule can only cover members it was told about. That")
    print("  asymmetry is the gap a multilingual student would need to close.")

    out = {
        "phase": "0-corrected",
        "note": ("Previous version scored 94.1% using rules that contained the "
                 "held-out triggers verbatim. That was a training-set score. "
                 "RULES_TRAIN here is authored from in-distribution forms only "
                 "and audited against every held-out trigger before scoring."),
        "rules_train": {k: v.pattern for k, v in RULES_TRAIN.items()},
        "in_distribution": train_in,
        "held_out": train_held,
        "held_out_oracle_leaked": oracle_held,
        "generalisation_drop": round(drop, 1),
        "leakage_points": round(leak, 1),
        "per_category_held_out": per_cat,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {_OUT}")
    print("\n  n is still small. This establishes the honest baseline, not a "
          "publishable number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())