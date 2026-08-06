#!/usr/bin/env python3
"""
Cascade feasibility analysis.

QUESTION
--------
The gate is free and ~500ms. The LLM judge costs ~$1.83/1k and ~750ms but is
22 points more accurate. A cascade is only worth building if the gate is
RELIABLE WHEN CONFIDENT — i.e. accuracy rises with max(P_S).

If accuracy is flat across confidence buckets, there is nothing to route on and
the cascade is dead. That is a real finding too, and it takes ten minutes to
learn instead of two days.

This script answers three things:
  1. Does gate accuracy increase with confidence? (bucketed accuracy + trend)
  2. If so, what escalation threshold gives the best cost/accuracy tradeoff?
  3. Do the two arms fail on DIFFERENT examples? (ensemble headroom)

No new inference. No API calls. Reads cached per-example predictions.

USAGE
-----
    python scripts/cascade_analysis.py \
        --results eval/results/comparison.json \
        --out     eval/results/cascade.json

If comparison.json holds only aggregates, pass --recompute to run the gate
locally over the dataset (still no API calls — the judge's cached predictions
are reused).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Cost/latency constants — measured, from the 600-sample comparison run.
GATE_COST_PER_1K = 0.00
JUDGE_COST_PER_1K = 1.83
GATE_P50_MS = 500.0
JUDGE_P50_MS = 751.0

BEHAVIOURS = ("ANSWER", "ALTERNATIVES", "CLARIFY")


def load_per_example(results_path: Path) -> dict:
    """Pull per-example predictions out of the comparison results file."""
    with open(results_path) as f:
        data = json.load(f)

    # Try per_row_details first (dict keyed by arm name),
    # then fall back to arms[].predictions.
    per_example: dict = {}

    prd = data.get("per_row_details")
    if isinstance(prd, dict) and prd:
        per_example = {k: v for k, v in prd.items() if isinstance(v, list) and v}

    if not per_example:
        for arm in data.get("arms", []):
            name = arm.get("arm_name", "")
            rows = arm.get("predictions") or arm.get("per_example")
            if not rows:
                continue
            per_example[name] = rows

    if not per_example:
        print(
            "ERROR: no per-example predictions in the results file.\n"
            "The comparison run stored only aggregates. Re-run it with\n"
            "per-example logging enabled, or pass --recompute.",
            file=sys.stderr,
        )
        sys.exit(1)
    return per_example


def find_arm(per_example: dict, *keywords: str) -> tuple[str, list]:
    for name, rows in per_example.items():
        low = name.lower()
        if any(k in low for k in keywords):
            return name, rows
    raise KeyError(f"No arm matching {keywords}. Available: {list(per_example)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/comparison.json")
    ap.add_argument("--out", default="eval/results/cascade.json")
    ap.add_argument("--buckets", type=int, default=5)
    args = ap.parse_args()

    per_example = load_per_example(Path(args.results))
    gate_name, gate_rows = find_arm(per_example, "centerdistill")
    judge_name, judge_rows = find_arm(per_example, "llm judge", "judge")

    n = len(gate_rows)
    if len(judge_rows) != n:
        print(f"ERROR: arm lengths differ ({n} vs {len(judge_rows)})", file=sys.stderr)
        return 1

    gold = np.array([r.get("expected") or r.get("gold") for r in gate_rows])
    gate_pred = np.array([r["prediction"] for r in gate_rows])
    judge_pred = np.array([r["prediction"] for r in judge_rows])

    def _get_conf(r: dict) -> float:
        """Extract max_prob from top-level or nested metadata."""
        if "max_prob" in r:
            return float(r["max_prob"])
        meta = r.get("metadata", {})
        if "max_prob" in meta:
            return float(meta["max_prob"])
        cd = meta.get("center_distribution") or r.get("center_distribution", [0.0])
        return float(max(cd))

    conf = np.array([_get_conf(r) for r in gate_rows], dtype=float)

    if not np.any(conf > 0):
        print("ERROR: no confidence values found on the gate arm.", file=sys.stderr)
        return 1

    gate_ok = gate_pred == gold
    judge_ok = judge_pred == gold

    print("=" * 74)
    print(f"CASCADE FEASIBILITY — n={n}")
    print("=" * 74)
    print(f"  {gate_name:<32} {gate_ok.mean():.1%}")
    print(f"  {judge_name:<32} {judge_ok.mean():.1%}")
    print(f"  confidence range: [{conf.min():.4f}, {conf.max():.4f}]  "
          f"mean {conf.mean():.4f}")

    # ── Q1: does accuracy rise with confidence? ───────────────────
    print("\n" + "-" * 74)
    print("Q1  Gate accuracy by confidence bucket")
    print("-" * 74)
    edges = np.quantile(conf, np.linspace(0, 1, args.buckets + 1))
    edges[-1] += 1e-9
    print(f"{'bucket':<10} {'conf range':<22} {'n':>5} {'gate acc':>10} {'judge acc':>10}")
    bucket_stats = []
    for i in range(args.buckets):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        g, j = gate_ok[m].mean(), judge_ok[m].mean()
        bucket_stats.append({
            "bucket": i + 1,
            "conf_lo": round(float(edges[i]), 4),
            "conf_hi": round(float(edges[i + 1]), 4),
            "n": int(m.sum()),
            "gate_acc": round(float(g), 4),
            "judge_acc": round(float(j), 4),
        })
        print(f"{i+1:<10} [{edges[i]:.4f}, {edges[i+1]:.4f})  "
              f"{int(m.sum()):>5} {g:>9.1%} {j:>10.1%}")

    # Trend test: correlation between confidence and correctness.
    if conf.std() > 0:
        r = float(np.corrcoef(conf, gate_ok.astype(float))[0, 1])
    else:
        r = 0.0
    top, bottom = bucket_stats[-1]["gate_acc"], bucket_stats[0]["gate_acc"]
    spread = top - bottom
    print(f"\n  corr(confidence, correct) = {r:+.4f}")
    print(f"  top bucket - bottom bucket = {spread:+.1%}")

    feasible = r > 0.10 and spread > 0.10
    if feasible:
        print("  ✅ Confidence carries signal — a cascade can work.")
    else:
        print("  ❌ Confidence does not predict correctness.")
        print("     A cascade cannot beat routing everything to the judge.")
        print("     Skip to Q3 (ensemble) instead.")

    # ── Q2: sweep the escalation threshold ────────────────────────
    print("\n" + "-" * 74)
    print("Q2  Cascade sweep — keep gate above tau, escalate below")
    print("-" * 74)
    print(f"{'tau':>8} {'kept%':>8} {'accuracy':>10} {'$/1k':>8} {'p50 ms':>9} {'vs judge':>10}")

    judge_acc = judge_ok.mean()
    rows = []
    for tau in np.quantile(conf, np.linspace(0, 0.95, 20)):
        keep = conf >= tau
        # Gate handles kept examples; judge handles the rest.
        correct = np.where(keep, gate_ok, judge_ok)
        acc = correct.mean()
        escalated = 1.0 - keep.mean()
        cost = escalated * JUDGE_COST_PER_1K
        # Escalated examples pay both latencies (gate runs first).
        lat = GATE_P50_MS + escalated * JUDGE_P50_MS
        rows.append({
            "tau": round(float(tau), 4),
            "kept_frac": round(float(keep.mean()), 4),
            "accuracy": round(float(acc), 4),
            "cost_per_1k": round(float(cost), 3),
            "p50_ms": round(float(lat), 1),
            "acc_vs_judge": round(float(acc - judge_acc), 4),
        })
        print(f"{tau:>8.4f} {keep.mean():>7.1%} {acc:>9.1%} "
              f"{cost:>8.2f} {lat:>9.1f} {acc - judge_acc:>+9.1%}")

    # Best operating point: cheapest config within 2 points of the judge.
    viable = [r_ for r_ in rows if r_["acc_vs_judge"] >= -0.02]
    best = min(viable, key=lambda r_: r_["cost_per_1k"]) if viable else None

    if best:
        saving = (1 - best["cost_per_1k"] / JUDGE_COST_PER_1K) * 100
        print(f"\n  Best operating point: tau={best['tau']:.4f}")
        print(f"    accuracy   {best['accuracy']:.1%}  "
              f"({best['acc_vs_judge']:+.1%} vs judge-only)")
        print(f"    cost       ${best['cost_per_1k']:.2f}/1k  "
              f"({saving:.0f}% cheaper than judge-only)")
        print(f"    latency    {best['p50_ms']:.0f} ms p50")
        print(f"    escalated  {(1-best['kept_frac']):.1%} of queries")
    else:
        print("\n  ⚠ No threshold stays within 2 points of judge-only accuracy.")
        print("    The gate is not reliable enough at any confidence level.")

    # ── Q3: ensemble headroom ─────────────────────────────────────
    print("\n" + "-" * 74)
    print("Q3  Error overlap — is there ensemble headroom?")
    print("-" * 74)
    both_right = (gate_ok & judge_ok).mean()
    gate_only = (gate_ok & ~judge_ok).mean()
    judge_only = (~gate_ok & judge_ok).mean()
    both_wrong = (~gate_ok & ~judge_ok).mean()
    oracle = 1.0 - both_wrong

    print(f"  both correct        {both_right:>7.1%}")
    print(f"  gate only correct   {gate_only:>7.1%}   <- signal the judge lacks")
    print(f"  judge only correct  {judge_only:>7.1%}")
    print(f"  both wrong          {both_wrong:>7.1%}")
    print(f"\n  Oracle ceiling (perfect selection): {oracle:.1%}")
    print(f"  Judge alone:                        {judge_acc:.1%}")
    print(f"  Ensemble headroom:                  {oracle - judge_acc:+.1%}")

    if gate_only > 0.05:
        print(f"\n  ✅ The gate is right on {gate_only:.1%} of cases the judge misses.")
        print("     Worth feeding P_S to the judge as a feature, or learning a selector.")
    else:
        print("\n  ❌ The gate rarely adds anything the judge lacks.")
        print("     Its correct answers are a near-subset of the judge's.")

    # Per-behaviour breakdown — where does each arm actually help?
    print("\n  Per-behaviour accuracy:")
    print(f"{'behaviour':<16} {'n':>5} {'gate':>8} {'judge':>8}")
    per_beh = {}
    for b in BEHAVIOURS:
        m = gold == b
        if m.sum() == 0:
            continue
        g, j = gate_ok[m].mean(), judge_ok[m].mean()
        per_beh[b] = {"n": int(m.sum()),
                      "gate": round(float(g), 4),
                      "judge": round(float(j), 4)}
        flag = "  <- gate wins" if g > j else ""
        print(f"{b:<16} {int(m.sum()):>5} {g:>7.1%} {j:>8.1%}{flag}")

    # ── Save ──────────────────────────────────────────────────────
    out = {
        "n": n,
        "gate_arm": gate_name,
        "judge_arm": judge_name,
        "gate_accuracy": round(float(gate_ok.mean()), 4),
        "judge_accuracy": round(float(judge_acc), 4),
        "confidence_correlation": round(r, 4),
        "bucket_spread": round(float(spread), 4),
        "cascade_feasible": bool(feasible),
        "buckets": bucket_stats,
        "sweep": rows,
        "best_operating_point": best,
        "error_overlap": {
            "both_correct": round(float(both_right), 4),
            "gate_only": round(float(gate_only), 4),
            "judge_only": round(float(judge_only), 4),
            "both_wrong": round(float(both_wrong), 4),
            "oracle_ceiling": round(float(oracle), 4),
            "ensemble_headroom": round(float(oracle - judge_acc), 4),
        },
        "per_behaviour": per_beh,
        "constants": {
            "gate_cost_per_1k": GATE_COST_PER_1K,
            "judge_cost_per_1k": JUDGE_COST_PER_1K,
            "gate_p50_ms": GATE_P50_MS,
            "judge_p50_ms": JUDGE_P50_MS,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
