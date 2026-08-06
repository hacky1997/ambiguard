#!/usr/bin/env python3
"""
Controlled leakage ablation.

CLAIM UNDER TEST
----------------
gpt-4o-mini scored 70.8% on the original golden_gate.jsonl and 53.3% on the
rebuilt one. The two datasets differ in several ways at once (context source,
formatting, label scheme), so the drop is currently *inferred*, not measured.

This script isolates the variable. It takes ONE dataset and produces three
variants that differ only in how the context is rendered:

  A. clean      real passage, uniform formatting                (control)
  B. formatted  same passage, but AMBIGUOUS rows get a bulleted
                sub-question list and ANSWER rows get a
                "The documented answer is: X" suffix            (the old leak)
  C. blind      question only, context replaced with a constant  (floor)

Same model, same prompt, same rows, same order. If B >> A, the formatting is
doing the work and the original benchmark measured presentation, not ambiguity.
If C is also high, the question text alone is diagnostic and the context is
irrelevant either way.

Cost: 3 x N calls. At N=200 with gpt-4o-mini this is roughly a dollar.

USAGE
    python scripts/leakage_ablation.py --n 200
    python scripts/leakage_ablation.py --n 200 --dry-run   # no API calls
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DATASET = Path("eval/datasets/golden_gate.jsonl")
_OUT = Path("eval/results/leakage_ablation.json")
_SEED = 42

_PROMPT = """You are an ambiguity classifier. Given a question and context, decide the routing behaviour.

Rules:
- ANSWER: The question has a single clear answer given the context.
- AMBIGUOUS: The question is ambiguous or has multiple valid interpretations.

Question: {question}
Context: {context}

Respond with exactly one word: ANSWER or AMBIGUOUS."""

_BLIND_CONTEXT = "No context provided."


def _parse(raw: str) -> str:
    u = raw.strip().upper()
    if "AMBIGUOUS" in u or "CLARIFY" in u or "ALTERNATIVES" in u:
        return "AMBIGUOUS"
    return "ANSWER"


def render_clean(row: dict[str, Any]) -> str:
    """Control: the real passage, unmodified."""
    return row["context"]


def render_formatted(row: dict[str, Any]) -> str:
    """Reintroduce the original formatting tell, and nothing else.

    The passage text is identical to the clean variant. Only the wrapper
    differs, and it differs BY CLASS — which is exactly the defect.
    """
    passage = row["context"]
    title = "Wikipedia Reference"
    if row["expected_behaviour"] == "ANSWER":
        return (
            f"Source Document ({title}): {passage} "
            f"The documented answer is stated above."
        )
    # Ambiguous rows got a bulleted sub-question list in the original.
    q = row["question"].rstrip("?")
    return (
        f"Source Document ({title}):\n{passage}\n"
        f"- For '{q} (first reading)': see above\n"
        f"- For '{q} (second reading)': see above"
    )


def render_blind(row: dict[str, Any]) -> str:
    """Floor: no context at all. Tests whether the question alone is diagnostic."""
    return _BLIND_CONTEXT


VARIANTS = {
    "clean": render_clean,
    "formatted": render_formatted,
    "blind": render_blind,
}


def boot_ci(correct: np.ndarray, n_boot: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(_SEED)
    b = np.array([correct[rng.integers(0, len(correct), len(correct))].mean()
                  for _ in range(n_boot)])
    return (float(correct.mean()),
            float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Rows per variant. 3 x n API calls total.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print rendered contexts, make no API calls.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = [json.loads(line) for line in open(_DATASET, encoding="utf-8")]
    labels = {r["expected_behaviour"] for r in rows}
    if labels != {"ANSWER", "AMBIGUOUS"}:
        logger.error("Expected binary labels, found %s", labels)
        return 1

    # Balanced sample, fixed seed, same rows across all three variants.
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["expected_behaviour"], []).append(r)
    per = args.n // 2
    rng = random.Random(_SEED)
    sample = []
    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        sample.extend(bucket[:per])
    rng.shuffle(sample)

    logger.info("Sample: %d rows  %s", len(sample),
                dict(Counter(r["expected_behaviour"] for r in sample)))

    if args.dry_run:
        for name, fn in VARIANTS.items():
            print(f"\n{'=' * 70}\nVARIANT: {name}\n{'=' * 70}")
            for r in sample[:2]:
                print(f"\n[{r['expected_behaviour']}] {r['question']}")
                print(fn(r)[:400])
        print("\nDry run — no API calls made.")
        return 0

    from app.llm.registry import get_provider
    from app.settings import get_settings

    settings = get_settings()
    if settings.llm_provider == "mock":
        logger.error(
            "LLM_PROVIDER is 'mock'. This ablation needs a real model — "
            "a mock provider cannot exhibit the leak."
        )
        return 1

    provider = get_provider(
        settings.llm_provider,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    logger.info("Model: %s", provider.model_name)

    gold = np.array([r["expected_behaviour"] for r in sample])
    results: dict[str, Any] = {}
    total_cost = 0.0

    for name, render in VARIANTS.items():
        logger.info("Running variant: %s", name)
        preds, cost = [], 0.0
        for i, row in enumerate(sample):
            resp = provider.complete(
                _PROMPT.format(question=row["question"], context=render(row)),
                temperature=0.0,
            )
            preds.append(_parse(resp["content"]))
            cost += resp.get("cost_usd", 0.0)
            if (i + 1) % 25 == 0:
                logger.info("  %d/%d", i + 1, len(sample))

        pred = np.array(preds)
        correct = pred == gold
        acc, lo, hi = boot_ci(correct)

        # Per-class recall, so a collapsed predictor is visible.
        recalls = {
            c: float((pred[gold == c] == c).mean())
            for c in ("ANSWER", "AMBIGUOUS")
        }
        results[name] = {
            "accuracy": round(acc, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "recall": {k: round(v, 4) for k, v in recalls.items()},
            "pred_dist": dict(Counter(preds)),
            "cost_usd": round(cost, 4),
        }
        total_cost += cost
        logger.info("  %s: %.1f%% [%.1f%%, %.1f%%]  cost $%.3f",
                    name, acc * 100, lo * 100, hi * 100, cost)

    # ── Report ────────────────────────────────────────────────────
    clean = results["clean"]
    fmt = results["formatted"]
    blind = results["blind"]
    delta = fmt["accuracy"] - clean["accuracy"]
    overlap = not (fmt["ci95"][0] > clean["ci95"][1] or
                   clean["ci95"][0] > fmt["ci95"][1])

    print("\n" + "=" * 74)
    print(f"LEAKAGE ABLATION — {provider.model_name}, n={len(sample)} per variant")
    print("=" * 74)
    print(f"{'variant':<14} {'accuracy':>10} {'CI95':>18} "
          f"{'rec ANSWER':>12} {'rec AMBIG':>11}")
    print("-" * 74)
    for name in ("clean", "formatted", "blind"):
        r = results[name]
        ci_str = f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]"
        print(f"{name:<14} {r['accuracy']:>9.1%} "
              f"{ci_str:>18} "
              f"{r['recall']['ANSWER']:>11.1%} {r['recall']['AMBIGUOUS']:>10.1%}")
    print("=" * 74)

    print(f"\n  formatted - clean = {delta:+.1%}")
    if delta > 0.05 and not overlap:
        print("  ✅ CONFIRMED. Class-dependent formatting inflates accuracy with")
        print("     no change to the underlying passage. The original benchmark")
        print("     measured presentation, not ambiguity.")
    elif delta > 0.05:
        print("  ⚠ Directionally consistent but CIs overlap. Increase --n.")
    else:
        print("  ❌ NOT CONFIRMED. Formatting does not explain the earlier gap.")
        print("     Something else differs between the two datasets — check the")
        print("     label scheme and context source.")

    print(f"\n  blind (question only) = {blind['accuracy']:.1%}")
    if blind["accuracy"] > clean["accuracy"] - 0.03:
        print("     Context adds little over the question alone. Worth stating:")
        print("     this task may not be solvable from the passage.")

    print(f"\n  total cost: ${total_cost:.2f}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump({
            "model": provider.model_name,
            "n_per_variant": len(sample),
            "seed": _SEED,
            "variants": results,
            "formatted_minus_clean": round(delta, 4),
            "ci_overlap": bool(overlap),
            "confirmed": bool(delta > 0.05 and not overlap),
            "total_cost_usd": round(total_cost, 4),
            "note": (
                "All three variants use identical rows and identical passage "
                "text. Only the wrapper formatting differs, and in the "
                "'formatted' variant it differs by class — reproducing the "
                "defect found in the original dataset builder."
            ),
        }, f, indent=2)
    print(f"\n✅ Saved → {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
