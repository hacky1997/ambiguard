#!/usr/bin/env python3
"""
Section 9 Test — LLM Judge Log-Probability Scoring on Golden Gate Dataset.

Tests CLAM-style log-probability scoring over ambiguous vs unambiguous completion tokens:
    log_p(AMBIGUOUS) - log_p(ANSWER)

Evaluates whether log-probability scoring recovers ambiguity signal obscured by default text generation.

USAGE
    python scripts/eval_llm_judge_logprobs.py --n 300
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GOLDEN = Path("eval/datasets/golden_gate.jsonl")
_OUT = Path("eval/results/llm_judge_logprobs.json")
_SEED = 42

_JUDGE_PROMPT_BINARY: str = """You are an ambiguity classifier. Given a question and context, decide the routing behaviour.

Rules:
- ANSWER: The question has a single clear answer given the context.
- AMBIGUOUS: The question is ambiguous or has multiple valid interpretations.

Question: {question}
Context: {context}

Respond with exactly one word: ANSWER or AMBIGUOUS."""


def _auc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def boot_auc(s: np.ndarray, y: np.ndarray, n_boot: int = 2_000):
    base = _auc(s, y)
    rng = np.random.default_rng(_SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_auc(s[idx], y[idx]))
    if not vals:
        return base, base, base
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    api_key = os.getenv("AMBIGUARD_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        raise SystemExit("Error: Valid AMBIGUARD_OPENAI_API_KEY required to run logprob evaluation.")

    import openai

    client = openai.OpenAI(api_key=api_key)

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[: args.n]

    logger.info("Evaluating LLM Judge logprobs on %d rows using %s...", len(rows), args.model)

    log_diffs: list[float] = []
    gold_labels: list[int] = []
    generated_preds: list[int] = []

    for i, r in enumerate(rows):
        prompt = _JUDGE_PROMPT_BINARY.format(question=r["question"], context=r["context"])
        gold = 0 if r["expected_behaviour"] == "ANSWER" else 1
        gold_labels.append(gold)

        res = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )

        content = res.choices[0].message.content.strip().upper()
        generated_preds.append(1 if "AMBIGUOUS" in content else 0)

        # Extract logprobs for first token
        first_token_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        lp_map = {item.token.strip().upper(): item.logprob for item in first_token_logprobs}

        lp_ambig = lp_map.get("AMBIGUOUS", -20.0)
        lp_ans = lp_map.get("ANSWER", -20.0)

        # Score = log_p(AMBIGUOUS) - log_p(ANSWER)
        score = lp_ambig - lp_ans
        log_diffs.append(score)

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            logger.info("Processed %d/%d rows...", i + 1, len(rows))

    s_arr = np.array(log_diffs, dtype=float)
    y_arr = np.array(gold_labels, dtype=int)
    g_arr = np.array(generated_preds, dtype=int)

    gen_acc = float(np.mean(g_arr == y_arr))
    gen_auc = _auc(g_arr.astype(float), y_arr)

    lp_auc, lp_lo, lp_hi = boot_auc(s_arr, y_arr)

    print("\n" + "=" * 78)
    print(f"CATA SECTION 9 TEST — LLM JUDGE LOGPROB vs GENERATION ({args.model}, n={len(rows)})")
    print("=" * 78)
    print(f"Greedy Generation Accuracy:  {gen_acc:.1%}")
    print(f"Greedy Generation AUC:       {gen_auc:.3f}")
    print(f"Log-Prob Score AUC (CLAM):   {lp_auc:.3f} [{lp_lo:.3f}, {lp_hi:.3f}]")
    print("-" * 78)

    out = {
        "model": args.model,
        "n": len(rows),
        "greedy_gen_acc": round(gen_acc, 4),
        "greedy_gen_auc": round(gen_auc, 4),
        "logprob_auc": round(lp_auc, 4),
        "logprob_auc_ci95": [round(lp_lo, 4), round(lp_hi, 4)],
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Saved -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
