#!/usr/bin/env python3
"""
Answer-diversity teacher: measure ambiguity from the answer side.

THE REFRAMING
-------------
Every previous attempt asked "is this question ambiguous?" by looking at the
QUESTION. Question embeddings, cluster assignments, entropy over centers,
translation displacement. All null.

But ambiguity is not a property visible in a question. It is a property of the
answer distribution:

    A question is ambiguous iff competent answerers produce DIFFERENT answers.

That is the definition, not a proxy for it. The original teacher — spectral
clustering over question embeddings, silhouette 0.03–0.04 — was attempting to
infer it from the wrong side. The student inherited a ceiling that was never
about the student.

THE MEASUREMENT
---------------
Sample N answers per question at temperature > 0. Embed them. Measure semantic
spread. A question with one reading produces answers that cluster tightly. A
question with two readings produces a bimodal answer set.

Five statistics, each isolating a different failure mode:

    mean_pairwise_dist   average cosine distance between sampled answers
    max_pairwise_dist    the furthest two answers — catches bimodality that
                         averaging washes out
    centroid_spread      mean distance to the answer centroid
    n_clusters           distinct answer clusters at a distance threshold
    disagreement_ratio   fraction of answer pairs beyond the threshold

WHY THIS IS A TEACHER, NOT A DETECTOR
-------------------------------------
If answer-spread predicts the gold labels well, it becomes a supervision signal:
a continuous, per-example target that a cheap student can be distilled against.
That is a different research arc from tuning thresholds on a weak teacher — the
teacher itself is replaced.

It is also expensive at inference (N generations per query), which is exactly
why distilling it into a 460 ms free classifier is worth doing.

CONTROLS BUILT IN
-----------------
  * Temperature-0 baseline: if spread at T=0 is already high, the model is just
    unstable and the signal is noise, not ambiguity.
  * Answer-length confound: long answers may spread more regardless of ambiguity.
    Correlation with length is reported.
  * Refusal/hedge detection: "it depends", "could refer to" are the model
    verbalising ambiguity, which is a different signal — counted separately.
  * Every statistic is scored against the gate's own confidence as a baseline.

COST
    n=300, 5 samples, gpt-4o-mini short answers: roughly $3-5.
    Run --dry-run first to see the exact estimate.

USAGE
    python scripts/answer_diversity_teacher.py --n 300 --samples 5 --dry-run
    python scripts/answer_diversity_teacher.py --n 300 --samples 5 --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GOLDEN = Path("eval/datasets/golden_gate.jsonl")
_OUT = Path("eval/results/answer_diversity.json")
_CACHE = Path("eval/results/answer_samples_cache.json")
_SEED = 42

# Two answers are "different readings" beyond this cosine distance.
_CLUSTER_THRESHOLD = 0.35

_HEDGE_PATTERNS = [
    r"\bit depends\b", r"\bcould (?:refer|mean|be)\b", r"\bambiguous\b",
    r"\bunclear\b", r"\bwhich (?:one|of)\b", r"\bnot specified\b",
    r"\bmultiple\b.*\b(?:interpretations|readings|answers)\b",
]

_ANSWER_PROMPT = """Answer the question using only the context. Be concise — one or two sentences.
If the context supports more than one answer, give the single answer you consider most likely.
Do not explain your reasoning. Do not mention ambiguity.

Context: {context}

Question: {question}

Answer:"""


def _auc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    
    order = np.argsort(all_scores)
    all_scores = all_scores[order]
    all_labels = all_labels[order]
    
    distinct_value_indices = np.where(np.diff(all_scores))[0]
    threshold_idxs = np.concatenate([[-1], distinct_value_indices, [len(all_scores) - 1]])
    ranks = np.empty(len(all_scores), dtype=float)
    for i in range(len(threshold_idxs) - 1):
        start = threshold_idxs[i] + 1
        end = threshold_idxs[i + 1] + 1
        ranks[start:end] = (start + end + 1) / 2.0

    pos_ranks = ranks[all_labels == 1]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boot_auc(scores: np.ndarray, labels: np.ndarray,
             n_boot: int = 2_000) -> tuple[float, float, float]:
    base = _auc(scores, labels)
    rng = np.random.default_rng(_SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(_auc(scores[idx], labels[idx]))
    if not vals:
        return base, base, base
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_auc_delta(a: np.ndarray, b: np.ndarray, labels: np.ndarray,
                     n_boot: int = 2_000) -> tuple[float, float, float]:
    """Bootstrap the DIFFERENCE on identical resamples."""
    rng = np.random.default_rng(_SEED)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        if len(np.unique(labels[idx])) < 2:
            continue
        deltas.append(_auc(a[idx], labels[idx]) - _auc(b[idx], labels[idx]))
    if not deltas:
        return 0.0, 0.0, 0.0
    return (float(np.mean(deltas)),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def diversity_stats(embs: np.ndarray) -> dict[str, float]:
    """Spread statistics over one question's sampled answers.

    embs: (n_samples, dim), L2-normalised.
    """
    n = len(embs)
    if n < 2:
        return {k: 0.0 for k in
                ("mean_pairwise", "max_pairwise", "centroid_spread",
                 "n_clusters", "disagreement_ratio")}

    sim = embs @ embs.T
    iu = np.triu_indices(n, k=1)
    dists = 1.0 - sim[iu]

    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12

    # Greedy single-link clustering at the threshold — cheap and adequate for n<=8.
    assigned = [-1] * n
    cluster = 0
    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster
        for j in range(i + 1, n):
            if assigned[j] == -1 and (1.0 - sim[i, j]) < _CLUSTER_THRESHOLD:
                assigned[j] = cluster
        cluster += 1

    return {
        "mean_pairwise": float(dists.mean()),
        "max_pairwise": float(dists.max()),
        "centroid_spread": float(np.mean(1.0 - embs @ centroid)),
        "n_clusters": float(cluster),
        "disagreement_ratio": float(np.mean(dists > _CLUSTER_THRESHOLD)),
    }


def count_hedges(answers: list[str]) -> int:
    return sum(
        1 for a in answers
        if any(re.search(p, a, flags=re.I) for p in _HEDGE_PATTERNS)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[: args.n]
    gold = np.array([0 if r["expected_behaviour"] == "ANSWER" else 1 for r in rows])

    # +1 per row for the temperature-0 control sample.
    total_calls = len(rows) * (args.samples + 1)
    est_cost = total_calls * 0.0004

    print("=" * 72)
    print(f"  rows {len(rows)}   samples {args.samples} @ T={args.temperature}")
    print(f"  plus 1 control sample per row at T=0")
    print(f"  total calls ~{total_calls}   estimated cost ~${est_cost:.2f}")
    print("=" * 72)

    if args.dry_run:
        print("\nSample prompt:\n")
        print(_ANSWER_PROMPT.format(
            context=rows[0].get("context", "")[:200] + " ...",
            question=rows[0]["question"]))
        print("\nDry run — no calls made.")
        return 0

    if not args.yes:
        print("\nPass --yes to proceed.")
        return 0

    from sentence_transformers import SentenceTransformer

    from app.gate.centerdistill import CenterDistillGate
    from app.llm.registry import get_provider
    from app.settings import get_settings

    settings = get_settings()
    if settings.llm_provider == "mock":
        logger.error("LLM_PROVIDER is 'mock'. A mock provider produces identical "
                     "answers and cannot exhibit answer diversity.")
        return 1

    provider = get_provider(settings.llm_provider,
                            api_key=settings.openai_api_key,
                            model=settings.openai_model)
    logger.info("Provider: %s", provider.model_name)

    cache: dict[str, Any] = {}
    if _CACHE.exists():
        with open(_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Cache: %d entries", len(cache))

    logger.info("Loading LaBSE ...")
    encoder = SentenceTransformer("sentence-transformers/LaBSE")

    gate = CenterDistillGate(settings)

    per_row: list[dict[str, Any]] = []
    spent = 0.0

    for i, r in enumerate(rows):
        prompt = _ANSWER_PROMPT.format(
            context=(r.get("context") or "")[:1200], question=r["question"])
        key = f"{r['id']}:{args.samples}:{args.temperature}"

        if key in cache:
            answers, control = cache[key]["answers"], cache[key]["control"]
        else:
            answers = []
            for _ in range(args.samples):
                resp = provider.complete(prompt, temperature=args.temperature)
                answers.append(resp["content"].strip())
                spent += resp.get("cost_usd", 0.0)
            resp0 = provider.complete(prompt, temperature=0.0)
            control = resp0["content"].strip()
            spent += resp0.get("cost_usd", 0.0)
            cache[key] = {"answers": answers, "control": control}

        embs = encoder.encode(answers, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        stats = diversity_stats(embs)
        stats["hedges"] = float(count_hedges(answers))
        stats["mean_answer_len"] = float(np.mean([len(a) for a in answers]))
        stats["gate_uncertainty"] = 1.0 - gate.decide(
            r["question"], r.get("context"))["max_prob"]
        per_row.append(stats)

        if (i + 1) % 25 == 0:
            logger.info("  %d/%d  spent ~$%.2f", i + 1, len(rows), spent)
            with open(_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    # ── Score every statistic ─────────────────────────────────────
    names = ["mean_pairwise", "max_pairwise", "centroid_spread",
             "n_clusters", "disagreement_ratio", "hedges"]
    baseline = np.array([p["gate_uncertainty"] for p in per_row])
    b_auc, b_lo, b_hi = boot_auc(baseline, gold)

    results: dict[str, Any] = {}
    print("\n" + "=" * 78)
    print(f"ANSWER-DIVERSITY TEACHER — n={len(rows)}, {args.samples} samples "
          f"@ T={args.temperature}")
    print("=" * 78)
    print(f"{'statistic':<24} {'AUC':>7} {'CI95':>18} {'paired Δ vs gate':>22}")
    print("-" * 78)
    print(f"{'gate uncertainty':<24} {b_auc:>7.3f} "
          f"{f'[{b_lo:.3f}, {b_hi:.3f}]':>18} {'(baseline)':>22}")

    best_name, best_lo = None, -1.0
    for name in names:
        s = np.array([p[name] for p in per_row], dtype=float)
        if s.std() == 0:
            print(f"{name:<24} {'constant':>7}")
            continue
        a, lo, hi = boot_auc(s, gold)
        d, dlo, dhi = paired_auc_delta(s, baseline, gold)
        results[name] = {
            "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "paired_delta_vs_gate": round(d, 4),
            "paired_delta_ci95": [round(dlo, 4), round(dhi, 4)],
            "beats_gate": bool(dlo > 0),
        }
        mark = "  <- beats gate" if dlo > 0 else ""
        print(f"{name:<24} {a:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
              f"{f'{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]':>22}{mark}")
        if dlo > best_lo:
            best_name, best_lo = name, dlo
    print("-" * 78)

    # ── Confound checks ───────────────────────────────────────────
    lens = np.array([p["mean_answer_len"] for p in per_row])
    best = np.array([p[best_name] for p in per_row], dtype=float)
    len_corr = float(np.corrcoef(best, lens)[0, 1])
    hedge_rate = float(np.mean([p["hedges"] > 0 for p in per_row]))

    print("\nCONFOUND CHECKS")
    print(f"  corr(best statistic, answer length) = {len_corr:+.3f}")
    if abs(len_corr) > 0.5:
        print("    ⚠ Spread tracks answer length. May be measuring verbosity,")
        print("      not ambiguity.")
    print(f"  rows where the model hedged at least once: {hedge_rate:.1%}")
    print("    (hedging is the model VERBALISING ambiguity — a different signal)")

    print("\nVERDICT")
    if best_lo > 0:
        print(f"  ✅ '{best_name}' beats the gate's own confidence on identical")
        print("     resamples. Answer-side measurement carries signal the")
        print("     question-side classifier does not.")
        print("     Next: distil this as a continuous per-example target.")
    else:
        print("  ❌ NULL. Answer diversity does not predict the gold labels better")
        print("     than the existing gate. Combined with the earlier finding that")
        print("     context adds ~2.5 points over no context, this points at the")
        print("     labels rather than the models.")

    out = {
        "n": len(rows), "samples": args.samples, "temperature": args.temperature,
        "model": provider.model_name,
        "cost_usd": round(spent, 3),
        "gate_baseline": {"auc": round(b_auc, 4),
                          "ci95": [round(b_lo, 4), round(b_hi, 4)]},
        "statistics": results,
        "best": best_name,
        "answer_length_corr": round(len_corr, 4),
        "hedge_rate": round(hedge_rate, 4),
        "gold_dist": dict(Counter(gold.tolist())),
        "cluster_threshold": _CLUSTER_THRESHOLD,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved -> {_OUT}   (spent ~${spent:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
