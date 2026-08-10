#!/usr/bin/env python3
"""
Re-score answer diversity with CORRECT clustering. Uses cached samples only.

THE BUG BEING FIXED
-------------------
The clustering in `answer_diversity_teacher.py` was labelled single-link but was
not single-link:

    for i in range(n):
        if assigned[i] != -1: continue
        assigned[i] = cluster
        for j in range(i + 1, n):
            if assigned[j] == -1 and (1 - sim[i, j]) < threshold:
                assigned[j] = cluster
        cluster += 1

Each point is compared only against the FIRST unassigned point of a group, never
transitively. If A~B and B~C but A is far from C, true single-link merges all
three; this splits them. Worse, the result depends on iteration order, and
iteration order is the order the samples came back from the API — effectively
random.

So `n_clusters` was a partly-random function of sampling order. That plausibly
explains why AUC *dropped* from 0.729 to 0.611 when samples went from 5 to 10:
more points means more chances for the greedy pass to fragment a true cluster.

WHAT THIS SCRIPT DOES
---------------------
Re-scores the cached completions with three correct linkage methods, plus the
buggy one for direct comparison:

    buggy_greedy      the original, for reference
    single_link       transitive closure at the distance threshold
    complete_link     all pairs within threshold (stricter, no chaining)
    average_link      scipy hierarchical, average linkage

Single-link is prone to chaining — one bridging answer merges two real clusters.
Complete-link is prone to over-splitting. Reporting all three shows whether the
signal depends on that choice, which is itself diagnostic: a result that survives
only one linkage rule is fragile.

Also sweeps the distance threshold, because 0.35 was a guess and a statistic that
only works at one threshold is not a finding.

NO API CALLS. Reads eval/results/answer_samples_cache.json.

USAGE
    python scripts/rescore_answer_diversity.py --samples 10
    python scripts/rescore_answer_diversity.py --samples 5
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

_GOLDEN = Path("eval/datasets/golden_gate.jsonl")
_CACHE = Path("eval/results/answer_samples_cache.json")
_OUT = Path("eval/results/answer_diversity_rescored.json")
_SEED = 42

# A statistic that is nonzero on fewer than this fraction of rows cannot carry a
# meaningful AUC — it is a handful of rows driving the whole rank statistic.
_MIN_NONZERO_RATE = 0.05


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


def paired_delta(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int = 2_000):
    rng = np.random.default_rng(_SEED)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(_auc(a[idx], y[idx]) - _auc(b[idx], y[idx]))
    if not d:
        return 0.0, 0.0, 0.0
    return (float(np.mean(d)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


# ── Clustering methods ────────────────────────────────────────────
def cluster_buggy_greedy(D: np.ndarray, thr: float) -> int:
    """The original implementation. Kept for comparison only."""
    n = len(D)
    assigned = [-1] * n
    c = 0
    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = c
        for j in range(i + 1, n):
            if assigned[j] == -1 and D[i, j] < thr:
                assigned[j] = c
        c += 1
    return c


def cluster_single_link(D: np.ndarray, thr: float) -> int:
    """Transitive closure: connected components of the threshold graph."""
    n = len(D)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(n)})


def cluster_complete_link(D: np.ndarray, thr: float) -> int:
    """Agglomerative with complete linkage — merge only if ALL pairs are close."""
    n = len(D)
    clusters: list[list[int]] = [[i] for i in range(n)]
    merged = True
    while merged:
        merged = False
        best, bi, bj = None, -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                worst = max(D[i, j] for i in clusters[a] for j in clusters[b])
                if worst < thr and (best is None or worst < best):
                    best, bi, bj = worst, a, b
        if best is not None:
            clusters[bi] = clusters[bi] + clusters[bj]
            clusters.pop(bj)
            merged = True
    return len(clusters)


def cluster_average_link(D: np.ndarray, thr: float) -> int:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return -1
    if len(D) < 2:
        return 1
    Z = linkage(squareform(D, checks=False), method="average")
    return int(fcluster(Z, t=thr, criterion="distance").max())


METHODS = {
    "buggy_greedy": cluster_buggy_greedy,
    "single_link": cluster_single_link,
    "complete_link": cluster_complete_link,
    "average_link": cluster_average_link,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not _CACHE.exists():
        raise SystemExit(f"{_CACHE} missing — run answer_diversity_teacher.py first.")

    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[: args.n]

    with open(_CACHE, encoding="utf-8") as f:
        cache = json.load(f)

    logger.info("Loading LaBSE ...")
    encoder = SentenceTransformer("sentence-transformers/LaBSE")

    # ── Embed cached answers once ─────────────────────────────────
    D_all: list[np.ndarray] = []
    gold: list[int] = []
    missing = 0
    for r in rows:
        key = f"{r['id']}:{args.samples}:{args.temperature}"
        if key not in cache:
            missing += 1
            continue
        answers = cache[key]["answers"]
        embs = encoder.encode(answers, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        D = np.clip(1.0 - (embs @ embs.T), 0.0, 2.0)
        D_all.append(D)
        gold.append(0 if r["expected_behaviour"] == "ANSWER" else 1)

    if missing:
        logger.warning("%d rows missing from cache at samples=%d",
                       missing, args.samples)
    gold_arr = np.asarray(gold)
    logger.info("Scoring %d rows  (%d ambiguous / %d unambiguous)",
                len(D_all), int(gold_arr.sum()), int((1 - gold_arr).sum()))

    # ── Sweep methods x thresholds ────────────────────────────────
    print("\n" + "=" * 82)
    print(f"CLUSTERING RESCORE — samples={args.samples}, n={len(D_all)}")
    print("=" * 82)
    print(f"{'method':<16} {'thr':>5} {'AUC':>7} {'CI95':>18} "
          f"{'mean k':>7} {'>1 clust':>9}")
    print("-" * 82)

    results: dict[str, Any] = {}
    best = {"auc": -1.0, "method": None, "thr": None, "lo": 0.0}

    for mname, fn in METHODS.items():
        for thr in args.thresholds:
            ks = np.array([fn(D, thr) for D in D_all], dtype=float)
            if ks.min() < 0:
                continue  # scipy unavailable
            multi_rate = float(np.mean(ks > 1))
            if ks.std() == 0:
                print(f"{mname:<16} {thr:>5.2f} {'constant':>7}")
                continue
            a, lo, hi = boot_auc(ks, gold_arr)
            key = f"{mname}@{thr:.2f}"
            results[key] = {
                "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "mean_k": round(float(ks.mean()), 3),
                "multi_cluster_rate": round(multi_rate, 4),
            }
            flag = ""
            if multi_rate < _MIN_NONZERO_RATE:
                flag = "  <- too few multi-cluster rows to trust"
            elif a > best["auc"] and mname != "buggy_greedy":
                best = {"auc": a, "method": mname, "thr": thr, "lo": lo}
            print(f"{mname:<16} {thr:>5.2f} {a:>7.3f} "
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18} {ks.mean():>7.2f} "
                  f"{multi_rate:>8.1%}{flag}")
        print()

    # ── Is the corrected version better than the buggy one? ───────
    if best["method"]:
        fixed = np.array([METHODS[best["method"]](D, best["thr"]) for D in D_all],
                         dtype=float)
        buggy = np.array([cluster_buggy_greedy(D, 0.35) for D in D_all], dtype=float)
        d, dlo, dhi = paired_delta(fixed, buggy, gold_arr)

        print("=" * 82)
        print(f"  best corrected: {best['method']} @ {best['thr']:.2f}  "
              f"AUC {best['auc']:.3f}")
        print(f"  vs buggy_greedy @ 0.35: paired Δ {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        if dlo > 0:
            print("  ✅ The clustering bug was suppressing real signal.")
        elif dhi < 0:
            print("  ⚠ The buggy version scored HIGHER — its apparent signal was an")
            print("    artefact of order-dependent fragmentation, not a real effect.")
        else:
            print("  ➖ No difference. The bug was not what drove the earlier result.")

        # Distribution table for the best configuration.
        print("\n  cluster-count distribution, best configuration:")
        for lab, name in ((0, "ANSWER     "), (1, "AMBIGUOUS  ")):
            c = Counter(fixed[gold_arr == lab].astype(int).tolist())
            print(f"    {name} {dict(sorted(c.items()))}")

        sep = abs(fixed[gold_arr == 1].mean() - fixed[gold_arr == 0].mean())
        print(f"\n  mean k: ambiguous {fixed[gold_arr==1].mean():.3f}  "
              f"unambiguous {fixed[gold_arr==0].mean():.3f}  (gap {sep:.3f})")
        if sep < 0.15:
            print("  The two classes produce almost the same number of answer")
            print("  clusters. Whatever AUC appears here is driven by very few rows.")

    out = {
        "samples": args.samples, "n": len(D_all),
        "thresholds_swept": args.thresholds,
        "results": results,
        "best": best,
        "note": ("buggy_greedy is the original non-transitive implementation, "
                 "retained only for comparison. It is not a valid clustering."),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
