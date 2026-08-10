#!/usr/bin/env python3
"""
Graph-aware routing over the center manifold.

THE OBSERVATION THIS IS BUILT ON
--------------------------------
Centre assignment accuracy is 71.8%. Behaviour accuracy is 45.8%.

The model identifies the correct cluster 7 times in 10, then the decision rule
destroys 26 points converting that distribution into a label. Every fix attempted
so far — grid search, macro-F1 objective, temperature scaling — replaced one
hand-designed scalar function with another. None addressed the actual defect.

THE DEFECT
----------
Spectral clustering builds an affinity matrix, extracts K clusters, and then the
affinity is discarded. Silhouette is 0.03–0.04: these clusters heavily overlap.
They are not a partition. They are a graph with weighted edges.

The current policy uses `second_mass` — how much probability sits on the runner-up
center. It is blind to WHICH center holds it.

    P = [0.35, 0.30, 0.15, 0.12, 0.08]   second_mass = 0.30
    P = [0.35, 0.05, 0.30, 0.15, 0.15]   second_mass = 0.30

Identical under the current rule. But if centers 1 and 2 are adjacent — nearly the
same semantic region — the first is a confident question whose mass happens to
straddle a cluster seam. If centers 1 and 3 are distant, the second is a genuinely
bi-modal reading. Same statistic, opposite meaning.

**Ambiguity is mass spanning a HIGH-DISTANCE edge, not mass spanning any edge.**

THE FORMULATION
---------------
Treat the K centers as nodes in a weighted graph. Edge weight w_ij = cosine
distance between centroids i and j. Then define dispersion as the expected
pairwise distance under the predicted distribution:

    D(P) = sum_{i<j} P_i * P_j * w_ij

This is a quadratic form P^T W P — the graph-Laplacian view of how spread the
mass is over the manifold, rather than over the index set. Flat-but-local
distributions score low. Peaked-but-split-across-distant-centers score high.
Shannon entropy cannot distinguish these because it has no notion of which
outcomes are near which.

Five variants are scored here, each isolating one aspect:

    entropy          current baseline (geometry-blind)
    second_mass      current baseline (geometry-blind)
    dispersion       P^T W P — the core proposal
    top2_distance    w_ij between the top two centers, weighted by their mass
    laplacian        P^T L P where L is the graph Laplacian, a smoothness penalty

EDGE CASES HANDLED
------------------
  * Degenerate centroids — if all pairwise distances are near-equal, the graph
    carries no information and dispersion reduces to a monotone function of
    entropy. The script detects this and says so rather than reporting a
    meaningless AUC.
  * Near-uniform P — every statistic saturates. Reported separately.
  * Single dominant center — dispersion goes to zero regardless of geometry.
  * Anisotropy — LaBSE embeddings occupy a narrow cone, so raw cosine distances
    compress into a small range. Both raw and rank-normalised distance matrices
    are scored.
  * Asymmetric ambiguity — CLARIFY and ALTERNATIVES may live at different
    dispersion scales. Per-class breakdown is reported, not just a global AUC.

NO RETRAINING. NO API CALLS. Reads the existing checkpoint and dataset.

USAGE
    python scripts/graph_routing.py --n 600
    python scripts/graph_routing.py --n 600 --centroids checkpoints/centroids.npy
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
_OUT = Path("eval/results/graph_routing.json")
_SEED = 42


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
    """ROC-AUC with bootstrap CI. labels: 1 = ambiguous."""
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


def boot_paired_auc_delta(scores: np.ndarray, base_scores: np.ndarray, labels: np.ndarray,
                         n_boot: int = 2_000) -> tuple[float, float, float, float, float, float]:
    """ROC-AUC and paired AUC delta with bootstrap CIs on identical resamples."""
    base_auc = _auc(scores, labels)
    ref_auc = _auc(base_scores, labels)
    delta_main = base_auc - ref_auc

    rng = np.random.default_rng(_SEED)
    auc_vals = []
    delta_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        a_s = _auc(scores[idx], labels[idx])
        a_b = _auc(base_scores[idx], labels[idx])
        auc_vals.append(a_s)
        delta_vals.append(a_s - a_b)

    if not auc_vals:
        return base_auc, base_auc, base_auc, delta_main, delta_main, delta_main

    return (
        base_auc,
        float(np.percentile(auc_vals, 2.5)),
        float(np.percentile(auc_vals, 97.5)),
        delta_main,
        float(np.percentile(delta_vals, 2.5)),
        float(np.percentile(delta_vals, 97.5)),
    )


# ── Graph statistics over the center manifold ────────────────────
def dispersion(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Expected pairwise distance under P.  D = sum_{i<j} P_i P_j W_ij.

    Vectorised as the quadratic form P W P^T, halved because W is symmetric and
    the diagonal is zero.
    """
    return 0.5 * np.einsum("ni,ij,nj->n", P, W, P)


def top2_distance(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Distance between the top two centers, scaled by their combined mass.

    A sharper, more interpretable version of dispersion: it asks only whether the
    two leading readings are semantically far apart.
    """
    order = np.argsort(-P, axis=1)
    i, j = order[:, 0], order[:, 1]
    return W[i, j] * (P[np.arange(len(P)), i] + P[np.arange(len(P)), j])


def laplacian_energy(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """P^T L P with L = D - W. Low when mass sits on well-connected centers."""
    L = np.diag(W.sum(axis=1)) - W
    return np.einsum("ni,ij,nj->n", P, L, P)


def entropy_nats(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(P + 1e-12)).sum(axis=1)


def second_mass(P: np.ndarray) -> np.ndarray:
    return np.sort(P, axis=1)[:, -2]


def build_W(centroids: np.ndarray, rank_normalise: bool = False) -> np.ndarray:
    """Pairwise cosine distance between centroids, optionally rank-normalised.

    Rank normalisation addresses anisotropy: LaBSE embeddings occupy a narrow
    cone, so raw cosine distances compress into a small range and the graph
    carries less signal than it should. Ranking restores spread.
    """
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    W = 1.0 - (c @ c.T)
    np.fill_diagonal(W, 0.0)
    if rank_normalise:
        iu = np.triu_indices_from(W, k=1)
        vals = W[iu]
        ranks = np.argsort(np.argsort(vals)).astype(float)
        ranks = ranks / max(1.0, ranks.max())
        W = np.zeros_like(W)
        W[iu] = ranks
        W = W + W.T
    return W


def check_geometry(W: np.ndarray) -> dict[str, Any]:
    """Is the center graph informative at all?

    If every pairwise distance is nearly identical, the graph adds nothing and
    dispersion collapses to a monotone function of entropy. Better to detect that
    than to report an AUC that looks like a result.
    """
    iu = np.triu_indices_from(W, k=1)
    d = W[iu]
    spread = float(d.max() - d.min())
    rel = float(spread / (d.mean() + 1e-12))
    degenerate = rel < 0.15
    return {
        "pairwise_distances": [round(float(x), 4) for x in d],
        "min": round(float(d.min()), 4),
        "max": round(float(d.max()), 4),
        "mean": round(float(d.mean()), 4),
        "relative_spread": round(rel, 4),
        "degenerate": bool(degenerate),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--centroids", default=None,
                    help="Path to centroids .npy. Defaults to the checkpoint's.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch

    from app.gate.centerdistill import CenterDistillGate
    from app.settings import get_settings

    settings = get_settings()
    gate = CenterDistillGate(settings)
    if getattr(gate, "using_fallback", False):
        logger.error("Gate is in heuristic fallback — results meaningless.")
        return 1

    # ── Centroids: prefer an explicit file, else reconstruct from the head ──
    if args.centroids and Path(args.centroids).exists():
        centroids = np.load(args.centroids)
        source = args.centroids
        logger.info("Centroids from %s: %s", source, centroids.shape)
    else:
        # The center head's weight rows ARE the learned center directions —
        # logit_k = w_k . h, so w_k is the direction that maximally activates
        # center k. Using them keeps the graph consistent with what the model
        # actually learned, rather than with the original clustering.
        ckpt = Path(settings.gate_checkpoint_path) / "centerdistill_full.pt"
        d = torch.load(ckpt, map_location="cpu")
        centroids = d["state_dict"]["center_head.weight"].numpy()
        source = "center_head.weight (learned center directions)"
        logger.info("Centroids from %s: %s", source, centroids.shape)

    geo_raw = check_geometry(build_W(centroids, False))
    geo_rank = check_geometry(build_W(centroids, True))

    print("\n" + "=" * 72)
    print("CENTER GRAPH GEOMETRY")
    print("=" * 72)
    print(f"  source: {source}")
    print(f"  pairwise distances: {geo_raw['pairwise_distances']}")
    print(f"  min {geo_raw['min']}  max {geo_raw['max']}  "
          f"mean {geo_raw['mean']}  relative spread {geo_raw['relative_spread']}")
    if geo_raw["degenerate"]:
        print("  ⚠ Distances are near-uniform. The graph carries little")
        print("    information and dispersion will track entropy closely.")
        print("    Rank-normalised variant is scored as a fallback.")
    else:
        print("  ✅ Distances vary meaningfully — the graph is informative.")

    # ── Collect predictions ───────────────────────────────────────
    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[: args.n]
    logger.info("Scoring %d rows ...", len(rows))

    P, gold = [], []
    for i, r in enumerate(rows):
        d = gate.decide(r["question"], r.get("context"))
        P.append(d["center_distribution"])
        gold.append(0 if r["expected_behaviour"] == "ANSWER" else 1)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d", i + 1, len(rows))

    P = np.asarray(P, dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    gold = np.asarray(gold)
    logger.info("Gold: %d ambiguous / %d unambiguous",
                int(gold.sum()), int((1 - gold).sum()))

    # ── Score every statistic ─────────────────────────────────────
    W_raw = build_W(centroids, False)
    W_rank = build_W(centroids, True)

    stats: dict[str, np.ndarray] = {
        "entropy (baseline)": entropy_nats(P),
        "second_mass (baseline)": second_mass(P),
        "dispersion": dispersion(P, W_raw),
        "dispersion (rank-norm)": dispersion(P, W_rank),
        "top2_distance": top2_distance(P, W_raw),
        "top2_distance (rank-norm)": top2_distance(P, W_rank),
        "laplacian_energy": laplacian_energy(P, W_raw),
    }

    base_ent = stats["entropy (baseline)"]
    results: dict[str, Any] = {}
    print("\n" + "=" * 72)
    print(f"AMBIGUITY SIGNAL — n={len(rows)}, AUC vs gold")
    print("=" * 72)
    print(f"{'statistic':<28} {'AUC':>7} {'CI95':>18} {'Paired Δ CI95':>20}")
    print("-" * 72)
    for name, s in stats.items():
        a, lo, hi, d_val, d_lo, d_hi = boot_paired_auc_delta(s, base_ent, gold)
        results[name] = {
            "auc": round(a, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "paired_delta_auc": round(d_val, 4),
            "paired_delta_ci95": [round(d_lo, 4), round(d_hi, 4)],
        }
        marker = ""
        if "baseline" not in name and d_lo > 0.0:
            marker = "  <- paired Δ > 0"
        print(f"{name:<28} {a:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} {f'[{d_lo:+.3f}, {d_hi:+.3f}]':>20}{marker}")
    print("-" * 72)
    print("  AUC 0.5 = chance. Paired Δ CI95 is computed on identical resamples against entropy.")

    # ── Does geometry add anything over entropy? ──────────────────
    ent = stats["entropy (baseline)"]
    best_geo_name = max(
        (k for k in stats if "baseline" not in k),
        key=lambda k: results[k]["auc"],
    )
    best_geo = stats[best_geo_name]
    corr = float(np.corrcoef(ent, best_geo)[0, 1])
    best_res = results[best_geo_name]
    delta = best_res["paired_delta_auc"]
    p_lo, p_hi = best_res["paired_delta_ci95"]
    separated = p_lo > 0.0

    print(f"\n  best geometric statistic: {best_geo_name}")
    print(f"  corr with entropy: {corr:+.3f}")
    print(f"  paired AUC delta over entropy: {delta:+.3f} (95% CI [{p_lo:+.3f}, {p_hi:+.3f}])")
    if separated:
        print("  ✅ Geometry adds signal the flat statistics do not carry.")
        print("     Routing on the manifold beats routing on the index set.")
    elif abs(corr) > 0.95:
        print("  ❌ NULL. The geometric statistic is a near-linear function of")
        print("     entropy — the center graph is too flat to distinguish")
        print("     'mass on adjacent centers' from 'mass on distant centers'.")
    else:
        print("  ⚠ Paired delta CI includes zero. Directional gain, but not statistically separated. Report as null.")

    # ── Per-class: do CLARIFY and ALTERNATIVES sit at different scales? ──
    print("\n" + "-" * 72)
    print("Distribution of the best statistic by class")
    print("-" * 72)
    for lab, name in ((0, "unambiguous"), (1, "ambiguous")):
        v = best_geo[gold == lab]
        print(f"  {name:<14} n={len(v):<5} mean={v.mean():.4f}  "
              f"sd={v.std():.4f}  median={np.median(v):.4f}")

    # ── Saturation check: how many rows are near-uniform? ─────────
    near_uniform = float(np.mean(P.max(axis=1) < 1.5 / P.shape[1]))
    single_dom = float(np.mean(P.max(axis=1) > 0.6))
    print(f"\n  near-uniform rows (max P < {1.5/P.shape[1]:.2f}): {near_uniform:.1%}")
    print(f"  single-dominant rows (max P > 0.60): {single_dom:.1%}")
    if near_uniform > 0.5:
        print("  ⚠ Most rows are near-uniform. Every statistic saturates here and")
        print("    no policy over this distribution can separate the classes.")

    out = {
        "n": len(rows),
        "centroid_source": source,
        "geometry_raw": geo_raw,
        "geometry_rank_normalised": geo_rank,
        "statistics": results,
        "best_geometric": best_geo_name,
        "corr_with_entropy": round(corr, 4),
        "auc_delta_over_baseline": round(delta, 4),
        "ci_separated": bool(separated),
        "near_uniform_frac": round(near_uniform, 4),
        "single_dominant_frac": round(single_dom, 4),
        "gold_dist": dict(Counter(gold.tolist())),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
