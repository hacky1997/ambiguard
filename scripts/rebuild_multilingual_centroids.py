#!/usr/bin/env python3
"""
Multilingual centroid rebuild — targets translation stability only.

THE PROBLEM
-----------
Measured stability of the gate's routing decision under translation:

    Spanish 67.3%  German 72.7%  Japanese 64.7%
    Arabic  66.0%  Hindi  66.7%  Swahili  63.3%     mean 66.8%

A third of decisions flip when the same question is rendered in another language.
Ambiguity is a property of meaning, so every flip is wrong on one side or the other.

THE LIKELY CAUSE
----------------
The 5 centers were induced by spectral clustering over 500 *English* LaBSE
embeddings. LaBSE aligns languages well but not exactly: a Spanish question lands
near — not on — its English twin. Combined with the known fact that most errors sit
within 0.02 of a decision threshold, a small embedding shift is enough to cross a
boundary.

The geometry is English-shaped. Every other language is approximated into it.

THE FIX TESTED HERE
-------------------
Re-induce the centers over the English pool AND its translations, so each centroid
sits at the multilingual centre of its cluster rather than the English one. No
student retraining — this only moves the reference points the teacher was built
from, then re-derives thresholds against them.

WHAT THIS DOES NOT FIX
----------------------
Nothing in the typological arm. The gate scores 0% on subject_drop and 6.7% on
currency because a 5-way cluster assignment over sentence embeddings has no
mechanism for syntactic parsing or world knowledge. Re-centering will not change
that, and this script does not claim otherwise.

DIAGNOSTICS RUN FIRST
---------------------
Before rebuilding, the script measures whether the fix can plausibly work:

  D1  Flip direction — do translated questions drift toward ANSWER or AMBIGUOUS?
      Systematic drift is a centroid problem. Random flipping is a margin problem
      and needs a different fix.
  D2  Flip magnitude — how far from a threshold do flipped rows sit? If they move
      from 0.6 to 0.2, the embedding is moving a lot and re-centering will not help.
  D3  Embedding displacement — mean cosine distance between an English question and
      its translation, per language. Sets an upper bound on what any centroid fix
      can achieve.

If D1 shows random flips and D2 shows large jumps, the script says so and stops
rather than producing centroids that will not help.

USAGE
    python scripts/rebuild_multilingual_centroids.py --diagnose-only
    python scripts/rebuild_multilingual_centroids.py --rebuild
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
_CACHE = Path("eval/results/translation_cache.json")
_STABILITY = Path("eval/results/crosslingual_stability.json")
_OUT_DIAG = Path("eval/results/centroid_diagnostics.json")
_OUT_CENTROIDS = Path("checkpoints/centroids_multilingual.npy")
_OUT_REPORT = Path("eval/results/centroid_rebuild.json")
_SEED = 42

LANGS = ["es", "de", "ja", "ar", "hi", "sw"]

# A flip is "boundary-driven" if the deciding statistic moved less than this.
_BOUNDARY_BAND = 0.05


def load_cache() -> dict[str, str]:
    if not _CACHE.exists():
        raise SystemExit(
            f"{_CACHE} not found. Run crosslingual_eval.py --arm stability first "
            "so the translations exist; this script reuses them rather than "
            "re-translating."
        )
    with open(_CACHE, encoding="utf-8") as f:
        return json.load(f)


def encode(texts: list[str], encoder: Any, bs: int = 64) -> np.ndarray:
    return encoder.encode(texts, batch_size=bs, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)


# ─────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────
def diagnose(gate: Any, encoder: Any, rows: list[dict], cache: dict) -> dict[str, Any]:
    questions = [r["question"] for r in rows]
    logger.info("Diagnosing on %d questions x %d languages", len(rows), len(LANGS))

    en_dec = [gate.decide(q, r.get("context")) for q, r in zip(questions, rows)]
    en_label = [d["behaviour"] for d in en_dec]
    en_max = np.array([d["max_prob"] for d in en_dec])
    logger.info("English decisions: %s", dict(Counter(en_label)))

    en_emb = encode(questions, encoder)

    per_lang: dict[str, Any] = {}
    all_dirs: Counter[str] = Counter()
    all_deltas: list[float] = []

    for lang in LANGS:
        translated = [cache.get(f"{lang}:{q}") for q in questions]
        missing = sum(1 for t in translated if t is None)
        if missing:
            logger.warning("%s: %d translations missing from cache — skipping those",
                           lang, missing)
        pairs = [(i, t) for i, t in enumerate(translated) if t]
        if not pairs:
            continue
        idx = [i for i, _ in pairs]
        texts = [t for _, t in pairs]

        dec = [gate.decide(t, rows[i].get("context")) for i, t in pairs]
        lab = [d["behaviour"] for d in dec]
        mx = np.array([d["max_prob"] for d in dec])

        flipped = [k for k, i in enumerate(idx) if lab[k] != en_label[i]]

        # D1 — direction of drift
        dirs: Counter[str] = Counter()
        for k in flipped:
            dirs[f"{en_label[idx[k]]}->{lab[k]}"] += 1
        all_dirs.update(dirs)

        # D2 — how far the deciding statistic moved on flipped rows
        deltas = [abs(mx[k] - en_max[idx[k]]) for k in flipped]
        all_deltas.extend(deltas)
        near = float(np.mean([d < _BOUNDARY_BAND for d in deltas])) if deltas else 0.0

        # D3 — embedding displacement
        tr_emb = encode(texts, encoder)
        cos = np.sum(tr_emb * en_emb[idx], axis=1)  # both L2-normalised
        displacement = float(np.mean(1.0 - cos))

        per_lang[lang] = {
            "n": len(idx),
            "stability": round(1.0 - len(flipped) / len(idx), 4),
            "flip_directions": dict(dirs),
            "mean_delta_max_prob": round(float(np.mean(deltas)), 4) if deltas else 0.0,
            "frac_flips_near_boundary": round(near, 4),
            "mean_embedding_displacement": round(displacement, 4),
        }
        logger.info("  %s: stability %.1f%%  displacement %.4f  near-boundary flips %.0f%%",
                    lang, per_lang[lang]["stability"] * 100, displacement, near * 100)

    # ── Verdict ───────────────────────────────────────────────────
    total_flips = sum(all_dirs.values())
    if total_flips:
        top_dir, top_n = all_dirs.most_common(1)[0]
        skew = top_n / total_flips
    else:
        top_dir, skew = "none", 0.0
    near_overall = float(np.mean([d < _BOUNDARY_BAND for d in all_deltas])) if all_deltas else 0.0
    mean_disp = float(np.mean([v["mean_embedding_displacement"]
                               for v in per_lang.values()]))

    systematic = skew > 0.65
    boundary_driven = near_overall > 0.60

    verdict: str
    if systematic and boundary_driven:
        verdict = "centroid_fix_likely_helps"
    elif boundary_driven and not systematic:
        verdict = "margin_fix_indicated"
    else:
        verdict = "representation_problem"

    diag = {
        "n_questions": len(rows),
        "english_decision_dist": dict(Counter(en_label)),
        "per_language": per_lang,
        "flip_directions_overall": dict(all_dirs),
        "dominant_direction": top_dir,
        "direction_skew": round(skew, 4),
        "frac_flips_near_boundary": round(near_overall, 4),
        "mean_embedding_displacement": round(mean_disp, 4),
        "verdict": verdict,
    }

    print("\n" + "=" * 72)
    print("CENTROID DIAGNOSTICS")
    print("=" * 72)
    print(f"  flips total          {total_flips}")
    print(f"  dominant direction   {top_dir}  ({skew:.0%} of flips)")
    print(f"  near-boundary flips  {near_overall:.0%}  "
          f"(|delta max_prob| < {_BOUNDARY_BAND})")
    print(f"  mean displacement    {mean_disp:.4f}  (1 - cos, en vs translation)")
    print("-" * 72)
    if verdict == "centroid_fix_likely_helps":
        print("  ✅ Flips are systematic AND boundary-driven.")
        print("     Re-centering the centroids should recover a real fraction.")
    elif verdict == "margin_fix_indicated":
        print("  ⚠ Flips are boundary-driven but NOT directional.")
        print("     This is threshold noise, not a centroid offset. Re-centering will")
        print("     not help much — widen the decision margin instead, or default to")
        print("     CLARIFY inside the band.")
    else:
        print("  ❌ Flips are large and unsystematic. The embedding itself moves too")
        print("     far under translation for any re-centering to fix. The encoder,")
        print("     not the centroids, is the bottleneck.")
    print("=" * 72)

    _OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_DIAG, "w") as f:
        json.dump(diag, f, indent=2)
    logger.info("Saved -> %s", _OUT_DIAG)
    return diag


# ─────────────────────────────────────────────────────────────────
# Rebuild
# ─────────────────────────────────────────────────────────────────
def rebuild(encoder: Any, rows: list[dict], cache: dict, k: int) -> dict[str, Any]:
    """Re-induce centers over English + translations.

    Same algorithm as the original (spectral clustering, cosine affinity, seed 42),
    only the input pool changes: 1 x N English becomes 7 x N multilingual.
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    questions = [r["question"] for r in rows]
    pool: list[str] = list(questions)
    lang_of: list[str] = ["en"] * len(questions)

    for lang in LANGS:
        for q in questions:
            t = cache.get(f"{lang}:{q}")
            if t:
                pool.append(t)
                lang_of.append(lang)

    logger.info("Multilingual pool: %d texts (%s)", len(pool),
                dict(Counter(lang_of)))

    emb = encode(pool, encoder)
    logger.info("Embeddings: %s", emb.shape)

    logger.info("Spectral clustering, K=%d, cosine affinity, seed=%d ...", k, _SEED)
    affinity = np.clip(emb @ emb.T, 0.0, 1.0)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=_SEED, assign_labels="kmeans")
    labels = sc.fit_predict(affinity)

    centroids = np.zeros((k, emb.shape[1]), dtype=np.float32)
    for c in range(k):
        m = labels == c
        if m.sum() == 0:
            logger.error("Cluster %d is empty — K is too large for this pool.", c)
            raise SystemExit(1)
        v = emb[m].mean(axis=0)
        centroids[c] = v / np.linalg.norm(v)

    sil = float(silhouette_score(emb, labels, metric="cosine"))

    # Language balance per cluster: a cluster dominated by one language means the
    # geometry is still language-shaped rather than meaning-shaped.
    balance: dict[int, dict[str, int]] = {}
    for c in range(k):
        balance[c] = dict(Counter(
            lang_of[i] for i in range(len(pool)) if labels[i] == c
        ))

    print("\n" + "=" * 72)
    print(f"MULTILINGUAL CENTROIDS — K={k}, pool={len(pool)}")
    print("=" * 72)
    print(f"  silhouette (cosine): {sil:.4f}")
    print(f"  cluster sizes: {dict(Counter(labels.tolist()))}")
    print("\n  language mix per cluster:")
    for c in range(k):
        total = sum(balance[c].values())
        top_lang, top_n = max(balance[c].items(), key=lambda kv: kv[1])
        flag = "  <- language-dominated" if top_n / total > 0.40 else ""
        print(f"    cluster {c}: {balance[c]}{flag}")
    print("=" * 72)

    _OUT_CENTROIDS.parent.mkdir(parents=True, exist_ok=True)
    np.save(_OUT_CENTROIDS, centroids)
    logger.info("Saved centroids -> %s", _OUT_CENTROIDS)

    report = {
        "k": k,
        "pool_size": len(pool),
        "languages": dict(Counter(lang_of)),
        "silhouette_cosine": round(sil, 4),
        "cluster_sizes": {str(c): int((labels == c).sum()) for c in range(k)},
        "language_mix_per_cluster": {str(c): balance[c] for c in range(k)},
        "centroids_path": str(_OUT_CENTROIDS),
        "next_steps": [
            "Re-derive thresholds against these centroids "
            "(scripts/derive_thresholds.py).",
            "Re-run crosslingual_eval.py --arm stability and compare to "
            "eval/results/crosslingual_stability.json.",
            "Report before/after per language. If mean stability does not move "
            "more than the CI width (~5 points), the fix did not work — say so.",
        ],
    }
    with open(_OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Baseline reminder, so the comparison is not lost.
    if _STABILITY.exists():
        with open(_STABILITY) as f:
            prev = json.load(f)
        print("\n  BASELINE TO BEAT (gate, before rebuild):")
        for lang, v in prev.get("per_language_gate", {}).items():
            print(f"    {v['language']:<10} {v['stability']:.1%}")
        print(f"    mean       {prev.get('mean_stability_gate', 0):.1%}")
        print("\n  Anything inside +/-5 points of these is noise, not improvement.")

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even when diagnostics say it will not help.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not (args.diagnose_only or args.rebuild):
        ap.error("pass --diagnose-only or --rebuild")

    import random

    from sentence_transformers import SentenceTransformer

    from app.gate.centerdistill import CenterDistillGate
    from app.settings import get_settings

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[:args.n]

    cache = load_cache()
    logger.info("Translation cache: %d entries", len(cache))

    logger.info("Loading LaBSE ...")
    encoder = SentenceTransformer("sentence-transformers/LaBSE")

    gate = CenterDistillGate(get_settings())
    if getattr(gate, "using_fallback", False):
        logger.error("Gate is in heuristic fallback — diagnostics would be meaningless.")
        return 1

    diag = diagnose(gate, encoder, rows, cache)

    if args.diagnose_only:
        return 0

    if diag["verdict"] != "centroid_fix_likely_helps" and not args.force:
        print("\n  Diagnostics indicate re-centering is not the right fix.")
        print("  Rebuilding anyway would burn time on a change that will not move the")
        print("  numbers. Pass --force to override, or address the indicated cause.")
        return 0

    rebuild(encoder, rows, cache, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
