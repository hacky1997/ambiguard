#!/usr/bin/env python3
"""
Three cross-lingual ambiguity mechanisms — all testable on cached data.

MOTIVATION
----------
Measured: the gate's routing decision survives translation only 66.8% of the time,
and the flips are DIRECTIONAL — 78% go ANSWER -> ALTERNATIVES, almost none the
reverse. Translation systematically makes questions look more ambiguous.

The cross-lingual consistency literature (CrossConST and successors) treats
cross-lingual divergence as symmetric noise to be minimised. This asymmetry is not
symmetric noise. Three mechanisms exploit it instead of suppressing it.

  H1  DISPLACEMENT AS DETECTOR
      If a question is genuinely ambiguous, a translator must commit to one reading,
      so its translations scatter further in embedding space. Displacement magnitude
      then carries ambiguity signal on its own — no classifier head involved.
      Falsifiable: correlate per-question displacement with the gold label.

  H2  MULTILINGUAL DISAGREEMENT AS DETECTOR
      Route the same question through N languages and look at the spread of
      decisions. Disagreement across languages is itself evidence of ambiguity.
      This uses the instability as signal rather than treating it as error.

  H3  DIRECTIONAL CALIBRATION
      If translation adds a roughly constant bias toward the ambiguous region, that
      bias can be estimated per language and subtracted before the head runs —
      a training-free correction. The literature assumes symmetric divergence and
      so does not look for a bias vector.

ORDER IS DELIBERATE
-------------------
H1 is cheapest and dies fastest if wrong. H2 is nearly free once per-row decisions
are retained. H3 needs the most machinery. Each is reported independently with
bootstrap CIs and an explicit null verdict — a null result here is a real finding
and is printed as one, not buried.

PREREQUISITES
    eval/results/translation_cache.json   (from crosslingual_eval.py)
    eval/datasets/golden_gate.jsonl

USAGE
    python scripts/crosslingual_novel.py --n 300
    python scripts/crosslingual_novel.py --n 300 --hypotheses h1 h2
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
_OUT = Path("eval/results/crosslingual_novel.json")
_SEED = 42

LANGS = ["es", "de", "ja", "ar", "hi", "sw"]


def boot_ci(x: np.ndarray, n_boot: int = 10_000) -> tuple[float, float, float]:
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(_SEED)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via rank statistic. labels: 1 = positive class."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auc(scores: np.ndarray, labels: np.ndarray,
             n_boot: int = 2_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(_SEED)
    base = auc(scores, labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(auc(scores[idx], labels[idx]))
    if not vals:
        return base, base, base
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def is_ambiguous(label: str) -> int:
    return 0 if label == "ANSWER" else 1


# ═════════════════════════════════════════════════════════════════
# Shared setup: encode everything once, decide everything once.
# ═════════════════════════════════════════════════════════════════
def build_state(gate: Any, encoder: Any, rows: list[dict],
                cache: dict) -> dict[str, Any]:
    questions = [r["question"] for r in rows]
    gold = np.array([is_ambiguous(r["expected_behaviour"]) for r in rows])

    logger.info("Encoding English ...")
    en_emb = encoder.encode(questions, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)

    logger.info("Deciding English ...")
    en_cls = gate.encode_cls_batch(questions, [r.get("context") for r in rows])
    en_dec = [gate.decide_from_embedding(v) for v in en_cls]

    tr_emb: dict[str, np.ndarray] = {}
    tr_dec: dict[str, list[dict]] = {}
    coverage: dict[str, int] = {}

    for lang in LANGS:
        texts = [cache.get(f"{lang}:{q}") for q in questions]
        have = [i for i, t in enumerate(texts) if t]
        coverage[lang] = len(have)
        if len(have) < len(questions):
            logger.warning("%s: %d/%d translations cached",
                           lang, len(have), len(questions))
        if not have:
            continue

        logger.info("Encoding + deciding %s (%d) ...", lang, len(have))
        emb = encoder.encode([texts[i] for i in have], batch_size=64,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        full = np.full((len(questions), emb.shape[1]), np.nan, dtype=np.float32)
        full[have] = emb
        tr_emb[lang] = full

        dec: list[dict | None] = [None] * len(questions)
        have_texts = [texts[i] for i in have]
        have_contexts = [rows[i].get("context") for i in have]
        tr_cls = gate.encode_cls_batch(have_texts, have_contexts)
        for j, i in enumerate(have):
            dec[i] = gate.decide_from_embedding(tr_cls[j])
        tr_dec[lang] = dec  # type: ignore[assignment]

    return {
        "questions": questions, "gold": gold, "rows": rows, "cache": cache,
        "en_emb": en_emb, "en_dec": en_dec,
        "tr_emb": tr_emb, "tr_dec": tr_dec, "coverage": coverage,
    }


# ═════════════════════════════════════════════════════════════════
# H1 — displacement as an ambiguity detector
# ═════════════════════════════════════════════════════════════════
def run_h1(st: dict[str, Any]) -> dict[str, Any]:
    gold = st["gold"]
    n = len(gold)

    # Per-question displacement, averaged over languages with coverage.
    disp = np.full(n, np.nan)
    spread = np.full(n, np.nan)  # variance ACROSS languages, not just magnitude
    for i in range(n):
        ds, vecs = [], []
        for lang in LANGS:
            if lang not in st["tr_emb"]:
                continue
            v = st["tr_emb"][lang][i]
            if np.isnan(v).any():
                continue
            ds.append(1.0 - float(np.dot(v, st["en_emb"][i])))
            vecs.append(v)
        if ds:
            disp[i] = float(np.mean(ds))
        if len(vecs) >= 2:
            arr = np.stack(vecs)
            centroid = arr.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            spread[i] = float(np.mean(1.0 - arr @ centroid))

    valid = ~np.isnan(disp)
    d, g = disp[valid], gold[valid]
    a_disp, lo_d, hi_d = boot_auc(d, g)

    valid_s = ~np.isnan(spread)
    s, gs = spread[valid_s], gold[valid_s]
    a_spr, lo_s, hi_s = boot_auc(s, gs)

    # Baseline to beat: the gate's own confidence as a detector.
    conf = np.array([1.0 - dd["max_prob"] for dd in st["en_dec"]])
    a_conf, lo_c, hi_c = boot_auc(conf, gold)

    best = max(
        [("displacement", a_disp, lo_d, hi_d),
         ("inter-language spread", a_spr, lo_s, hi_s)],
        key=lambda t: t[1],
    )
    beats_chance = best[2] > 0.5
    beats_gate = best[2] > a_conf

    print("\n" + "=" * 72)
    print("H1 — DISPLACEMENT AS AMBIGUITY DETECTOR")
    print("=" * 72)
    print(f"{'signal':<26} {'AUC':>7} {'CI95':>18}")
    print("-" * 72)
    print(f"{'translation displacement':<26} {a_disp:>7.3f} "
          f"{f'[{lo_d:.3f}, {hi_d:.3f}]':>18}")
    print(f"{'inter-language spread':<26} {a_spr:>7.3f} "
          f"{f'[{lo_s:.3f}, {hi_s:.3f}]':>18}")
    print(f"{'gate confidence (baseline)':<26} {a_conf:>7.3f} "
          f"{f'[{lo_c:.3f}, {hi_c:.3f}]':>18}")
    print("-" * 72)
    print("  AUC 0.5 = chance. Displacement uses NO classifier head — it is a")
    print("  property of the encoder and the translations alone.")
    if beats_chance and beats_gate:
        print(f"\n  ✅ '{best[0]}' beats chance AND the trained gate.")
        print("     A training-free ambiguity signal derived from translation.")
    elif beats_chance:
        print(f"\n  ⚠ '{best[0]}' beats chance but not the gate. Weak but real.")
    else:
        print("\n  ❌ NULL. Displacement carries no ambiguity signal.")
        print("     Translation instability is not explained by ambiguity.")

    return {
        "displacement_auc": round(a_disp, 4), "displacement_ci": [round(lo_d, 4), round(hi_d, 4)],
        "spread_auc": round(a_spr, 4), "spread_ci": [round(lo_s, 4), round(hi_s, 4)],
        "gate_confidence_auc": round(a_conf, 4),
        "gate_confidence_ci": [round(lo_c, 4), round(hi_c, 4)],
        "n_valid": int(valid.sum()),
        "beats_chance": bool(beats_chance),
        "beats_gate": bool(beats_gate),
        "verdict": ("beats_gate" if beats_gate else
                    "weak_signal" if beats_chance else "null"),
    }


# ═════════════════════════════════════════════════════════════════
# H2 — multilingual disagreement as an ambiguity detector
# ═════════════════════════════════════════════════════════════════
def run_h2(st: dict[str, Any]) -> dict[str, Any]:
    gold = st["gold"]
    n = len(gold)

    votes: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        votes[i].append(st["en_dec"][i]["behaviour"])
        for lang in LANGS:
            if lang not in st["tr_dec"]:
                continue
            d = st["tr_dec"][lang][i]
            if d is not None:
                votes[i].append(d["behaviour"])

    # Disagreement = normalised entropy over the vote distribution.
    disagree = np.zeros(n)
    majority: list[str] = []
    for i, v in enumerate(votes):
        c = Counter(v)
        majority.append(c.most_common(1)[0][0])
        p = np.array(list(c.values()), dtype=float) / len(v)
        h = -(p * np.log(p + 1e-12)).sum()
        disagree[i] = h / np.log(len(v)) if len(v) > 1 else 0.0

    a_dis, lo, hi = boot_auc(disagree, gold)

    # Majority vote as a classifier, vs English alone.
    maj_pred = np.array([0 if m == "ANSWER" else 1 for m in majority])
    en_pred = np.array([0 if d["behaviour"] == "ANSWER" else 1 for d in st["en_dec"]])
    m_acc, m_lo, m_hi = boot_ci((maj_pred == gold).astype(float))
    e_acc, e_lo, e_hi = boot_ci((en_pred == gold).astype(float))
    maj_class = float(max(np.mean(gold), 1 - np.mean(gold)))

    print("\n" + "=" * 72)
    print("H2 — MULTILINGUAL DISAGREEMENT AS DETECTOR")
    print("=" * 72)
    print(f"  disagreement AUC   {a_dis:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  mean disagreement  ambiguous {disagree[gold == 1].mean():.3f}  "
          f"unambiguous {disagree[gold == 0].mean():.3f}")
    print("-" * 72)
    print(f"{'classifier':<26} {'accuracy':>10} {'CI95':>18}")
    print(f"{'English only':<26} {e_acc:>9.1%} {f'[{e_lo:.1%}, {e_hi:.1%}]':>18}")
    print(f"{'7-language majority vote':<26} {m_acc:>9.1%} "
          f"{f'[{m_lo:.1%}, {m_hi:.1%}]':>18}")
    print(f"{'majority class':<26} {maj_class:>9.1%} {'—':>18}")
    print("-" * 72)
    vote_helps = m_lo > e_hi
    dis_signal = lo > 0.5
    if dis_signal:
        print("  ✅ Cross-lingual disagreement carries ambiguity signal.")
        print("     Instability is evidence, not just error.")
    else:
        print("  ❌ NULL. Questions do not disagree more when they are ambiguous.")
    if vote_helps:
        print("  ✅ Ensemble voting beats single-language routing.")
    else:
        print("  ⚠ Voting does not separate from English-only.")

    return {
        "disagreement_auc": round(a_dis, 4), "disagreement_ci": [round(lo, 4), round(hi, 4)],
        "mean_disagreement_ambiguous": round(float(disagree[gold == 1].mean()), 4),
        "mean_disagreement_unambiguous": round(float(disagree[gold == 0].mean()), 4),
        "english_accuracy": round(e_acc, 4), "english_ci": [round(e_lo, 4), round(e_hi, 4)],
        "vote_accuracy": round(m_acc, 4), "vote_ci": [round(m_lo, 4), round(m_hi, 4)],
        "majority_class_baseline": round(maj_class, 4),
        "disagreement_is_signal": bool(dis_signal),
        "voting_beats_english": bool(vote_helps),
    }


# ═════════════════════════════════════════════════════════════════
# H3 — directional calibration
# ═════════════════════════════════════════════════════════════════
def run_h3(st: dict[str, Any], gate: Any) -> dict[str, Any]:
    """Estimate a per-language bias vector on a train split, apply on held-out.

    Fitting and evaluating on the same rows would guarantee an improvement and
    prove nothing, so the split is enforced.
    """
    n = len(st["gold"])
    idx = list(range(n))
    random.Random(_SEED).shuffle(idx)
    cut = n // 2
    fit_idx, eval_idx = idx[:cut], idx[cut:]
    logger.info("H3 split: %d fit / %d eval", len(fit_idx), len(eval_idx))

    if not (hasattr(gate, "decide_from_embedding") and hasattr(gate, "encode_cls")):
        print("\n" + "=" * 72)
        print("H3 — DIRECTIONAL CALIBRATION")
        print("=" * 72)
        print("  SKIPPED. The gate exposes no embedding-level entry point.")
        print("  This test needs decide_from_embedding(vec) so a corrected vector")
        print("  can be pushed through the center head without re-tokenising.")
        print("  Add it to app/gate/centerdistill.py, then re-run.")
        print("=" * 72)
        return {"status": "skipped", "reason": "no decide_from_embedding on gate"}

    logger.info("Extracting gate CLS embeddings for H3 (1024-dim) ...")
    questions = st["questions"]
    rows = st["rows"]
    contexts = [r.get("context") for r in rows]
    gate_en_cls = gate.encode_cls_batch(questions, contexts)

    results: dict[str, Any] = {}
    cache = st.get("cache") or {}
    for lang in LANGS:
        texts = [cache.get(f"{lang}:{q}") for q in questions]
        have = [i for i, t in enumerate(texts) if t]
        if not have:
            continue

        gate_tr_cls = np.full((n, 1024), np.nan, dtype=np.float32)
        valid_texts = [texts[i] for i in have]
        valid_contexts = [rows[i].get("context") for i in have]
        tr_embeddings = gate.encode_cls_batch(valid_texts, valid_contexts)
        for idx, i in enumerate(have):
            gate_tr_cls[i] = tr_embeddings[idx]

        # Bias vector: mean displacement on the FIT split only.
        deltas = [gate_tr_cls[i] - gate_en_cls[i]
                  for i in fit_idx if not np.isnan(gate_tr_cls[i]).any()]
        if not deltas:
            continue
        bias = np.mean(np.stack(deltas), axis=0)

        before, after = [], []
        for i in eval_idx:
            if np.isnan(gate_tr_cls[i]).any():
                continue
            en_lab = st["en_dec"][i]["behaviour"]
            raw = st["tr_dec"][lang][i]
            if raw is None:
                continue
            before.append(1.0 if raw["behaviour"] == en_lab else 0.0)

            corrected = gate_tr_cls[i] - bias
            lab = gate.decide_from_embedding(corrected)["behaviour"]
            after.append(1.0 if lab == en_lab else 0.0)

        b, b_lo, b_hi = boot_ci(np.array(before))
        a, a_lo, a_hi = boot_ci(np.array(after))
        results[lang] = {
            "n_eval": len(after),
            "stability_before": round(b, 4), "ci_before": [round(b_lo, 4), round(b_hi, 4)],
            "stability_after": round(a, 4), "ci_after": [round(a_lo, 4), round(a_hi, 4)],
            "delta": round(a - b, 4),
            "bias_norm": round(float(np.linalg.norm(bias)), 4),
        }

    print("\n" + "=" * 72)
    print("H3 — DIRECTIONAL CALIBRATION  (bias fit on held-out split)")
    print("=" * 72)
    print(f"{'lang':<8} {'before':>9} {'after':>9} {'delta':>9} {'|bias|':>9}")
    print("-" * 72)
    for lang, v in results.items():
        print(f"{lang:<8} {v['stability_before']:>8.1%} {v['stability_after']:>8.1%} "
              f"{v['delta']:>+8.1%} {v['bias_norm']:>9.4f}")
    if results:
        mean_delta = float(np.mean([v["delta"] for v in results.values()]))
        print("-" * 72)
        print(f"  mean delta {mean_delta:+.1%}")
        if mean_delta > 0.05:
            print("  ✅ A per-language bias vector recovers real stability,")
            print("     with no retraining. Divergence is partly a constant offset.")
        elif mean_delta > 0.0:
            print("  ⚠ Positive but inside noise at this sample size.")
        else:
            print("  ❌ NULL. Divergence is not a constant offset.")
        results["mean_delta"] = round(mean_delta, 4)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--hypotheses", nargs="+", default=["h1", "h2", "h3"],
                    choices=["h1", "h2", "h3"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not _CACHE.exists():
        raise SystemExit(
            f"{_CACHE} missing. Run crosslingual_eval.py --arm stability first."
        )

    from sentence_transformers import SentenceTransformer

    from app.gate.centerdistill import CenterDistillGate
    from app.settings import get_settings

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    random.Random(_SEED).shuffle(rows)
    rows = rows[: args.n]

    with open(_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    logger.info("Cache: %d entries", len(cache))

    gate = CenterDistillGate(get_settings())
    if getattr(gate, "using_fallback", False):
        logger.error("Gate is in heuristic fallback — results meaningless.")
        return 1

    logger.info("Loading LaBSE ...")
    encoder = SentenceTransformer("sentence-transformers/LaBSE")

    st = build_state(gate, encoder, rows, cache)
    gold = st["gold"]
    logger.info("Gold: %d ambiguous / %d unambiguous",
                int(gold.sum()), int((1 - gold).sum()))

    out: dict[str, Any] = {
        "n": len(rows),
        "seed": _SEED,
        "coverage": st["coverage"],
        "gold_dist": {"ambiguous": int(gold.sum()), "unambiguous": int((1 - gold).sum())},
    }
    if "h1" in args.hypotheses:
        out["h1_displacement_detector"] = run_h1(st)
    if "h2" in args.hypotheses:
        out["h2_disagreement_detector"] = run_h2(st)
    if "h3" in args.hypotheses:
        out["h3_directional_calibration"] = run_h3(st, gate)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved -> {_OUT}")
    print("\n  Null results here are findings. Report them as such — 'translation")
    print("  instability is not explained by ambiguity' is worth knowing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
