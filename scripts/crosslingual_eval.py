#!/usr/bin/env python3
"""
Cross-lingual evaluation harness: translation stability & typological ambiguity.

ARMS
----
1. stability     Translate English golden rows into 6 languages. Measure whether the
                 routing decision survives translation, relative to each system's OWN
                 English decision. Ambiguity is a property of meaning, so a decision
                 that flips on translation is wrong regardless of which side was right.

2. typological   Evaluate on the hand-specified typological set: ambiguity that is
                 created or erased by translation (formality, subject drop, date
                 format, entity collision, ...). ~20% of rows are controls whose
                 correct label is ANSWER.

3. --include-llm Run gpt-4o-mini on identical inputs for side-by-side comparison.

USAGE
    python scripts/crosslingual_eval.py --arm both --n 300
    python scripts/crosslingual_eval.py --arm both --n 300 --include-llm --yes
    python scripts/crosslingual_eval.py --arm typological --label-mode three_way
    python scripts/crosslingual_eval.py --arm stability --n 20 --dump-translations
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GOLDEN = Path("eval/datasets/golden_gate.jsonl")
_TYPOLOGICAL = Path("eval/datasets/typological_ambiguity.jsonl")
_OUT_STABILITY = Path("eval/results/crosslingual_stability.json")
_OUT_TYPOLOGICAL = Path("eval/results/crosslingual_typological.json")
_TRANSLATIONS_DUMP = Path("eval/results/translations_sample.jsonl")
_CACHE_PATH = Path("eval/results/translation_cache.json")
_SEED = 42

# NLLB codes, spanning resource levels and typological families.
LANGUAGES: dict[str, tuple[str, str]] = {
    "es": ("spa_Latn", "Spanish"),
    "de": ("deu_Latn", "German"),
    "ja": ("jpn_Jpan", "Japanese"),
    "ar": ("arb_Arab", "Arabic"),
    "hi": ("hin_Deva", "Hindi"),
    "sw": ("swh_Latn", "Swahili"),
}

# If one class exceeds this share of English decisions, stability is trivial.
_COLLAPSE_THRESHOLD = 0.80
# Below this, the system is flagging unambiguous queries as ambiguous.
_CONTROL_FLOOR = 0.50


class Translator:
    """NLLB-600M translator with disk caching.

    Mock mode is opt-in and taints the run. It exists so the plumbing can be tested
    offline — never so that a failed download silently produces publishable-looking
    numbers.
    """

    def __init__(self, use_mock: bool = False) -> None:
        self.use_mock = use_mock
        self.pipeline = None
        self.tainted = use_mock
        self.cache: dict[str, str] = self._load_cache()

        if use_mock:
            logger.warning(
                "MOCK TRANSLATION ENABLED. Questions stay in English with a language "
                "tag prefixed. Stability numbers from this run are MEANINGLESS and the "
                "output will be marked tainted."
            )
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        model_name = "facebook/nllb-200-distilled-600M"
        logger.info("Loading %s (first run downloads ~2.4 GB) ...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "translation", model=model, tokenizer=tokenizer,
            src_lang="eng_Latn", device=-1,
        )
        logger.info("Translator ready.")

    def _load_cache(self) -> dict[str, str]:
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to load translation cache (%s). Starting fresh.", exc)
        return {}

    def _save_cache(self) -> None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def translate_batch(self, texts: list[str], lang: str) -> list[str]:
        if self.use_mock:
            return [f"[{lang.upper()}] {t}" for t in texts]

        results: list[str] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for idx, text in enumerate(texts):
            key = f"{lang}:{text}"
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append("")  # placeholder
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            logger.info("Translating %d missing items for %s via NLLB...", len(missing_texts), lang)
            assert self.pipeline is not None
            tgt = LANGUAGES[lang][0]
            translated_missing: list[str] = []
            failures = 0
            for i in range(0, len(missing_texts), 8):
                chunk = missing_texts[i:i + 8]
                try:
                    res = self.pipeline(chunk, tgt_lang=tgt, max_length=512)
                    translated_missing.extend(r["translation_text"] for r in res)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Translation failed for a batch in %s: %s", lang, exc)
                    failures += len(chunk)
                    translated_missing.extend(chunk)

            for idx, orig_text, trans_text in zip(missing_indices, missing_texts, translated_missing):
                results[idx] = trans_text
                key = f"{lang}:{orig_text}"
                self.cache[key] = trans_text

            self._save_cache()
            if failures:
                logger.warning("%d/%d translations failed for %s and fell back to English.", failures, len(missing_texts), lang)

        return results


def boot_ci(x: np.ndarray, n_boot: int = 10_000) -> tuple[float, float, float]:
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(_SEED)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def normalise(label: str, mode: str) -> str:
    """binary collapses ALTERNATIVES/CLARIFY; three_way keeps them distinct."""
    if mode == "three_way":
        return label
    return "ANSWER" if label == "ANSWER" else "AMBIGUOUS"


def check_collapse(preds: list[str], who: str) -> str | None:
    """Warn when one class dominates — stability is then trivially high."""
    dist = Counter(preds)
    top, count = dist.most_common(1)[0]
    share = count / len(preds)
    if share >= _COLLAPSE_THRESHOLD:
        msg = (f"{who} predicts '{top}' on {share:.0%} of English rows. Stability "
               "is trivially high for a near-constant predictor and does not "
               "demonstrate cross-lingual robustness.")
        logger.warning(msg)
        return msg
    return None


# ─────────────────────────────────────────────────────────────────
# Arm 1 — translation stability
# ─────────────────────────────────────────────────────────────────
def run_stability_arm(
    gate: Any,
    translator: Translator,
    n_samples: int,
    label_mode: str,
    judge: Any = None,
    dump_translations: bool = False,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    rng = random.Random(_SEED)
    rng.shuffle(rows)
    sample = rows[:n_samples]
    logger.info("Stability arm: %d rows x %d languages", len(sample), len(LANGUAGES))

    # English baselines — each system compared against ITS OWN English decision.
    en_gate = [normalise(gate.decide(r["question"], r.get("context"))["behaviour"],
                         label_mode) for r in sample]
    warnings: list[str] = []
    w = check_collapse(en_gate, "Gate")
    if w:
        warnings.append(w)
    logger.info("English baseline (gate): %s", dict(Counter(en_gate)))

    en_llm: list[str] = []
    if judge is not None:
        for r in sample:
            res_pred = judge.predict(r["question"], r.get("context") or "")["prediction"]
            en_llm.append(normalise(res_pred, label_mode))
        w = check_collapse(en_llm, "LLM judge")
        if w:
            warnings.append(w)
        logger.info("English baseline (llm): %s", dict(Counter(en_llm)))

    per_lang_gate: dict[str, Any] = {}
    per_lang_llm: dict[str, Any] = {}
    all_gate: list[float] = []
    all_llm: list[float] = []
    dumps: list[dict[str, str]] = []

    for lang in LANGUAGES:
        name = LANGUAGES[lang][1]
        logger.info("Translating -> %s", name)
        translated = translator.translate_batch([r["question"] for r in sample], lang)

        # Check for unchanged translations (Bug 3)
        unchanged = sum(1 for orig, trans in zip([r["question"] for r in sample], translated)
                        if orig.strip().lower() == trans.strip().lower())
        unchanged_rate = unchanged / len(sample) if len(sample) > 0 else 0.0
        if unchanged_rate > 0.10:
            msg = f"{name} ({lang}): {unchanged_rate:.1%} of translations are identical to English original."
            logger.warning(msg)
            warnings.append(msg)

        if dump_translations:
            for orig, trans in list(zip([r["question"] for r in sample], translated))[:20]:
                dumps.append({"lang": lang, "english": orig, "translated": trans})

        # Context stays English: this isolates the question's surface language.
        m_gate = [
            1.0 if normalise(gate.decide(q, r.get("context"))["behaviour"], label_mode)
                   == en_gate[i] else 0.0
            for i, (q, r) in enumerate(zip(translated, sample))
        ]
        acc, lo, hi = boot_ci(np.array(m_gate))
        per_lang_gate[lang] = {
            "language": name,
            "stability": round(acc, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "unchanged_rate": round(unchanged_rate, 4),
        }
        all_gate.extend(m_gate)
        logger.info("  gate %s: %.1f%% (unchanged: %.1f%%)", name, acc * 100, unchanged_rate * 100)

        if judge is not None:
            m_llm = []
            for i, (q, r) in enumerate(zip(translated, sample)):
                res_pred = judge.predict(q, r.get("context") or "")["prediction"]
                m_llm.append(1.0 if normalise(res_pred, label_mode) == en_llm[i] else 0.0)

            acc_l, lo_l, hi_l = boot_ci(np.array(m_llm))
            per_lang_llm[lang] = {
                "language": name,
                "stability": round(acc_l, 4),
                "ci95": [round(lo_l, 4), round(hi_l, 4)],
                "unchanged_rate": round(unchanged_rate, 4),
            }
            all_llm.extend(m_llm)
            logger.info("  llm  %s: %.1f%%", name, acc_l * 100)

    if dumps:
        _TRANSLATIONS_DUMP.parent.mkdir(parents=True, exist_ok=True)
        with open(_TRANSLATIONS_DUMP, "w", encoding="utf-8") as f:
            for d in dumps:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.info("Wrote %d translations for spot-checking -> %s",
                    len(dumps), _TRANSLATIONS_DUMP)

    mg, lg, hg = boot_ci(np.array(all_gate))
    out: dict[str, Any] = {
        "n_samples": len(sample),
        "label_mode": label_mode,
        "translation_tainted": translator.tainted,
        "warnings": warnings,
        "english_baseline_dist_gate": dict(Counter(en_gate)),
        "mean_stability_gate": round(mg, 4),
        "ci95_gate": [round(lg, 4), round(hg, 4)],
        "per_language_gate": per_lang_gate,
        "worst_language_gate": min(per_lang_gate,
                                   key=lambda k: per_lang_gate[k]["stability"]),
    }
    if judge is not None:
        ml, ll, hl = boot_ci(np.array(all_llm))
        out.update({
            "english_baseline_dist_llm": dict(Counter(en_llm)),
            "mean_stability_llm": round(ml, 4),
            "ci95_llm": [round(ll, 4), round(hl, 4)],
            "per_language_llm": per_lang_llm,
            "gate_minus_llm": round(mg - ml, 4),
        })
    return out


# ─────────────────────────────────────────────────────────────────
# Arm 2 — typological ambiguity
# ─────────────────────────────────────────────────────────────────
def run_typological_arm(
    gate: Any,
    label_mode: str,
    judge: Any = None,
) -> dict[str, Any]:
    if not _TYPOLOGICAL.exists():
        logger.error("Missing %s — build the typological set first.", _TYPOLOGICAL)
        raise SystemExit(1)

    rows = [json.loads(line) for line in open(_TYPOLOGICAL, encoding="utf-8")]
    logger.info("Typological arm: %d rows, label_mode=%s", len(rows), label_mode)

    gold = [normalise(r["expected_behaviour"], label_mode) for r in rows]
    gate_pred = [normalise(gate.decide(r["question"], r.get("context"))["behaviour"],
                           label_mode) for r in rows]
    llm_pred: list[str] = []
    if judge is not None:
        for r in rows:
            res_pred = judge.predict(r["question"], r.get("context") or "")["prediction"]
            llm_pred.append(normalise(res_pred, label_mode))

    def summarise(pred: list[str], who: str) -> dict[str, Any]:
        ok = np.array([p == g for p, g in zip(pred, gold)], dtype=float)
        acc, lo, hi = boot_ci(ok)

        ctrl_idx = [i for i, r in enumerate(rows) if r["expected_behaviour"] == "ANSWER"]
        ctrl = np.array([pred[i] == gold[i] for i in ctrl_idx], dtype=float)
        c_acc, c_lo, c_hi = boot_ci(ctrl)

        per_cat: dict[str, Any] = {}
        for cat in sorted({r["category"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["category"] == cat]
            v = np.array([pred[i] == gold[i] for i in idx], dtype=float)
            a, l, h = boot_ci(v)
            per_cat[cat] = {"n": len(idx), "accuracy": round(a, 4),
                            "ci95": [round(l, 4), round(h, 4)]}

        warn = None
        if c_acc < _CONTROL_FLOOR:
            warn = (f"{who} control accuracy {c_acc:.1%} is below {_CONTROL_FLOOR:.0%}: "
                    "unambiguous cross-lingual queries are being flagged as ambiguous. "
                    "This is over-triggering, not detection — overall accuracy on this "
                    "set is not meaningful while controls fail.")
            logger.warning(warn)

        return {
            "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "control_accuracy": round(c_acc, 4),
            "control_ci95": [round(c_lo, 4), round(c_hi, 4)],
            "n_controls": len(ctrl_idx),
            "pred_dist": dict(Counter(pred)),
            "per_category": per_cat,
            "warning": warn,
        }

    out: dict[str, Any] = {
        "n_samples": len(rows),
        "label_mode": label_mode,
        "gold_dist": dict(Counter(gold)),
        "gate": summarise(gate_pred, "Gate"),
    }
    if judge is not None:
        out["llm"] = summarise(llm_pred, "LLM judge")
        out["gate_minus_llm"] = round(
            out["gate"]["accuracy"] - out["llm"]["accuracy"], 4)
    return out


# ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["stability", "typological", "both"], default="both")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--label-mode", choices=["binary", "three_way"], default="binary",
                    help="three_way keeps ALTERNATIVES and CLARIFY distinct.")
    ap.add_argument("--include-llm", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="Automatically confirm cost estimate for LLM comparison arm.")
    ap.add_argument("--mock-translator", action="store_true",
                    help="Offline plumbing test only. Taints the output.")
    ap.add_argument("--dump-translations", action="store_true",
                    help="Write 20 translations per language for manual spot-checking.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.gate.centerdistill import CenterDistillGate
    from app.settings import get_settings

    settings = get_settings()
    gate = CenterDistillGate(settings)
    if getattr(gate, "using_fallback", False):
        logger.error("Gate is in heuristic fallback — results would be meaningless.")
        return 1

    # Cost Guard for LLM arm
    if args.include_llm:
        if settings.llm_provider == "mock":
            logger.error("--include-llm with LLM_PROVIDER=mock produces nothing useful. "
                         "Set a real provider or drop the flag.")
            return 1

        n_typo = 210 if _TYPOLOGICAL.exists() else 0
        n_stab_calls = (args.n * 7) if args.arm in ("stability", "both") else 0
        n_typo_calls = n_typo if args.arm in ("typological", "both") else 0
        total_calls = n_stab_calls + n_typo_calls
        est_cost = total_calls * 0.005

        print("\n" + "=" * 72)
        print("COST ESTIMATE FOR LLM COMPARISON ARM")
        print("=" * 72)
        print(f"  Estimated LLM API calls: ~{total_calls:,} calls (~${est_cost:.2f} USD using {settings.openai_model})")
        print("=" * 72)

        if not args.yes:
            try:
                ans = input("Proceed with LLM evaluation? [y/N]: ")
                if ans.strip().lower() not in ("y", "yes"):
                    print("Aborted.")
                    return 0
            except EOFError:
                logger.error("Non-interactive terminal without --yes. Aborting.")
                return 1

        from app.llm.registry import get_provider
        from eval.arms.llm_judge_arm import LLMJudgeArm
        provider = get_provider(
            settings.llm_provider,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        judge = LLMJudgeArm(provider, is_binary=(args.label_mode == "binary"))
        logger.info("LLM comparison arm initialized: %s", judge.name)

    translator = Translator(use_mock=args.mock_translator)

    if args.arm in ("stability", "both"):
        res = run_stability_arm(gate, translator, args.n, args.label_mode,
                                judge, args.dump_translations)
        _OUT_STABILITY.parent.mkdir(parents=True, exist_ok=True)
        with open(_OUT_STABILITY, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print(f"TRANSLATION STABILITY — n={res['n_samples']}, mode={args.label_mode}")
        print("=" * 72)
        if res["translation_tainted"]:
            print("  *** MOCK TRANSLATION — THESE NUMBERS ARE NOT VALID ***")
        for wmsg in res["warnings"]:
            print(f"  WARNING: {wmsg}")
        print(f"  English baseline (gate): {res['english_baseline_dist_gate']}")
        print(f"\n{'language':<12} {'gate':>9} {'CI95':>18}", end="")
        print(f" {'llm':>9} {'CI95':>18}" if judge is not None else "")
        print("-" * 72)
        for lang, v in res["per_language_gate"].items():
            ci = f"[{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]"
            line = f"{v['language']:<12} {v['stability']:>8.1%} {ci:>18}"
            if judge is not None:
                lv = res["per_language_llm"][lang]
                lci = f"[{lv['ci95'][0]:.1%}, {lv['ci95'][1]:.1%}]"
                line += f" {lv['stability']:>8.1%} {lci:>18}"
            print(line)
        print("-" * 72)
        print(f"  mean gate {res['mean_stability_gate']:.1%}  "
              f"worst: {LANGUAGES[res['worst_language_gate']][1]}")
        if judge is not None:
            print(f"  mean llm  {res['mean_stability_llm']:.1%}  "
                  f"gate - llm = {res['gate_minus_llm']:+.1%}")
        print(f"\n  Saved -> {_OUT_STABILITY}")

    if args.arm in ("typological", "both"):
        res = run_typological_arm(gate, args.label_mode, judge)
        _OUT_TYPOLOGICAL.parent.mkdir(parents=True, exist_ok=True)
        with open(_OUT_TYPOLOGICAL, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print(f"TYPOLOGICAL AMBIGUITY — n={res['n_samples']}, mode={args.label_mode}")
        print("=" * 72)
        print(f"  gold {res['gold_dist']}")
        for who in ("gate", "llm"):
            if who not in res:
                continue
            r = res[who]
            print(f"\n  {who.upper()}")
            print(f"    accuracy {r['accuracy']:.1%} "
                  f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")
            print(f"    controls {r['control_accuracy']:.1%} "
                  f"[{r['control_ci95'][0]:.1%}, {r['control_ci95'][1]:.1%}] "
                  f"(n={r['n_controls']})")
            print(f"    pred {r['pred_dist']}")
            if r["warning"]:
                print(f"    WARNING: {r['warning']}")
        print(f"\n{'category':<20} {'n':>4} {'gate':>9}", end="")
        print(f" {'llm':>9}" if judge is not None else "")
        print("-" * 72)
        cats = res["gate"]["per_category"]
        for cat in sorted(cats, key=lambda c: cats[c]["accuracy"]):
            line = f"{cat:<20} {cats[cat]['n']:>4} {cats[cat]['accuracy']:>8.1%}"
            if judge is not None:
                line += f" {res['llm']['per_category'][cat]['accuracy']:>8.1%}"
            print(line)
        if judge is not None:
            print(f"\n  gate - llm = {res['gate_minus_llm']:+.1%}")
        print(f"\n  Saved -> {_OUT_TYPOLOGICAL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
