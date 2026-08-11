#!/usr/bin/env python3
"""Phase 0b — Multilingual Transfer Evaluation.

Evaluates the UNCHANGED English-authored regex rules on NLLB-translated queries
across 5 target languages (es, de, hi, ja, sw) for 4 in-scope categories:
  - date_format
  - currency
  - measurement
  - number_ambiguity (numeric_scale)

Includes ~25% negative controls per language (ISO dates, explicit currency codes, stated units).

Output: eval/results/phase0b_multilingual.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from scripts.crosslingual_eval import Translator
from scripts.eval_phase0_regex import REGEX_RULES, predict_regex, wilson_ci

_TYPOLOGICAL_PATH = Path("eval/datasets/typological_ambiguity.jsonl")
_OUT_JSON = Path("eval/results/phase0b_multilingual.json")

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "sw": "Swahili",
}

IN_SCOPE_CATEGORIES = {
    "date_format": "date_format",
    "currency": "currency",
    "measurement": "measurement",
    "numeric_scale": "number_ambiguity",
    "number_ambiguity": "number_ambiguity",
}

def load_english_in_scope_rows() -> list[dict[str, Any]]:
    """Load and normalize in-scope category rows from typological_ambiguity.jsonl."""
    rows: list[dict[str, Any]] = []
    with open(_TYPOLOGICAL_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cat = item.get("category")
            if cat in IN_SCOPE_CATEGORIES:
                norm_cat = IN_SCOPE_CATEGORIES[cat]
                rows.append({
                    "id": item.get("id"),
                    "question": item["question"],
                    "category": norm_cat,
                    "is_ambiguous": (item.get("expected_behaviour") == "AMBIGUOUS"),
                    "is_control": (item.get("expected_behaviour") == "ANSWER"),
                })
    return rows

def build_40_per_language_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exactly 40 balanced rows (10 per category, ~25% controls)."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)
    
    selected: list[dict[str, Any]] = []
    for cat in ["date_format", "currency", "measurement", "number_ambiguity"]:
        cat_rows = by_cat.get(cat, [])
        controls = [r for r in cat_rows if r["is_control"]]
        ambig = [r for r in cat_rows if not r["is_control"]]
        
        # Target: 10 per category (approx 2-3 controls, 7-8 ambiguous)
        n_ctrl = min(3, len(controls))
        n_amb = 10 - n_ctrl
        cat_selected = controls[:n_ctrl] + ambig[:n_amb]
        selected.extend(cat_selected)
    return selected

def main() -> None:
    print("=" * 75)
    print("PHASE 0b — MULTILINGUAL TRANSFER EVALUATION FOR UNCHANGED REGEX TEACHER")
    print("=" * 75)
    
    raw_rows = load_english_in_scope_rows()
    eval_rows = build_40_per_language_set(raw_rows)
    
    n_total = len(eval_rows)
    n_ctrl = sum(1 for r in eval_rows if r["is_control"])
    n_amb = sum(1 for r in eval_rows if not r["is_control"])
    print(f"English Source Set: {n_total} rows ({n_amb} ambiguous, {n_ctrl} controls [{n_ctrl/n_total:.1%}])")
    
    translator = Translator(use_mock=False)
    
    results_by_lang: dict[str, Any] = {}
    multilingual_accuracies: list[float] = []
    
    for lang_code, lang_name in LANGUAGES.items():
        print(f"\nTranslating and evaluating {lang_name} ({lang_code})...")
        
        if lang_code == "en":
            translated_questions = [r["question"] for r in eval_rows]
        else:
            translated_questions = translator.translate_batch([r["question"] for r in eval_rows], lang_code)
        
        correct_total = 0
        correct_ctrl = 0
        total_ctrl = 0
        correct_amb = 0
        total_amb = 0
        
        cat_correct: dict[str, int] = {}
        cat_total: dict[str, int] = {}
        cat_failures: dict[str, list[dict[str, Any]]] = {}
        
        for r, trans_q in zip(eval_rows, translated_questions):
            cat = r["category"]
            is_amb = r["is_ambiguous"]
            is_ctrl = r["is_control"]
            
            # Predict using UNCHANGED English regex rules
            rule = REGEX_RULES.get(cat)
            pred_amb = bool(rule.search(trans_q)) if rule else False
            is_correct = (pred_amb == is_amb)
            
            if is_correct:
                correct_total += 1
            else:
                cat_failures.setdefault(cat, []).append({
                    "id": r["id"],
                    "question_en": r["question"],
                    "question_trans": trans_q,
                    "expected": "AMBIGUOUS" if is_amb else "ANSWER",
                    "predicted": "AMBIGUOUS" if pred_amb else "ANSWER",
                })
                
            if is_ctrl:
                total_ctrl += 1
                if is_correct:
                    correct_ctrl += 1
            else:
                total_amb += 1
                if is_correct:
                    correct_amb += 1
                    
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if is_correct:
                cat_correct[cat] = cat_correct.get(cat, 0) + 1
                
        acc_total = correct_total / n_total
        acc_ctrl = correct_ctrl / total_ctrl if total_ctrl > 0 else 0.0
        acc_amb = correct_amb / total_amb if total_amb > 0 else 0.0
        
        if lang_code != "en":
            multilingual_accuracies.append(acc_total)
            
        ci_l, ci_h = wilson_ci(correct_total, n_total)
        
        cat_breakdown: dict[str, Any] = {}
        for cat in sorted(cat_total.keys()):
            c_corr = cat_correct.get(cat, 0)
            c_tot = cat_total[cat]
            c_acc = c_corr / c_tot if c_tot > 0 else 0.0
            cat_breakdown[cat] = {
                "correct": c_corr,
                "total": c_tot,
                "accuracy": round(c_acc * 100, 1),
                "failures": cat_failures.get(cat, []),
            }
            
        results_by_lang[lang_code] = {
            "language_name": lang_name,
            "n_total": n_total,
            "accuracy_overall": round(acc_total * 100, 1),
            "accuracy_controls": round(acc_ctrl * 100, 1),
            "accuracy_ambiguous": round(acc_amb * 100, 1),
            "wilson_ci_95": [round(ci_l * 100, 1), round(ci_h * 100, 1)],
            "categories": cat_breakdown,
        }

    avg_multi_acc = sum(multilingual_accuracies) / len(multilingual_accuracies) if multilingual_accuracies else 0.0

    print("\n" + "=" * 75)
    print("PHASE 0b MULTILINGUAL EVALUATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Language':<12} {'Overall Acc (%)':<18} {'Control Acc (%)':<18} {'Ambiguous Acc (%)':<18} {'95% Wilson CI'}")
    print("-" * 75)
    
    for lang_code in ["en", "es", "de", "hi", "ja", "sw"]:
        res = results_by_lang[lang_code]
        print(f"{res['language_name']:<12} {res['accuracy_overall']:>6.1f}%             {res['accuracy_controls']:>6.1f}%             {res['accuracy_ambiguous']:>6.1f}%             [{res['wilson_ci_95'][0]:.1f}%, {res['wilson_ci_95'][1]:.1f}%]")

    print("-" * 75)
    print(f"5-Language Multilingual Average Accuracy: {avg_multi_acc * 100:.1f}%\n")

    print("CATEGORY BREAKDOWN ACROSS LANGUAGES (Accuracy %):")
    headers = ["Category"] + [LANGUAGES[l] for l in ["en", "es", "de", "hi", "ja", "sw"]]
    print(f"{headers[0]:<20} " + " ".join(f"{h:>10}" for h in headers[1:]))
    print("-" * 85)
    
    for cat in ["date_format", "currency", "measurement", "number_ambiguity"]:
        row_str = f"{cat:<20}"
        for l in ["en", "es", "de", "hi", "ja", "sw"]:
            c_acc = results_by_lang[l]["categories"].get(cat, {}).get("accuracy", 0.0)
            row_str += f" {c_acc:>9.1f}%"
        print(row_str)

    output_payload = {
        "phase": "0b",
        "n_per_language": n_total,
        "n_controls_per_language": n_ctrl,
        "average_multilingual_accuracy": round(avg_multi_acc * 100, 1),
        "kill_criterion_threshold_high": 85.0,
        "kill_criterion_threshold_low": 70.0,
        "results_by_language": results_by_lang,
    }
    
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved Phase 0b results to {_OUT_JSON}")

if __name__ == "__main__":
    main()
