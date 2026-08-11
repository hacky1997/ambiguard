#!/usr/bin/env python3
"""Phase 0b — Multilingual Transfer Evaluation (Corrected with RULES_TRAIN, RULES_ORACLE, Seeded Sampling, and Audit).

1. Evaluates RULES_TRAIN (un-leakaged rules) vs RULES_ORACLE (previous rules with embedded triggers)
   across 5 target languages (es, de, hi, ja, sw) for 4 in-scope categories:
     - date_format
     - currency
     - measurement
     - number_ambiguity (numeric_scale)

2. Uses seeded random sampling (seed=42) to select 40 rows per language (10 per category).

3. Audits 20 randomly sampled translations (seed=42) for label transfer validity.

Output: eval/results/phase0b_multilingual.json
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

from scripts.crosslingual_eval import Translator
from scripts.eval_phase0_regex import RULES_TRAIN, RULES_ORACLE, wilson_ci

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
                    "note": item.get("note", ""),
                })
    return rows

def build_40_per_language_set_seeded(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Select exactly 40 balanced rows (10 per category, ~25% controls) using seed 42."""
    rng = random.Random(seed)
    
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)
    
    selected: list[dict[str, Any]] = []
    for cat in ["date_format", "currency", "measurement", "number_ambiguity"]:
        cat_rows = by_cat.get(cat, [])
        # Sort by ID for deterministic shuffle
        cat_rows = sorted(cat_rows, key=lambda x: x["id"])
        
        controls = [r for r in cat_rows if r["is_control"]]
        ambig = [r for r in cat_rows if not r["is_control"]]
        
        n_ctrl = min(3, len(controls))
        n_amb = 10 - n_ctrl
        
        sampled_controls = rng.sample(controls, n_ctrl)
        sampled_ambig = rng.sample(ambig, n_amb)
        
        selected.extend(sorted(sampled_controls + sampled_ambig, key=lambda x: x["id"]))
        
    return selected

def evaluate_ruleset(eval_rows: list[dict[str, Any]], rules: dict[str, re.Pattern[str]], translator: Translator) -> dict[str, Any]:
    n_total = len(eval_rows)
    results_by_lang: dict[str, Any] = {}
    multilingual_accs: list[float] = []
    
    for lang_code, lang_name in LANGUAGES.items():
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
        failures: list[dict[str, Any]] = []
        
        for r, trans_q in zip(eval_rows, translated_questions):
            cat = r["category"]
            is_amb = r["is_ambiguous"]
            is_ctrl = r["is_control"]
            
            rule = rules.get(cat)
            pred_amb = bool(rule.search(trans_q)) if rule else False
            is_correct = (pred_amb == is_amb)
            
            if is_correct:
                correct_total += 1
            else:
                failures.append({
                    "id": r["id"],
                    "lang": lang_code,
                    "question_en": r["question"],
                    "question_trans": trans_q,
                    "category": cat,
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
            multilingual_accs.append(acc_total)
            
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
            }
            
        results_by_lang[lang_code] = {
            "language_name": lang_name,
            "n_total": n_total,
            "accuracy_overall": round(acc_total * 100, 1),
            "accuracy_controls": round(acc_ctrl * 100, 1),
            "accuracy_ambiguous": round(acc_amb * 100, 1),
            "wilson_ci_95": [round(ci_l * 100, 1), round(ci_h * 100, 1)],
            "categories": cat_breakdown,
            "failures": failures,
        }
        
    avg_multi = sum(multilingual_accs) / len(multilingual_accs) if multilingual_accs else 0.0
    return {
        "avg_multilingual_accuracy": round(avg_multi * 100, 1),
        "results_by_language": results_by_lang,
    }

def audit_label_transfer(eval_rows: list[dict[str, Any]], translator: Translator, seed: int = 42) -> list[dict[str, Any]]:
    """Sample 20 translated rows across languages and evaluate label validity."""
    rng = random.Random(seed)
    
    # Target non-English languages
    non_en_langs = ["es", "de", "hi", "ja", "sw"]
    
    all_translated_samples: list[dict[str, Any]] = []
    for lang in non_en_langs:
        translations = translator.translate_batch([r["question"] for r in eval_rows], lang)
        for r, trans_q in zip(eval_rows, translations):
            all_translated_samples.append({
                "id": r["id"],
                "lang": lang,
                "question_en": r["question"],
                "question_trans": trans_q,
                "category": r["category"],
                "is_control": r["is_control"],
                "transferred_label": "ANSWER" if r["is_control"] else "AMBIGUOUS",
            })
            
    # Sample 20 across all 200 translated rows
    sampled_20 = rng.sample(all_translated_samples, 20)
    
    audited: list[dict[str, Any]] = []
    invalid_count = 0
    
    print("\n" + "=" * 80)
    print("STEP 3 — LABEL TRANSFER AUDIT (20 RANDOMLY SAMPLED TRANSLATIONS, SEED 42)")
    print("=" * 80)
    
    for idx, item in enumerate(sampled_20, 1):
        q_en = item["question_en"]
        q_tr = item["question_trans"]
        lang = item["lang"]
        lbl = item["transferred_label"]
        is_ctrl = item["is_control"]
        cat = item["category"]
        
        # Heuristic check for label validity:
        # e.g., if a control had "USD" or "ISO 8601 YYYY-MM-DD" and NLLB stripped/translated it into a generic form like "dollar"
        label_holds = True
        reason = "Label holds."
        
        if is_ctrl:
            if cat == "currency" and "USD" in q_en and "USD" not in q_tr and "dólar" in q_tr.lower():
                label_holds = False
                reason = "Control lost explicit ISO code 'USD' (translated to generic 'dólar'), creating ambiguity."
            elif cat == "currency" and "JPY" in q_en and "JPY" not in q_tr:
                label_holds = False
                reason = "Control lost explicit ISO code 'JPY'."
            elif cat == "currency" and "GBP" in q_en and "GBP" not in q_tr:
                label_holds = False
                reason = "Control lost explicit ISO code 'GBP'."
            elif cat == "currency" and "CAD" in q_en and "CAD" not in q_tr and "dólar" in q_tr.lower():
                label_holds = False
                reason = "Control lost explicit ISO code 'CAD'."
            elif cat == "date_format" and ("2024-05-06" in q_en or "2023-12-25" in q_en) and ("2024-05-06" not in q_tr and "2023-12-25" not in q_tr):
                label_holds = False
                reason = "Control lost ISO date format during translation."
                
        if not label_holds:
            invalid_count += 1
            
        audited_entry = {
            "index": idx,
            "id": item["id"],
            "lang": lang,
            "category": cat,
            "question_en": q_en,
            "question_trans": q_tr,
            "transferred_label": lbl,
            "label_holds": label_holds,
            "reason": reason,
        }
        audited.append(audited_entry)
        
        status_str = "VALID" if label_holds else "INVALID"
        print(f"[{idx:02d}] ID: {item['id']} ({lang}) | Cat: {cat} | Label: {lbl} | Status: [{status_str}]")
        print(f"     EN: \"{q_en}\"")
        print(f"     TR: \"{q_tr}\"")
        if not label_holds:
            print(f"     ⚠️  REASON: {reason}")
        print()
        
    print(f"Label Transfer Audit Summary: {invalid_count}/20 ({invalid_count/20:.1%}) labels invalidated by translation.")
    return audited

def main() -> None:
    print("=" * 80)
    print("PHASE 0b — MULTILINGUAL TRANSFER EVALUATION (RULES_TRAIN vs RULES_ORACLE)")
    print("=" * 80)
    
    raw_rows = load_english_in_scope_rows()
    eval_rows = build_40_per_language_set_seeded(raw_rows, seed=42)
    
    selected_ids = [r["id"] for r in eval_rows]
    print(f"\nStep 2 — Seeded Selection (Seed 42, {len(eval_rows)} rows total):")
    print("Selected Row IDs:", ", ".join(selected_ids))
    
    translator = Translator(use_mock=False)
    
    print("\nRunning RULES_TRAIN (un-leakaged rules)...")
    res_train = evaluate_ruleset(eval_rows, RULES_TRAIN, translator)
    
    print("\nRunning RULES_ORACLE (previous rules with embedded triggers)...")
    res_oracle = evaluate_ruleset(eval_rows, RULES_ORACLE, translator)
    
    audited_20 = audit_label_transfer(eval_rows, translator, seed=42)
    invalid_count = sum(1 for a in audited_20 if not a["label_holds"])
    
    print("\n" + "=" * 85)
    print("COMPARISON: RULES_TRAIN (UN-LEAKAGED) VS RULES_ORACLE (PREVIOUS LEAKED)")
    print("=" * 85)
    print(f"{'Language':<12} {'RULES_TRAIN Acc (%)':<22} {'RULES_ORACLE Acc (%)':<22} {'Leakage Delta'}")
    print("-" * 85)
    
    for lang_code in ["en", "es", "de", "hi", "ja", "sw"]:
        t_acc = res_train["results_by_language"][lang_code]["accuracy_overall"]
        o_acc = res_oracle["results_by_language"][lang_code]["accuracy_overall"]
        delta = o_acc - t_acc
        lang_name = LANGUAGES[lang_code]
        print(f"{lang_name:<12} {t_acc:>6.1f}%                 {o_acc:>6.1f}%                 {delta:>+5.1f} pts")
        
    print("-" * 85)
    t_avg = res_train["avg_multilingual_accuracy"]
    o_avg = res_oracle["avg_multilingual_accuracy"]
    print(f"5-Lang Avg   {t_avg:>6.1f}%                 {o_avg:>6.1f}%                 {o_avg - t_avg:>+5.1f} pts\n")
    
    print("PER-CATEGORY BREAKDOWN: RULES_TRAIN vs RULES_ORACLE (% Accuracy):")
    headers = ["Category"] + [f"{LANGUAGES[l]} (Tr/Or)" for l in ["en", "es", "de", "hi", "ja", "sw"]]
    print(f"{'Category':<18} " + " ".join(f"{h:>15}" for h in headers[1:]))
    print("-" * 110)
    
    for cat in ["date_format", "currency", "measurement", "number_ambiguity"]:
        row_str = f"{cat:<18}"
        for l in ["en", "es", "de", "hi", "ja", "sw"]:
            t_acc = res_train["results_by_language"][l]["categories"].get(cat, {}).get("accuracy", 0.0)
            o_acc = res_oracle["results_by_language"][l]["categories"].get(cat, {}).get("accuracy", 0.0)
            row_str += f"   {t_acc:>5.1f}/{o_acc:<5.1f}"
        print(row_str)
        
    output_payload = {
        "phase": "0b-corrected",
        "seed": 42,
        "selected_row_ids": selected_ids,
        "rules_train": res_train,
        "rules_oracle": res_oracle,
        "multilingual_leakage_delta": round(o_avg - t_avg, 1),
        "label_transfer_audit_20": audited_20,
        "invalid_label_transfer_count": invalid_count,
        "invalid_label_transfer_rate": round(invalid_count / 20 * 100, 1),
    }
    
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved Phase 0b results to {_OUT_JSON}")

if __name__ == "__main__":
    main()
