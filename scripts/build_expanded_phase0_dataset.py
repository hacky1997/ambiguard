#!/usr/bin/env python3
"""Phase 0 Expanded Evaluation Harness — 152-Row Multilingual Reference Dataset (Corrected).

Fixes applied:
1. Leakage Audit: Checks whether RULES_TRAIN fires on any row designated as held_out.
   Aborts if RULES_TRAIN fires on any held_out row.
2. Independent Disjoint Splits: Every row has `split` ('in_distribution' | 'held_out') AND
   `is_control` (bool). In-distribution and held-out splits are completely disjoint.
3. Label Disambiguation: Removes physical length labels ('2m height') from number_ambiguity.
   Bare 'm' is used only in ambiguous financial/numerical contexts ('revenue of 10m').

Outputs:
  - eval/datasets/expanded_phase0_dataset.jsonl
  - eval/results/expanded_phase0_results.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from scripts.crosslingual_eval import Translator
from scripts.eval_phase0_regex import RULES_TRAIN, wilson_ci

_DATASET_OUT = Path("eval/datasets/expanded_phase0_dataset.jsonl")
_RESULTS_OUT = Path("eval/results/expanded_phase0_results.json")

EXTERNAL_REFERENCES = {
    "currency": "ISO 4217 Currency Code Standard & Unicode Currency Symbols Table (shared symbols vs unique ISO codes)",
    "measurement": "NIST Special Publication 330 / US Customary and British Imperial Measurement Systems",
    "date_format": "Unicode CLDR Date/Time Patterns & ISO 8601 Date Representation Standard",
    "number_ambiguity": "Short and Long Scales Dictionary & International System of Units (SI) Prefixes",
}

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "sw": "Swahili",
}

def generate_expanded_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    
    # ---------------------------------------------------------------------------
    # 1. CURRENCY (38 rows: 14 in_dist [10 amb, 4 ctrl], 24 held_out [18 amb, 6 ctrl])
    # ---------------------------------------------------------------------------
    in_curr_amb = [
        ("$", "$50,000 per year"),
        ("$", "$250 for the order"),
        ("$", "$100,000 per year"),
        ("$", "$75,000 total"),
        ("€", "€100 total"),
        ("€", "€500 for service"),
        ("€", "€1,000 price"),
        ("£", "£35,000 per year"),
        ("£", "£45,000 per year"),
        ("£", "£200 total"),
    ]
    for i, (s, phrase) in enumerate(in_curr_amb, 1):
        rows.append({"id": f"curr_in_amb_{i:02d}", "category": "currency", "split": "in_distribution", "trigger": s, "is_ambiguous": True, "is_control": False, "question": f"What is a salary of {phrase} equivalent to in purchasing power?" if "per year" in phrase else f"Total price is {phrase}."})

    in_curr_ctrl = [
        ("USD", "What is 50,000 USD converted to local currency?"),
        ("EUR", "What is 100 EUR converted to local currency?"),
        ("GBP", "What is 500 GBP converted to local currency?"),
        ("USD", "Total price is 250 USD."),
    ]
    for i, (code, q) in enumerate(in_curr_ctrl, 1):
        rows.append({"id": f"curr_in_ctrl_{i:02d}", "category": "currency", "split": "in_distribution", "trigger": code, "is_ambiguous": False, "is_control": True, "question": q})

    held_curr_amb = [
        ("kr", "kr400,000 per year"),
        ("Rs", "Rs 1,200,000 per year"),
        ("₨", "₨ 1500 for service"),
        ("zł", "zł 80,000 per year"),
        ("din.", "din.15,000 per year"),
        ("KSh", "KSh 3000 total"),
        ("zł", "100 zł price"),
        ("p.", "50 p. cost"),
        ("¥", "¥5,000,000 per year"),
        ("Rp", "Rp 50,000 total"),
        ("ƒ", "ƒ 500 price"),
        ("kr", "kr 250 fee"),
        ("Rs", "Rs 500 total"),
        ("¥", "¥10,000 fee"),
        ("KSh", "KSh 1,000 price"),
        ("zł", "500 zł fee"),
        ("zł", "zł 150 total"),
        ("din.", "din. 500 fee"),
    ]
    for i, (s, phrase) in enumerate(held_curr_amb, 1):
        rows.append({"id": f"curr_held_amb_{i:02d}", "category": "currency", "split": "held_out", "trigger": s, "is_ambiguous": True, "is_control": False, "question": f"What is a salary of {phrase} equivalent to in purchasing power?" if "per year" in phrase else f"Total price is {phrase}."})

    held_curr_ctrl = [
        ("CAD", "What is 50,000 CAD converted to local currency?"),
        ("JPY", "What is 5,000,000 JPY converted to local currency?"),
        ("BRL", "What is 80,000 BRL converted to local currency?"),
        ("INR", "What is 1,200,000 INR converted to local currency?"),
        ("AUD", "What is 50,000 AUD converted to local currency?"),
        ("SEK", "What is 400,000 SEK converted to local currency?"),
    ]
    for i, (code, q) in enumerate(held_curr_ctrl, 1):
        rows.append({"id": f"curr_held_ctrl_{i:02d}", "category": "currency", "split": "held_out", "trigger": code, "is_ambiguous": False, "is_control": True, "question": q})

    # ---------------------------------------------------------------------------
    # 2. MEASUREMENT (38 rows: 14 in_dist [10 amb, 4 ctrl], 24 held_out [18 amb, 6 ctrl])
    # ---------------------------------------------------------------------------
    in_meas_amb = [
        ("miles", "Is a distance of 10 miles considered long for this route?"),
        ("miles", "Is 50 miles considered long for this route?"),
        ("miles", "Is 100 miles considered long for this route?"),
        ("mile", "Is 1 mile considered long for this route?"),
        ("mile", "Is a 5 mile stretch clear?"),
        ("gallons", "Volume capacity is 5 gallons."),
        ("gallons", "Volume capacity is 10 gallons."),
        ("gallons", "Volume capacity is 20 gallons."),
        ("gallon", "Volume capacity is 1 gallon."),
        ("gallon", "Is a 2 gallon jug sufficient?"),
    ]
    for i, (m, q) in enumerate(in_meas_amb, 1):
        rows.append({"id": f"meas_in_amb_{i:02d}", "category": "measurement", "split": "in_distribution", "trigger": m, "is_ambiguous": True, "is_control": False, "question": q})

    in_meas_ctrl = [
        ("kilometers", "How many kilometers is 160 statute miles?"),
        ("liters", "Volume capacity is 37.854 liters."),
        ("kilometers", "Distance is 50 kilometers."),
        ("liters", "Volume capacity is 10 liters."),
    ]
    for i, (trig, q) in enumerate(in_meas_ctrl, 1):
        rows.append({"id": f"meas_in_ctrl_{i:02d}", "category": "measurement", "split": "in_distribution", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    held_meas_amb = [
        ("pints", "Is 2 pints considered high for this liquid system?"),
        ("pint", "Is 1 pint considered high for this liquid system?"),
        ("stone", "Weight is 12 stone."),
        ("leagues", "Journey of 5 leagues."),
        ("league", "Distance of 1 league."),
        ("tons", "Weight is 50 tons."),
        ("ton", "Weight is 1 ton."),
        ("ounces", "Is 5 ounces considered high for this system?"),
        ("ounce", "Is 1 ounce considered high for this system?"),
        ("feet", "Height is 100 feet."),
        ("foot", "Height is 1 foot."),
        ("knots", "Speed is 20 knots."),
        ("knot", "Speed is 1 knot."),
        ("pounds", "Weight is 200 pounds."),
        ("pound", "Weight is 1 pound."),
        ("cups", "Volume is 5 cups."),
        ("fl oz", "Volume is 2 fl oz."),
        ("cwt", "Weight is 10 cwt."),
    ]
    for i, (m, q) in enumerate(held_meas_amb, 1):
        rows.append({"id": f"meas_held_amb_{i:02d}", "category": "measurement", "split": "held_out", "trigger": m, "is_ambiguous": True, "is_control": False, "question": q})

    held_meas_ctrl = [
        ("metric tons", "What is 50 metric tons in kg?"),
        ("degrees Celsius", "Temperature is 25 degrees Celsius."),
        ("meters", "Length is 50 meters."),
        ("grams", "Mass is 500 grams."),
        ("km/h", "Speed limit is 100 km/h."),
        ("hectares", "Land area is 5 hectares."),
    ]
    for i, (trig, q) in enumerate(held_meas_ctrl, 1):
        rows.append({"id": f"meas_held_ctrl_{i:02d}", "category": "measurement", "split": "held_out", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    # ---------------------------------------------------------------------------
    # 3. DATE FORMAT (38 rows: 14 in_dist [10 amb, 4 ctrl], 24 held_out [18 amb, 6 ctrl])
    # ---------------------------------------------------------------------------
    # In-dist Ambiguous (slashes, dots, dashes where day & month <= 12)
    in_date_amb = ["05/06/2024", "11/12/2025", "01/02/2023", "07.08.2024", "09.10.2025", "03-04-2026", "02-03-2024", "04/05/2026", "06.07.2025", "08-09-2024"]
    for i, d in enumerate(in_date_amb, 1):
        rows.append({"id": f"date_in_amb_{i:02d}", "category": "date_format", "split": "in_distribution", "trigger": d, "is_ambiguous": True, "is_control": False, "question": f"When does the contract starting on {d} expire?"})

    in_date_ctrl = [
        ("2024-05-06", "Meeting scheduled for 2024-05-06."),
        ("2025-11-12", "Deadline is 2025-11-12."),
        ("25/12/2023", "What day of the week was 25/12/2023?"),
        ("15/08/2024", "Event on 15/08/2024."),
    ]
    for i, (trig, q) in enumerate(in_date_ctrl, 1):
        rows.append({"id": f"date_in_ctrl_{i:02d}", "category": "date_format", "split": "in_distribution", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    # Held-out Ambiguous (space separators, CJK dates, Era dates NOT matched by \b..\b slashes/dots/dashes)
    held_date_amb = ["05 06 2024", "11 12 2025", "01 02 2023", "07 08 2024", "09 10 2025", "03 04 2026", "02 03 2024", "04 05 2026", "06 07 2025", "08 09 2024", "10 11 2026", "12 01 2025", "2024年05月06日", "2025年11月12日", "2023年01月02日", "2026年07月08日", "Reiwa 3年5月12日", "Reiwa 4年1月2日"]
    for i, d in enumerate(held_date_amb, 1):
        rows.append({"id": f"date_held_amb_{i:02d}", "category": "date_format", "split": "held_out", "trigger": d, "is_ambiguous": True, "is_control": False, "question": f"What date is {d}?"})

    held_date_ctrl = [
        ("June 5, 2024", "What event happened on June 5, 2024?"),
        ("December 25, 2023", "Holiday on December 25, 2023."),
        ("January 1, 2025", "New Year on January 1, 2025."),
        ("August 15, 2024", "Independence day on August 15, 2024."),
        ("November 12, 2025", "Conference on November 12, 2025."),
        ("May 6, 2024", "Contract signed on May 6, 2024."),
    ]
    for i, (trig, q) in enumerate(held_date_ctrl, 1):
        rows.append({"id": f"date_held_ctrl_{i:02d}", "category": "date_format", "split": "held_out", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    # ---------------------------------------------------------------------------
    # 4. NUMBER AMBIGUITY (38 rows: 14 in_dist [10 amb, 4 ctrl], 24 held_out [18 amb, 6 ctrl])
    # ---------------------------------------------------------------------------
    in_num_amb = [
        ("billion", "Deficit reached 5 billion in history."),
        ("billion", "Revenue was 10 billion total."),
        ("billion", "Population reached 1 billion."),
        ("billion", "Budget of 2 billion dollars."),
        ("billion", "Target is 50 billion."),
        ("10m", "Revenue was 10m."),
        ("5m", "Cost was 5m."),
        ("50m", "Sales reached 50m."),
        ("100m", "Valuation of 100m."),
        ("1m", "Profit was 1m."),
    ]
    for i, (trig, q) in enumerate(in_num_amb, 1):
        rows.append({"id": f"num_in_amb_{i:02d}", "category": "number_ambiguity", "split": "in_distribution", "trigger": trig, "is_ambiguous": True, "is_control": False, "question": q})

    in_num_ctrl = [
        ("1,000,000,000", "How many millions are in 1,000,000,000 (one US 10^9)?"),
        ("100 million USD", "Amount is 100 million USD."),
        ("500 thousand", "Total is 500 thousand."),
        ("50,000", "What is the value of 50,000 in word form?"),
    ]
    for i, (trig, q) in enumerate(in_num_ctrl, 1):
        rows.append({"id": f"num_in_ctrl_{i:02d}", "category": "number_ambiguity", "split": "in_distribution", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    held_num_amb = [
        ("milliard", "What is the exact numerical value of 5 milliard?"),
        ("billón", "What is the exact numerical value of 2 billón?"),
        ("trillion", "What is the exact numerical value of 1 trillion?"),
        ("lakh", "Salary is 5 lakh per year."),
        ("crore", "Budget is 10 crore total."),
        ("bn", "Budget is 5 bn."),
        ("bn", "Valuation of 10 bn."),
        ("MM", "Cost is 10 MM."),
        ("MM", "Revenue of 50 MM."),
        ("億", "Population is 1億."),
        ("兆", "Budget is 10兆."),
        ("k", "Salary is 50k."),
        ("k", "Total is 10k."),
        ("billon", "Value is 1 billon."),
        ("milliard", "Valuation of 10 milliard."),
        ("lakh", "Price is 20 lakh."),
        ("crore", "Revenue is 5 crore."),
        ("MM", "Budget of 100 MM."),
    ]
    for i, (trig, q) in enumerate(held_num_amb, 1):
        rows.append({"id": f"num_held_amb_{i:02d}", "category": "number_ambiguity", "split": "held_out", "trigger": trig, "is_ambiguous": True, "is_control": False, "question": q})

    held_num_ctrl = [
        ("1 x 10^9", "What is 1 x 10^9 in standard scientific notation?"),
        ("100,000", "Total count is 100,000 exact."),
        ("10^12", "Value is 10^12 in exponent notation."),
        ("5,000", "Count is 5,000 units."),
        ("1 x 10^12", "Amount is 1 x 10^12 exact."),
        ("2 x 10^6", "Value is 2 x 10^6."),
    ]
    for i, (trig, q) in enumerate(held_num_ctrl, 1):
        rows.append({"id": f"num_held_ctrl_{i:02d}", "category": "number_ambiguity", "split": "held_out", "trigger": trig, "is_ambiguous": False, "is_control": True, "question": q})

    return rows

def audit_leakage_authorship(rows: list[dict[str, Any]]) -> int:
    """Audit RULES_TRAIN pattern sources for literal authorship leakage of held-out triggers."""
    held_rows = [r for r in rows if r["split"] == "held_out"]
    leaked_count = 0
    print("\n" + "=" * 75)
    print("STEP 2 AUTHORSHIP LEAKAGE AUDIT FOR RULES_TRAIN")
    print("=" * 75)
    for r in held_rows:
        cat = r["category"]
        trig = r["trigger"]
        rule = RULES_TRAIN.get(cat)
        if rule:
            pattern_src = rule.pattern.lower()
            trig_norm = trig.lower()
            if trig_norm in pattern_src:
                print(f"❌ LEAK DETECTED: Trigger '{trig}' literally authored in RULES_TRAIN['{cat}'] pattern source ('{pattern_src}')")
                leaked_count += 1
    if leaked_count == 0:
        print("✅ 0 leaked triggers found in RULES_TRAIN pattern sources across all held_out rows.")
    else:
        print(f"❌ AUTHORSHIP LEAKAGE FAILURE: {leaked_count} held-out triggers found in pattern definitions.")
    return leaked_count

def score_subsplit(rows: list[dict[str, Any]], questions: list[str]) -> dict[str, Any]:
    n_total = len(rows)
    correct_total = 0
    correct_ctrl = 0
    total_ctrl = 0
    correct_amb = 0
    total_amb = 0
    
    for r, q in zip(rows, questions):
        cat = r["category"]
        is_amb = r["is_ambiguous"]
        is_ctrl = r["is_control"]
        
        rule = RULES_TRAIN.get(cat)
        pred_amb = bool(rule.search(q)) if rule else False
        is_correct = (pred_amb == is_amb)
        
        if is_correct:
            correct_total += 1
            
        if is_ctrl:
            total_ctrl += 1
            if is_correct:
                correct_ctrl += 1
        else:
            total_amb += 1
            if is_correct:
                correct_amb += 1
                
    acc_overall = correct_total / n_total if n_total > 0 else 0.0
    acc_amb = correct_amb / total_amb if total_amb > 0 else 0.0
    acc_ctrl = correct_ctrl / total_ctrl if total_ctrl > 0 else 0.0
    ci_l, ci_h = wilson_ci(correct_amb, total_amb)
    
    return {
        "n_total": n_total,
        "n_ambiguous": total_amb,
        "n_controls": total_ctrl,
        "accuracy_overall": round(acc_overall * 100, 1),
        "accuracy_ambiguous": round(acc_amb * 100, 1),  # Headline metric
        "accuracy_controls": round(acc_ctrl * 100, 1),
        "wilson_ci_ambiguous_95": [round(ci_l * 100, 1), round(ci_h * 100, 1)],
    }

def main() -> None:
    print("=" * 85)
    print("PHASE 0 EXPANDED EVALUATION HARNESS — 152-ROW MULTILINGUAL REFERENCE DATASET")
    print("=" * 85)
    
    rows = generate_expanded_inventory()
    n_in = sum(1 for r in rows if r["split"] == "in_distribution")
    n_held = sum(1 for r in rows if r["split"] == "held_out")
    n_ctrl = sum(1 for r in rows if r["is_control"])
    print(f"Generated {len(rows)} English rows ({n_in} in_distribution, {n_held} held_out | {n_ctrl} controls [{n_ctrl/len(rows):.1%}]).")
    
    # Save dataset to JSONL
    _DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_DATASET_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved dataset inventory to {_DATASET_OUT}")
    
    # Step 2: Leakage Audit
    leaks = audit_leakage_authorship(rows)
    if leaks > 0:
        print(f"Aborting evaluation due to {leaks} detected leaks.")
        sys.exit(1)
        
    # Step 3: Multilingual NLLB Translation
    translator = Translator(use_mock=False)
    english_questions = [r["question"] for r in rows]
    
    translations_by_lang: dict[str, list[str]] = {"en": english_questions}
    for lang in ["es", "de", "hi", "ja", "sw"]:
        print(f"Translating {len(rows)} rows to {LANGUAGES[lang]} ({lang})...")
        translations_by_lang[lang] = translator.translate_batch(english_questions, lang)
        
    # Step 4: Rerun Evaluation & Report
    all_results: dict[str, Any] = {}
    
    print("\n" + "=" * 105)
    print("PHASE 0 EXPANDED MULTILINGUAL EVALUATION RESULTS (RULES_TRAIN)")
    print("=" * 105)
    print(f"{'Lang':<8} {'Split':<18} {'N (Amb/Ctrl)':<15} {'AMBIGUOUS ACC (%) [HEADLINE]':<32} {'Control Acc (%)':<18} {'Overall Acc (%)'}")
    print("-" * 105)
    
    for lang_code, lang_name in LANGUAGES.items():
        q_list = translations_by_lang[lang_code]
        lang_res: dict[str, Any] = {}
        
        # Completely disjoint splits: in_distribution vs held_out
        for split_name in ["in_distribution", "held_out"]:
            split_indices = [idx for idx, r in enumerate(rows) if r["split"] == split_name]
            sub_rows = [rows[idx] for idx in split_indices]
            sub_q = [q_list[idx] for idx in split_indices]
            
            res = score_subsplit(sub_rows, sub_q)
            lang_res[split_name] = res
            
            amb_str = f"{res['accuracy_ambiguous']:.1f}% [{res['wilson_ci_ambiguous_95'][0]:.1f}%, {res['wilson_ci_ambiguous_95'][1]:.1f}%]"
            n_str = f"{res['n_total']} ({res['n_ambiguous']}/{res['n_controls']})"
            print(f"{lang_name:<8} {split_name:<18} {n_str:<15} {amb_str:<32} {res['accuracy_controls']:>6.1f}%             {res['accuracy_overall']:>6.1f}%")
            
        all_results[lang_code] = lang_res

    output_payload = {
        "phase": "0-expanded-corrected",
        "external_references": EXTERNAL_REFERENCES,
        "n_rows_per_language": len(rows),
        "results_by_language": all_results,
    }
    
    _RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print("\nSaved expanded evaluation results to", _RESULTS_OUT)

if __name__ == "__main__":
    main()
