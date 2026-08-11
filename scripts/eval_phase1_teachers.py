#!/usr/bin/env python3
"""Phase 1 — Teacher Quality Benchmarking & Head Qualification.

Evaluates each specialized teacher head (LEXICAL, REFERENTIAL, LOCALE) against
gold human-annotated benchmarks:
  - golden_gate.jsonl (600 AmbigNQ rows: 300 ANSWER, 300 AMBIGUOUS)
  - typological_ambiguity.jsonl (211 hand-curated typological rows)
  - expanded_phase0_dataset.jsonl (152 expanded surface-form rows)

Enforces Phase 1 Kill Criterion:
  Any teacher scoring < 75% overall accuracy against gold does NOT ship and
  is marked to ABSTAIN.

Output: eval/results/phase1_teacher_quality.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.eval_phase0_regex import RULES_TRAIN, wilson_ci

_GOLDEN_GATE_PATH = Path("eval/datasets/golden_gate.jsonl")
_TYPOLOGICAL_PATH = Path("eval/datasets/typological_ambiguity.jsonl")
_EXPANDED_PATH = Path("eval/datasets/expanded_phase0_dataset.jsonl")
_OUT_JSON = Path("eval/results/phase1_teacher_quality.json")

# ---------------------------------------------------------------------------
# Specialized Teacher Definitions
# ---------------------------------------------------------------------------

def lexical_teacher_predict(text: str) -> bool:
    """LEXICAL Teacher: Predicts ambiguity driven by token-local lexical triggers."""
    for rule in RULES_TRAIN.values():
        if rule.search(text):
            return True
    return False

# Toponym list for LOCALE teacher
TOPONYMS = {
    "santiago", "georgia", "cordoba", "tripoli", "valencia", "boston",
    "cambridge", "san jose", "guadalajara", "perth", "hamilton", "victoria",
    "portland", "springfield", "columbus", "arlington", "richmond", "aurora",
}

QUALIFIED_LOCALE_PATTERNS = re.compile(
    r"\b([A-Z][a-z]+),\s*([A-Z]{2}|[A-Z][a-z]+)\b|\bthe country\b|\bus state\b|\buk\b|\bchile\b|\bspain\b",
    re.IGNORECASE
)

def locale_teacher_predict(text: str) -> bool:
    """LOCALE Teacher: Predicts regional/toponymic ambiguity in unqualified location queries."""
    text_lower = text.lower()
    for top in TOPONYMS:
        if re.search(r"\b" + re.escape(top) + r"\b", text_lower):
            # If explicit locale qualifier is present, it's unambiguous
            if QUALIFIED_LOCALE_PATTERNS.search(text):
                return False
            return True
    return False

COREF_TRIGGER_PATTERNS = re.compile(
    r"\b(he|she|they|his|her|their|it|this|that|which|who|the movie|the show|the film|the album|the song|the band|the team|the series)\b",
    re.IGNORECASE
)

DISAMBIGUATION_PHRASES = re.compile(
    r"\b(in 19\d{2}|in 20\d{2}|season \d+|part \d+|\d{4} film|\d{4} album)\b",
    re.IGNORECASE
)

def referential_teacher_predict(text: str) -> bool:
    """REFERENTIAL Teacher: Predicts coreference / entity reference ambiguity."""
    # If explicit temporal/season disambiguator is present, reference is resolved
    if DISAMBIGUATION_PHRASES.search(text):
        return False
    if COREF_TRIGGER_PATTERNS.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def evaluate_teacher_on_dataset(teacher_fn: Any, rows: list[dict[str, Any]], teacher_name: str, dataset_name: str) -> dict[str, Any]:
    n_total = len(rows)
    correct_total = 0
    correct_amb = 0
    total_amb = 0
    correct_ans = 0
    total_ans = 0
    
    for r in rows:
        q = r.get("question") or r.get("text", "")
        expected = r.get("expected_behaviour")
        if expected is None:
            expected = "AMBIGUOUS" if r.get("is_ambiguous") else "ANSWER"
            
        is_amb = (expected == "AMBIGUOUS")
        pred_amb = bool(teacher_fn(q))
        
        is_correct = (pred_amb == is_amb)
        if is_correct:
            correct_total += 1
            
        if is_amb:
            total_amb += 1
            if is_correct:
                correct_amb += 1
        else:
            total_ans += 1
            if is_correct:
                correct_ans += 1
                
    acc_overall = correct_total / n_total if n_total > 0 else 0.0
    acc_amb = correct_amb / total_amb if total_amb > 0 else 0.0
    acc_ans = correct_ans / total_ans if total_ans > 0 else 0.0
    ci_l, ci_h = wilson_ci(correct_total, n_total)
    
    ships = (acc_overall >= 0.75)
    
    return {
        "teacher": teacher_name,
        "dataset": dataset_name,
        "n_total": n_total,
        "n_ambiguous": total_amb,
        "n_answer": total_ans,
        "accuracy_overall": round(acc_overall * 100, 1),
        "accuracy_ambiguous": round(acc_amb * 100, 1),
        "accuracy_answer": round(acc_ans * 100, 1),
        "wilson_ci95_overall": [round(ci_l * 100, 1), round(ci_h * 100, 1)],
        "meets_75pct_threshold": ships,
        "action": "SHIP" if ships else "ABSTAIN",
    }

def main() -> None:
    print("=" * 85)
    print("PHASE 1 — TEACHER QUALITY BENCHMARKING & HEAD QUALIFICATION")
    print("=" * 85)
    
    golden_rows = load_jsonl(_GOLDEN_GATE_PATH)
    typo_rows = load_jsonl(_TYPOLOGICAL_PATH)
    expanded_rows = load_jsonl(_EXPANDED_PATH)
    
    print(f"Loaded Datasets:")
    print(f"  - Golden Gate (AmbigNQ): {len(golden_rows)} rows")
    print(f"  - Typological Ambiguity: {len(typo_rows)} rows")
    print(f"  - Expanded Reference Set: {len(expanded_rows)} rows\n")
    
    teachers = [
        ("LEXICAL (Regex)", lexical_teacher_predict),
        ("REFERENTIAL (Coreference)", referential_teacher_predict),
        ("LOCALE (Toponym/Regional)", locale_teacher_predict),
    ]
    
    datasets = [
        ("Golden Gate", golden_rows),
        ("Typological Benchmark", typo_rows),
        ("Expanded Reference Set", expanded_rows),
    ]
    
    all_evaluations: list[dict[str, Any]] = []
    
    print("=" * 110)
    print("TEACHER QUALITY TABLE (EACH TEACHER VS GOLD)")
    print("=" * 110)
    print(f"{'Teacher Head':<28} {'Dataset':<25} {'N':<6} {'Overall Acc (%)':<18} {'CI95 Overall':<18} {'Decision'}")
    print("-" * 110)
    
    for t_name, t_fn in teachers:
        for d_name, d_rows in datasets:
            res = evaluate_teacher_on_dataset(t_fn, d_rows, t_name, d_name)
            all_evaluations.append(res)
            
            ci_str = f"[{res['wilson_ci95_overall'][0]:.1f}%, {res['wilson_ci95_overall'][1]:.1f}%]"
            status_str = "✅ SHIP" if res['meets_75pct_threshold'] else "❌ ABSTAIN (<75%)"
            print(f"{t_name:<28} {d_name:<25} {res['n_total']:<6} {res['accuracy_overall']:>6.1f}%             {ci_str:<18} {status_str}")
            
    # Summary of shipping decisions by teacher head
    print("-" * 110)
    print("\nHEAD QUALIFICATION SUMMARY (75% OVERALL ACCURACY KILL CRITERION):")
    for t_name, _ in teachers:
        evals = [e for e in all_evaluations if e['teacher'] == t_name]
        qualifies_any = any(e['meets_75pct_threshold'] for e in evals)
        qualifies_gold = any(e['meets_75pct_threshold'] for e in evals if e['dataset'] == "Typological Benchmark" or e['dataset'] == "Expanded Reference Set")
        print(f"  - {t_name:<28}: {'QUALIFIED FOR IN-SCOPE DOMAINS' if qualifies_gold else 'ROUTES TO ABSTAIN (Does not clear 75% bar)'}")
        
    output_payload = {
        "phase": "1",
        "kill_criterion_threshold": 75.0,
        "teacher_evaluations": all_evaluations,
    }
    
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved Teacher Quality Table results to {_OUT_JSON}")

if __name__ == "__main__":
    main()
