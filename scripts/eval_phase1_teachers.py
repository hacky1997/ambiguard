#!/usr/bin/env python3
"""Phase 1 — Scoped Teacher Quality Benchmarking & Head Qualification.

Evaluates each specialized teacher head (LEXICAL, REFERENTIAL, LOCALE) STRICTLY
within its in-domain scoped subset where its trigger class applies:

  - LEXICAL: Evaluated on lexical categories (currency, date_format, measurement, number_ambiguity)
    across Expanded Set (in_distribution vs held_out) and Typological Benchmark.
  - REFERENTIAL: Evaluated on coreference/referential categories (subject_drop, word_order, gender, honorific)
    and Golden Gate coref queries.
  - LOCALE: Evaluated on locale/toponymic categories (entity_collision, formality, script_variant, calendar)
    and Golden Gate toponym queries.

Enforces Phase 1 Kill Criterion:
  Any teacher scoring < 75% in-domain accuracy against gold does NOT ship and
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

TOPONYMS = {
    "santiago", "georgia", "cordoba", "tripoli", "valencia", "boston",
    "cambridge", "san jose", "guadalajara", "perth", "hamilton", "victoria",
    "portland", "springfield", "columbus", "arlington", "richmond", "aurora",
    "eugene", "oregon", "dallas", "texas",
}

QUALIFIED_LOCALE_PATTERNS = re.compile(
    r"\b([A-Z][a-z]+),\s*([A-Z]{2}|[A-Z][a-z]+)\b|\bthe country\b|\bus state\b|\buk\b|\bchile\b|\bspain\b|\boregon\b|\btexas\b",
    re.IGNORECASE
)

def locale_teacher_predict(text: str) -> bool:
    """LOCALE Teacher: Predicts regional/toponymic ambiguity in unqualified location queries."""
    text_lower = text.lower()
    for top in TOPONYMS:
        if re.search(r"\b" + re.escape(top) + r"\b", text_lower):
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

def evaluate_teacher_on_scoped_rows(teacher_fn: Any, rows: list[dict[str, Any]], teacher_name: str, domain_label: str) -> dict[str, Any]:
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
        "domain_scope": domain_label,
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
    print("=" * 95)
    print("PHASE 1 — SCOPED IN-DOMAIN TEACHER QUALITY BENCHMARKING & HEAD QUALIFICATION")
    print("=" * 95)
    
    golden_rows = load_jsonl(_GOLDEN_GATE_PATH)
    typo_rows = load_jsonl(_TYPOLOGICAL_PATH)
    expanded_rows = load_jsonl(_EXPANDED_PATH)
    
    # ---------------------------------------------------------------------------
    # Scoped Subdomains for Each Teacher Head
    # ---------------------------------------------------------------------------
    lexical_cats = {"currency", "date_format", "measurement", "numeric_scale", "number_ambiguity"}
    referential_cats = {"subject_drop", "word_order", "gender", "honorific"}
    locale_cats = {"entity_collision", "formality", "script_variant", "calendar", "code_switching"}
    
    # 1. LEXICAL Teacher In-Domain Scopes
    lex_expanded_in = [r for r in expanded_rows if r.get("split") == "in_distribution"]
    lex_expanded_held = [r for r in expanded_rows if r.get("split") == "held_out"]
    lex_typo = [r for r in typo_rows if r.get("category") in lexical_cats]
    
    # 2. REFERENTIAL Teacher In-Domain Scopes
    ref_typo = [r for r in typo_rows if r.get("category") in referential_cats]
    ref_golden = [r for r in golden_rows if COREF_TRIGGER_PATTERNS.search(r.get("question", ""))]
    
    # 3. LOCALE Teacher In-Domain Scopes
    loc_typo = [r for r in typo_rows if r.get("category") in locale_cats]
    loc_golden = [r for r in golden_rows if any(t in r.get("question", "").lower() for t in TOPONYMS)]
    
    eval_tasks = [
        # LEXICAL
        (lexical_teacher_predict, "LEXICAL (Regex)", "Expanded Set (in_distribution)", lex_expanded_in),
        (lexical_teacher_predict, "LEXICAL (Regex)", "Expanded Set (held_out)", lex_expanded_held),
        (lexical_teacher_predict, "LEXICAL (Regex)", "Typological (Lexical Subdomain)", lex_typo),
        
        # REFERENTIAL
        (referential_teacher_predict, "REFERENTIAL (Coref)", "Typological (Coref Subdomain)", ref_typo),
        (referential_teacher_predict, "REFERENTIAL (Coref)", "Golden Gate (Coref Subdomain)", ref_golden),
        
        # LOCALE
        (locale_teacher_predict, "LOCALE (Toponym)", "Typological (Locale Subdomain)", loc_typo),
        (locale_teacher_predict, "LOCALE (Toponym)", "Golden Gate (Toponym Subdomain)", loc_golden),
    ]
    
    all_evaluations: list[dict[str, Any]] = []
    
    print("=" * 115)
    print("SCOPED IN-DOMAIN TEACHER QUALITY TABLE")
    print("=" * 115)
    print(f"{'Teacher Head':<24} {'Scoped Domain / Dataset':<36} {'N':<6} {'Overall Acc (%)':<18} {'CI95 Overall':<18} {'Decision'}")
    print("-" * 115)
    
    for t_fn, t_name, d_scope, rows in eval_tasks:
        res = evaluate_teacher_on_scoped_rows(t_fn, rows, t_name, d_scope)
        all_evaluations.append(res)
        
        ci_str = f"[{res['wilson_ci95_overall'][0]:.1f}%, {res['wilson_ci95_overall'][1]:.1f}%]"
        status_str = "✅ SHIP" if res['meets_75pct_threshold'] else "❌ ABSTAIN (<75%)"
        print(f"{t_name:<24} {d_scope:<36} {res['n_total']:<6} {res['accuracy_overall']:>6.1f}%             {ci_str:<18} {status_str}")

    print("-" * 115)
    print("\nHEAD QUALIFICATION DECISION BY SCOPE:")
    for t_name in ["LEXICAL (Regex)", "REFERENTIAL (Coref)", "LOCALE (Toponym)"]:
        evals = [e for e in all_evaluations if e['teacher'] == t_name]
        ships = [e['domain_scope'] for e in evals if e['meets_75pct_threshold']]
        abstains = [e['domain_scope'] for e in evals if not e['meets_75pct_threshold']]
        print(f"\n  • {t_name}:")
        if ships:
            print(f"      ✅ SHIPS FOR: {', '.join(ships)}")
        if abstains:
            print(f"      ❌ ABSTAINS FOR: {', '.join(abstains)}")

    output_payload = {
        "phase": "1-scoped",
        "kill_criterion_threshold": 75.0,
        "teacher_evaluations": all_evaluations,
    }
    
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved Scoped Teacher Quality Table results to {_OUT_JSON}")

if __name__ == "__main__":
    main()
