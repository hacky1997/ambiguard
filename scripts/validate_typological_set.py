#!/usr/bin/env python3
"""
Validate the typological ambiguity dataset.

Asserts:
- Total rows ~210
- Category balance (no category > 12% of total)
- Control ratio in [0.15, 0.25]
- No duplicate questions
- All required schema fields present per AGENTS.md rule 6
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_DATASET = Path("eval/datasets/typological_ambiguity.jsonl")

REQUIRED_FIELDS = {
    "id",
    "question",
    "context",
    "expected_behaviour",
    "category",
    "ambiguity_present_in",
    "ambiguity_absent_in",
    "resolves_with",
    "annotation_provenance",
    "source",
    "source_id",
    "note",
}


def validate() -> int:
    if not _DATASET.exists():
        print(f"❌ Error: Dataset file not found: {_DATASET}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in open(_DATASET, encoding="utf-8")]
    n_total = len(rows)

    if n_total < 150 or n_total > 300:
        print(f"❌ Expected ~210 rows, found {n_total}", file=sys.stderr)
        return 1

    # Check required fields
    for idx, r in enumerate(rows):
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            print(f"❌ Row {idx} ({r.get('id')}) missing fields: {missing}", file=sys.stderr)
            return 1

    # Check duplicate questions
    questions = [r["question"] for r in rows]
    if len(questions) != len(set(questions)):
        dups = [q for q, count in Counter(questions).items() if count > 1]
        print(f"❌ Found duplicate questions: {dups}", file=sys.stderr)
        return 1

    # Check category balance
    cats = Counter(r["category"] for r in rows)
    max_cat_pct = max(cats.values()) / n_total
    if max_cat_pct > 0.12:
        print(f"❌ Category balance check failed: max category is {max_cat_pct:.1%} > 12%", file=sys.stderr)
        return 1

    # Check control ratio (ANSWER expected_behaviour)
    n_controls = sum(1 for r in rows if r["expected_behaviour"] == "ANSWER")
    ctrl_ratio = n_controls / n_total
    if not (0.15 <= ctrl_ratio <= 0.25):
        print(f"❌ Control ratio check failed: {ctrl_ratio:.1%} not in [15%, 25%]", file=sys.stderr)
        return 1

    print(f"✅ Validation passed for {_DATASET}:")
    print(f"   Total rows: {n_total}")
    print(f"   Categories: {len(cats)} (max share {max_cat_pct:.1%})")
    print(f"   Controls:   {n_controls}/{n_total} ({ctrl_ratio:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
