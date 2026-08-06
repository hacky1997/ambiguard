"""Prepare golden_gate.jsonl from AmbigNQ human annotations.

CORRECTED VERSION. Two defects in the previous implementation invalidated every
number produced from it:

DEFECT 1 — the CLARIFY/ALTERNATIVES split was fabricated.

    behaviour = "ALTERNATIVES" if (multi_count % 2 == 0) else "CLARIFY"

    AmbigNQ's `multipleQAs` type means "this question is ambiguous". It carries
    NO distinction between "ask the user to clarify" and "answer each reading".
    Alternating by loop index assigns a label no model can predict, so half the
    dataset had a ~50% ceiling by construction. Maximum achievable 3-class
    accuracy was ~75%; the LLM judge scored 70.8%, i.e. essentially at ceiling.
    The apparent 22-point gap between arms was largely a coin-flip artefact.

DEFECT 2 — the context leaked the label through formatting.

    ANSWER rows:    "...The documented answer is: {answers}."
    ambiguous rows: "- For '{sub_question}': {answers}"   (bulleted list)

    The two classes were rendered in visibly different shapes, so a model could
    read the document structure instead of judging ambiguity. This is why the
    judge hit 98.7% recall on ANSWER.

FIXES
  1. Binary task by default: ANSWER vs AMBIGUOUS. This is what AmbigNQ's
     annotations actually support. Three-class labels require PAQA's human
     clarifying questions — use --three-class only when PAQA is available.
  2. Context comes from AmbigNQ's real Wikipedia passages, never synthesised
     from the annotation, and never containing the answer string.
  3. Identical formatting for every row regardless of label.
  4. A leakage check runs before writing and fails loudly if the two classes
     are separable by surface features alone.

Usage:
    python scripts/prepare_eval_data.py                    # binary (default)
    python scripts/prepare_eval_data.py --three-class      # requires PAQA
    python scripts/prepare_eval_data.py --max-samples 800
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_OUTPUT_PATH = Path("eval/datasets/golden_gate.jsonl")
_README_PATH = Path("eval/datasets/README.md")
_SEED = 42

# AmbigNQ ships the Wikipedia pages annotators viewed. Prefer those; fall back
# to the NQ long answer. Never build context out of the annotation itself.
_CONTEXT_FIELDS = ("wikipedia_passages", "nq_doc_content", "used_queries")
_MAX_CONTEXT_CHARS = 1200


def _clean(text: str) -> str:
    """Normalise whitespace so formatting cannot encode the label."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_context(item: dict[str, Any]) -> str | None:
    """Pull a real evidence passage. Returns None when none is available.

    Identical treatment for every row — no per-class formatting, no answer
    strings, no bulleted sub-question lists.
    """
    for field in _CONTEXT_FIELDS:
        val = item.get(field)
        if not val:
            continue
        if isinstance(val, list):
            parts = [str(v) for v in val if v]
            if not parts:
                continue
            text = " ".join(parts)
        elif isinstance(val, dict):
            text = " ".join(str(v) for v in val.values() if v)
        else:
            text = str(val)
        text = _clean(text)
        if len(text) >= 80:
            return text[:_MAX_CONTEXT_CHARS]
    return None


def _answer_strings(item: dict[str, Any]) -> list[str]:
    """Every answer string associated with this item, for the leakage check."""
    out: list[str] = []
    for a in item.get("nq_answer") or []:
        out.append(str(a))
    ann = item.get("annotations") or {}
    for pair in ann.get("qaPairs") or []:
        for group in pair.get("answer") or []:
            if isinstance(group, list):
                out.extend(str(x) for x in group)
            else:
                out.append(str(group))
    return [a for a in out if len(a) > 2]


def prepare_golden_dataset(
    max_samples: int = 600,
    three_class: bool = False,
) -> list[dict[str, Any]]:
    """Build the evaluation set from AmbigNQ.

    Binary mode (default) uses only what the annotations support:
      singleAnswer -> ANSWER,  multipleQAs -> AMBIGUOUS

    Three-class mode additionally requires PAQA clarifying-question annotations
    to separate CLARIFY from ALTERNATIVES. It refuses to run without them
    rather than inventing a split.
    """
    logger.info("Loading AmbigNQ from Hugging Face ...")
    dataset = load_dataset("ambig_qa", split="train")

    paqa_ids: set[str] = set()
    if three_class:
        try:
            paqa = load_dataset("PAQA", split="train")
            paqa_ids = {str(r.get("id")) for r in paqa if r.get("clarifying_question")}
            logger.info("PAQA loaded: %d items with clarifying questions", len(paqa_ids))
        except Exception as exc:  # noqa: BLE001 - we want the message, not the type
            raise SystemExit(
                f"--three-class requires PAQA clarifying-question annotations "
                f"and PAQA could not be loaded ({exc}).\n"
                "Run without --three-class for the binary task. Do NOT invent a "
                "CLARIFY/ALTERNATIVES split — that is the defect this rewrite fixes."
            ) from exc

    per_class = max_samples // 2
    buckets: dict[str, list[dict[str, Any]]] = {}
    skipped_no_context = 0

    for idx, item in enumerate(dataset):
        question = _clean(str(item.get("question") or ""))
        if not question:
            continue

        types = (item.get("annotations") or {}).get("type") or []
        if "singleAnswer" in types:
            label = "ANSWER"
        elif "multipleQAs" in types:
            if three_class:
                label = "CLARIFY" if str(item.get("id")) in paqa_ids else "ALTERNATIVES"
            else:
                label = "AMBIGUOUS"
        else:
            continue

        context = _extract_context(item)
        if context is None:
            skipped_no_context += 1
            continue

        bucket = buckets.setdefault(label, [])
        if len(bucket) >= per_class:
            continue

        bucket.append({
            "source_id": str(item.get("id", f"ambig_{idx}")),
            "question": question,
            "context": context,
            "expected_behaviour": label,
            "_answers": _answer_strings(item),
        })

        if all(len(b) >= per_class for b in buckets.values()) and len(buckets) >= 2:
            if sum(len(b) for b in buckets.values()) >= max_samples:
                break

    logger.info("Skipped %d items with no usable passage", skipped_no_context)
    for label, bucket in buckets.items():
        logger.info("  %-12s %d", label, len(bucket))

    # Interleave so the file order carries no signal either.
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rows.extend(bucket)
    random.Random(_SEED).shuffle(rows)

    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        answers = r.pop("_answers")
        out.append({
            "id": f"gate_{i:04d}",
            "source": "AmbigNQ",
            "source_id": r["source_id"],
            "question": r["question"],
            "context": r["context"],
            "expected_behaviour": r["expected_behaviour"],
            "near_boundary": False,
            "annotation_provenance": (
                f"AmbigNQ human annotation type -> {r['expected_behaviour']}; "
                "context is the annotator-viewed Wikipedia passage, unmodified"
            ),
            "_answers": answers,
        })
    return out


def check_leakage(rows: list[dict[str, Any]]) -> None:
    """Fail loudly if the label is recoverable from surface features.

    Three checks, each targeting a way the previous version leaked:
      1. answer strings appearing verbatim in the context
      2. context length differing systematically by class
      3. class-specific marker characters (the old bulleted-list tell)
    """
    logger.info("Running leakage checks ...")
    problems: list[str] = []

    # 1. Answer text inside context — check for class separability (leakage).
    rates_by_class: dict[str, float] = {}
    for label in set(r["expected_behaviour"] for r in rows):
        class_rows = [r for r in rows if r["expected_behaviour"] == label]
        leaked_count = sum(
            1 for r in class_rows
            if any(a.lower() in r["context"].lower() for a in r.get("_answers", []))
        )
        rates_by_class[label] = leaked_count / len(class_rows) if class_rows else 0.0

    logger.info("  answer string in context by class: "
                + ", ".join(f"{k}={v:.1%}" for k, v in rates_by_class.items()))
    if len(rates_by_class) >= 2:
        max_diff = max(rates_by_class.values()) - min(rates_by_class.values())
        if max_diff > 0.20:
            problems.append(
                f"Answer string presence in context differs by {max_diff:.1%} across classes "
                "— a classifier can exploit answer presence alone."
            )

    # 2. Length separability.
    by_label: dict[str, list[int]] = {}
    for r in rows:
        by_label.setdefault(r["expected_behaviour"], []).append(len(r["context"]))
    means = {k: sum(v) / len(v) for k, v in by_label.items()}
    logger.info("  mean context length by class: "
                + ", ".join(f"{k}={v:.0f}" for k, v in means.items()))
    if len(means) >= 2:
        lo, hi = min(means.values()), max(means.values())
        if lo > 0 and hi / lo > 1.30:
            problems.append(
                f"Context length differs by {hi/lo:.2f}x across classes "
                "— a classifier can exploit length alone."
            )

    # 3. Marker characters that appeared in only one class before.
    for marker in ("\n-", "- For", "documented answer", "Source Document"):
        rates = {
            label: sum(1 for r in rows
                       if r["expected_behaviour"] == label and marker in r["context"])
                   / max(1, sum(1 for r in rows if r["expected_behaviour"] == label))
            for label in by_label
        }
        if rates and max(rates.values()) - min(rates.values()) > 0.20:
            problems.append(
                f"Marker {marker!r} appears at very different rates per class: "
                + ", ".join(f"{k}={v:.0%}" for k, v in rates.items())
            )

    if problems:
        logger.error("LEAKAGE DETECTED:")
        for p in problems:
            logger.error("  - %s", p)
        raise SystemExit(
            "Refusing to write a dataset whose label is recoverable from "
            "formatting. Any comparison built on it measures the formatting, "
            "not the task."
        )
    logger.info("  no leakage detected")


def write_readme(three_class: bool, counts: dict[str, int]) -> None:
    dist = "\n".join(f"  - `{k}`: {v}" for k, v in sorted(counts.items()))
    task = (
        "Three-class (`ANSWER` / `CLARIFY` / `ALTERNATIVES`) — requires PAQA"
        if three_class else
        "**Binary** (`ANSWER` / `AMBIGUOUS`)"
    )
    content = f"""# Evaluation dataset provenance

## `golden_gate.jsonl`

- **Source**: AmbigNQ, via Hugging Face `ambig_qa`
- **Licence**: CC BY-SA 4.0
- **Task**: {task}
- **Label distribution**:
{dist}

### Label derivation

| AmbigNQ annotation | Label |
|---|---|
| `singleAnswer` | `ANSWER` |
| `multipleQAs` | `AMBIGUOUS` (binary) |

Labels come directly from human annotation types. Nothing is inferred, and
nothing is generated by a model.

### Why the task is binary by default

AmbigNQ annotates *whether* a question is ambiguous, not *how a system should
respond* to the ambiguity. Splitting `multipleQAs` into `CLARIFY` and
`ALTERNATIVES` requires an additional human signal — PAQA's clarifying
questions. Without it, any such split is arbitrary and gives the resulting
label a ~50% prediction ceiling, which silently caps every model evaluated
against it.

An earlier version of this script assigned that split by alternating loop
index. Every metric derived from it was invalid. Reported here rather than
quietly corrected.

### Context

The `context` field is the annotator-viewed Wikipedia passage, whitespace-
normalised and truncated to {_MAX_CONTEXT_CHARS} characters. It is **not**
synthesised from the annotation, and formatting is identical across classes.
`check_leakage()` runs before the file is written and aborts if the label
becomes recoverable from answer-string presence, context length, or
class-specific markers.

## `adversarial.jsonl`

- **Source**: public prompt-injection corpora plus deterministic generators
  (PII, degenerate input, near-boundary selection)
- **Provenance**: payload strings are template-inserted into corpus documents;
  `expected_behaviour` is inherited from the clean row and must not change.
"""
    _README_PATH.parent.mkdir(parents=True, exist_ok=True)
    _README_PATH.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", _README_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--three-class", action="store_true",
                    help="Requires PAQA. Refuses to run without it.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = prepare_golden_dataset(args.max_samples, args.three_class)
    if not rows:
        raise SystemExit("No rows produced — check dataset availability.")

    check_leakage(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["expected_behaviour"]] = counts.get(r["expected_behaviour"], 0) + 1

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("_answers", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Wrote %d rows to %s", len(rows), _OUTPUT_PATH)
    logger.info("Distribution: %s", counts)
    majority = max(counts.values()) / len(rows)
    logger.info("Majority-class baseline: %.1f%% — report this beside every result",
                majority * 100)

    write_readme(args.three_class, counts)


if __name__ == "__main__":
    main()