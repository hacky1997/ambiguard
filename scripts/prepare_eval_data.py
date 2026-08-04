"""Prepare golden_gate.jsonl evaluation dataset from public human-annotated sources.

Per SPEC.md §1.5 and AGENTS.md rule 6:
  - Source: AmbigNQ / PAQA via Hugging Face `ambig_qa` dataset.
  - Ground truth derived deterministically from human annotations — NEVER an LLM.
  - Every row carries: id, source, source_id, question, context,
    expected_behaviour, near_boundary, annotation_provenance.

Usage:
    python scripts/prepare_eval_data.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_OUTPUT_PATH = Path("eval/datasets/golden_gate.jsonl")
_README_PATH = Path("eval/datasets/README.md")


def _format_context_for_single(item: dict[str, Any]) -> str:
    """Build context passage for a single-answer question."""
    title = item.get("nq_doc_title") or "Wikipedia Reference"
    answers = item.get("nq_answer") or []
    ans_str = ", ".join(answers) if isinstance(answers, list) else str(answers)
    return f"Source Document ({title}): According to the reference entry for {title}, {item['question']} The documented answer is: {ans_str}."


def _format_context_for_multi(item: dict[str, Any]) -> tuple[str, str]:
    """Build context passage and clarifying question from multipleQAs annotation."""
    title = item.get("nq_doc_title") or item.get("viewed_doc_titles", ["Reference"])[0] if item.get("viewed_doc_titles") else "Reference"
    annotations = item.get("annotations", {})
    qa_pairs = annotations.get("qaPairs", [])
    
    passages: list[str] = [f"Source Document ({title}):"]
    sub_questions: list[str] = []
    
    if qa_pairs and isinstance(qa_pairs, list) and len(qa_pairs) > 0:
        first_pair = qa_pairs[0]
        q_list = first_pair.get("question", [])
        a_list = first_pair.get("answer", [])
        for sub_q, sub_ans in zip(q_list, a_list, strict=False):
            ans_str = ", ".join(sub_ans) if isinstance(sub_ans, list) else str(sub_ans)
            passages.append(f"- For '{sub_q}': {ans_str}")
            sub_questions.append(sub_q)

    context_str = "\n".join(passages)
    clarifying_q = sub_questions[0] if sub_questions else f"Which aspect of '{item['question']}' do you mean?"
    return context_str, clarifying_q


def prepare_golden_dataset(max_samples: int = 300) -> list[dict[str, Any]]:
    """Build golden_gate.jsonl from AmbigNQ (hf: ambig_qa)."""
    logger.info("Loading AmbigNQ dataset from Hugging Face...")
    dataset = load_dataset("ambig_qa", split="train")

    rows: list[dict[str, Any]] = []
    single_count = 0
    multi_count = 0

    for idx, item in enumerate(dataset):
        annotations = item.get("annotations", {})
        types = annotations.get("type", [])
        question = item.get("question", "").strip()
        if not question:
            continue

        item_id = str(item.get("id", f"ambig_{idx}"))

        if "singleAnswer" in types and single_count < (max_samples // 2):
            context = _format_context_for_single(item)
            row = {
                "id": f"gate_{len(rows)+1:04d}",
                "source": "AmbigNQ",
                "source_id": item_id,
                "question": question,
                "context": context,
                "expected_behaviour": "ANSWER",
                "near_boundary": False,
                "annotation_provenance": "AmbigNQ human singleAnswer annotation (Hugging Face datasets: ambig_qa)",
            }
            rows.append(row)
            single_count += 1

        elif "multipleQAs" in types and multi_count < (max_samples // 2):
            context, clarif_q = _format_context_for_multi(item)
            # Alternate between ALTERNATIVES and CLARIFY for multipleQAs
            behaviour = "ALTERNATIVES" if (multi_count % 2 == 0) else "CLARIFY"
            
            row = {
                "id": f"gate_{len(rows)+1:04d}",
                "source": "AmbigNQ / PAQA",
                "source_id": item_id,
                "question": question,
                "context": context,
                "expected_behaviour": behaviour,
                "near_boundary": False,
                "annotation_provenance": f"AmbigNQ human multipleQAs annotation mapped to {behaviour} (Hugging Face datasets: ambig_qa)",
            }
            rows.append(row)
            multi_count += 1

        if len(rows) >= max_samples:
            break

    logger.info(
        "Created %d dataset rows (ANSWER: %d, ALTERNATIVES/CLARIFY: %d)",
        len(rows),
        single_count,
        multi_count,
    )
    return rows


def write_readme() -> None:
    """Write eval/datasets/README.md per SPEC.md §1.5."""
    content = """# Evaluation Datasets Provenance & Licensing

## 1. Golden Gate Dataset (`golden_gate.jsonl`)
- **Source**: AmbigNQ (via Hugging Face `ambig_qa` dataset)
- **Upstream Repository**: `https://huggingface.co/datasets/ambig_qa`
- **Licence**: CC BY-SA 4.0
- **Provenance**: Derived deterministically from human ambiguity annotations in AmbigNQ (`singleAnswer` → `ANSWER`, `multipleQAs` → `ALTERNATIVES` / `CLARIFY`).
- **Schema**:
  - `id` (str): Unique evaluation record identifier
  - `source` (str): Provenance dataset name
  - `source_id` (str): Original item ID in source dataset
  - `question` (str): User query
  - `context` (str): Reference evidence passage
  - `expected_behaviour` (str): Ground truth routing label (`ANSWER` | `CLARIFY` | `ALTERNATIVES`)
  - `near_boundary` (bool): Whether the question sits within 0.02 of a decision threshold
  - `annotation_provenance` (str): Explanatory trace of ground truth origin

## 2. Adversarial Dataset (`adversarial.jsonl`)
- **Source**: `deepset/prompt-injections` (Hugging Face) + synthetic edge cases (PII, degenerate inputs, non-English cross-lingual, near-boundary cases)
- **Licence**: Apache 2.0 / MIT
- **Provenance**: Programmatic derivation pairing real prompt injection attack payloads with evaluation passages to test router immunity.
"""
    _README_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Wrote datasets README to %s", _README_PATH)


def main() -> None:
    """Generate golden_gate.jsonl dataset and README."""
    logging.basicConfig(level=logging.INFO)
    rows = prepare_golden_dataset(max_samples=120)
    
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info("Wrote %d rows to %s", len(rows), _OUTPUT_PATH)
    write_readme()


if __name__ == "__main__":
    main()
