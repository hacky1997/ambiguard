#!/usr/bin/env python3
"""
Repair the context field in golden_gate.jsonl.

THE PROBLEM
-----------
The `context` field is not prose. It is a serialised Python repr:

    "['Who dies in fast and furious tokyo drift?'] [{'title': ['Han Lue', ...],
      'snippet': [...]}]"

Three defects follow:

  1. The QUESTION IS REPEATED inside its own context. Both the gate and the LLM
     judge see the question twice, once as the query and once embedded in the
     evidence.
  2. The evidence is wrapped in Python syntax — brackets, quotes, dict keys. Neither
     model is reading a passage; both are reading a repr.
  3. `annotation_provenance` claims "annotator-viewed Wikipedia passage, unmodified".
     That string is false in every row.

This is NOT a label leak — structural markers are balanced across classes
(mean length 1197 vs 1198, identical bracket and title counts). It is malformed
input, which is a different failure: it degrades both arms equally rather than
favouring one.

WHY IT MATTERS
--------------
Every arm scored near chance on this dataset. Part of that may be the input, not
the task. Rebuilding the context as prose and re-running the comparison is the only
way to separate "ambiguity detection is hard" from "the evidence was unreadable".

WHAT THIS DOES
--------------
  * Parses the repr with ast.literal_eval (never eval).
  * Extracts snippet text only; drops the repeated question and the title list.
  * Rebuilds context as plain prose with identical formatting for every row.
  * Re-runs the leakage checks — including the answer-presence DIFFERENTIAL across
    classes, which is the check that actually detects synthesis.
  * Writes a repaired file and a diff report. Does not overwrite the original
    unless --in-place is passed.

USAGE
    python scripts/repair_context.py --dry-run       # inspect a few rows
    python scripts/repair_context.py                 # writes *_repaired.jsonl
    python scripts/repair_context.py --in-place      # replaces golden_gate.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import statistics as st
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GOLDEN = Path("eval/datasets/golden_gate.jsonl")
_REPAIRED = Path("eval/datasets/golden_gate_repaired.jsonl")
_REPORT = Path("eval/results/context_repair.json")
_MAX_CHARS = 1200


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_context(raw: str) -> tuple[list[str], list[str], list[str]]:
    """Return (queries, titles, snippets) from the serialised blob.

    Structure is str(list_of_queries) + " " + str(list_of_dicts). The split point
    is the boundary between the two top-level bracketed structures.

    ast.literal_eval fails because the snippet text contains HTML entities
    (&#39;, <b>, &amp;, &nbsp;) and escaped newlines (\\n).  We use balanced-
    bracket matching instead, then strip HTML and normalise whitespace.
    """
    import html as _html

    queries: list[str] = []
    titles: list[str] = []
    snippets: list[str] = []

    # Find the boundary: "] [" followed by a dict or list opener.
    m = re.search(r"\]\s*\[\s*[\{\[]", raw)
    if m:
        left, right = raw[: m.start() + 1], raw[m.start() + 1:].strip()
    else:
        left, right = "", raw

    if left:
        try:
            val = ast.literal_eval(left)
            if isinstance(val, list):
                queries = [str(v) for v in val if isinstance(v, str)]
        except (ValueError, SyntaxError):
            pass

    def _extract_quoted_strings(blob: str) -> list[str]:
        """Walk through a [...] body and extract top-level single-quoted strings.

        Handles escaped quotes (\\') and backslash escapes inside strings.
        Strips HTML tags and entities from the extracted text.
        """
        results: list[str] = []
        in_quote = False
        current: list[str] = []
        i = 0
        while i < len(blob):
            c = blob[i]
            if not in_quote:
                if c == "'":
                    in_quote = True
                    current = []
                    i += 1
                    continue
                i += 1
                continue
            # Inside a quoted string
            if c == "\\" and i + 1 < len(blob):
                nc = blob[i + 1]
                if nc == "n":
                    current.append(" ")
                elif nc == "'":
                    current.append("'")
                else:
                    current.append(nc)
                i += 2
                continue
            if c == "'":
                in_quote = False
                s = "".join(current)
                # Strip HTML tags, decode entities, normalise whitespace
                s = re.sub(r"<[^>]+>", "", s)
                s = _html.unescape(s)
                s = re.sub(r"\s+", " ", s).strip()
                if s and s != "...":
                    results.append(s)
                i += 1
                continue
            current.append(c)
            i += 1
        return results

    def _find_bracket_body(text: str, key: str) -> str | None:
        """Find 'key': [...] and return the bracket body."""
        pattern = re.compile(re.escape(f"'{key}'") + r"\s*:\s*\[")
        match = pattern.search(text)
        if not match:
            return None
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1
        return text[start : i - 1]

    # Extract titles
    title_body = _find_bracket_body(right, "title")
    if title_body is not None:
        titles = _extract_quoted_strings(title_body)

    # Extract snippets
    snippet_body = _find_bracket_body(right, "snippet")
    if snippet_body is not None:
        snippets = _extract_quoted_strings(snippet_body)

    return queries, titles, snippets


def rebuild(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Rebuild the context as prose. Returns (context, diagnostics)."""
    raw = row.get("context") or ""
    queries, titles, snippets = parse_context(raw)

    # Drop the repeated question. Compare loosely: punctuation and case vary.
    q_norm = re.sub(r"[^a-z0-9 ]", "", row["question"].lower()).strip()

    def is_the_question(s: str) -> bool:
        n = re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        return n == q_norm or (len(n) > 15 and (n in q_norm or q_norm in n))

    kept = [_clean(s) for s in snippets if s and not is_the_question(s)]

    # Deduplicate, preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for s in kept:
        if s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)

    prose = " ".join(uniq)
    if not prose:
        # No snippets survived — fall back to titles so the row is not empty,
        # and flag it so these rows can be dropped or inspected.
        prose = " ".join(_clean(t) for t in titles if t)

    prose = prose[:_MAX_CHARS]

    return prose, {
        "n_queries": len(queries),
        "n_titles": len(titles),
        "n_snippets": len(snippets),
        "n_kept": len(uniq),
        "question_removed": len(snippets) - len(kept),
        "empty": not prose,
        "raw_len": len(raw),
        "new_len": len(prose),
    }


def leakage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Differential checks — a rate that is HIGH in both classes is fine.

    What matters is whether the rate DIFFERS by class. That is the check that
    detects context built from the annotation.
    """
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["expected_behaviour"], []).append(r)

    out: dict[str, Any] = {"per_class": {}, "problems": []}
    lengths: dict[str, float] = {}
    q_in_ctx: dict[str, float] = {}

    for label, group in by_label.items():
        lens = [len(r["context"]) for r in group]
        qic = [
            1.0 if re.sub(r"[^a-z0-9 ]", "", r["question"].lower()).strip()
                    in re.sub(r"[^a-z0-9 ]", "", r["context"].lower())
            else 0.0
            for r in group
        ]
        lengths[label] = st.mean(lens)
        q_in_ctx[label] = st.mean(qic)
        out["per_class"][label] = {
            "n": len(group),
            "mean_len": round(st.mean(lens), 1),
            "question_in_context_rate": round(st.mean(qic), 4),
            "empty_rate": round(
                st.mean([1.0 if not r["context"].strip() else 0.0 for r in group]), 4),
        }

    if len(lengths) >= 2:
        lo, hi = min(lengths.values()), max(lengths.values())
        if lo > 0 and hi / lo > 1.30:
            out["problems"].append(
                f"context length differs {hi/lo:.2f}x across classes")
    if len(q_in_ctx) >= 2:
        spread = max(q_in_ctx.values()) - min(q_in_ctx.values())
        if spread > 0.20:
            out["problems"].append(
                f"question-in-context rate differs by {spread:.0%} across classes")

    for marker in ("'title'", "'snippet'", "[{", "}]"):
        rates = {
            label: st.mean([1.0 if marker in r["context"] else 0.0 for r in group])
            for label, group in by_label.items()
        }
        if max(rates.values()) > 0.05:
            out["problems"].append(
                f"serialisation marker {marker!r} still present "
                + ", ".join(f"{k}={v:.0%}" for k, v in rates.items()))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--drop-empty", action="store_true",
                    help="Drop rows whose context does not survive parsing.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = [json.loads(line) for line in open(_GOLDEN, encoding="utf-8")]
    logger.info("Loaded %d rows", len(rows))

    before = leakage_report(rows)
    print("\n" + "=" * 72)
    print("BEFORE")
    print("=" * 72)
    for label, v in before["per_class"].items():
        print(f"  {label:<12} n={v['n']:<5} len={v['mean_len']:<8} "
              f"question-in-context={v['question_in_context_rate']:.0%}")
    for p in before["problems"]:
        print(f"  PROBLEM: {p}")

    repaired: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    for r in rows:
        prose, d = rebuild(r)
        diags.append(d)
        nr = dict(r)
        nr["context"] = prose
        nr["annotation_provenance"] = (
            f"AmbigNQ human annotation type -> {r['expected_behaviour']}; "
            "context rebuilt as prose from the evidence snippets "
            "(question and serialisation removed)"
        )
        repaired.append(nr)

    if args.dry_run:
        print("\n" + "=" * 72)
        print("SAMPLE (first 3)")
        print("=" * 72)
        for r, nr, d in list(zip(rows, repaired, diags))[:3]:
            print(f"\n[{r['expected_behaviour']}] {r['question']}")
            print(f"  BEFORE ({d['raw_len']} chars): {r['context'][:200]}")
            print(f"  AFTER  ({d['new_len']} chars): {nr['context'][:200]}")
            print(f"  snippets={d['n_snippets']} kept={d['n_kept']} "
                  f"question_removed={d['question_removed']}")
        print("\nDry run — nothing written.")
        return 0

    empties = [i for i, d in enumerate(diags) if d["empty"]]
    if empties:
        logger.warning("%d rows produced empty context", len(empties))
        if args.drop_empty:
            keep = [i for i in range(len(repaired)) if i not in set(empties)]
            repaired = [repaired[i] for i in keep]
            logger.info("Dropped them; %d rows remain", len(repaired))

    after = leakage_report(repaired)
    print("\n" + "=" * 72)
    print("AFTER")
    print("=" * 72)
    for label, v in after["per_class"].items():
        print(f"  {label:<12} n={v['n']:<5} len={v['mean_len']:<8} "
              f"question-in-context={v['question_in_context_rate']:.0%}  "
              f"empty={v['empty_rate']:.0%}")
    if after["problems"]:
        for p in after["problems"]:
            print(f"  PROBLEM: {p}")
    else:
        print("  no problems detected")

    mean_before = st.mean(d["raw_len"] for d in diags)
    mean_after = st.mean(d["new_len"] for d in diags)
    removed = sum(1 for d in diags if d["question_removed"] > 0)
    print("-" * 72)
    print(f"  mean context length {mean_before:.0f} -> {mean_after:.0f} chars")
    print(f"  question removed from {removed}/{len(diags)} rows")

    out = _GOLDEN if args.in_place else _REPAIRED
    with open(out, "w", encoding="utf-8") as f:
        for r in repaired:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {len(repaired)} rows -> {out}")

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPORT, "w") as f:
        json.dump({
            "before": before, "after": after,
            "mean_len_before": round(mean_before, 1),
            "mean_len_after": round(mean_after, 1),
            "rows_with_question_removed": removed,
            "empty_after_repair": len(empties),
            "output": str(out),
        }, f, indent=2)

    print("\n  NEXT: re-run `make compare`. If the LLM judge moves materially above")
    print("  53%, the earlier 'nothing beats chance' result was partly an artefact")
    print("  of malformed input — which is itself a finding worth reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
