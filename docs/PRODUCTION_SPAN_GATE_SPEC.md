# AmbiGuard — Production Span & Category-Level Ambiguity Gate Specification

> **Specification Proposal: Token-Level Span Detection & Multi-Label Category Tagging for Production Ambiguity**
> 
> *Evolving from sequence-level CLS distribution prediction to token-level span detection and multi-label ambiguity category tagging.*

---

## 1. Executive Summary & Rationale

Sequence-level ambiguity classification on broad open-domain QA benchmarks (such as AmbigNQ) produces chance-level performance across both distilled encoders (CenterDistill ~50.5%) and frontier LLM judges (gpt-4o-mini ~53.3%). Broad semantic ambiguity in arbitrary trivia questions is noisy and unconstrained.

In production environments (enterprise search, support automation, financial/legal QA), ambiguity is overwhelmingly **lexical, surface-marked, and structural**.

By upgrading the router architecture from **pooled CLS classification** to a **token-level sequence tagging head**, the gate:
1. Predicts **which specific span(s)** triggered the ambiguity.
2. Assigns **multi-label ambiguity categories** to each span.
3. Emits an explicit **abstain / clear signal** when no span triggers ambiguity.
4. Provides structured span metadata to downstream Clarification Agents to close the dialogue loop with targeted questions.

---

## 2. Ambiguity Taxonomy (~25 Production Categories)

Production ambiguity spans three distinct domains:

### A. Locale & Formatting Bottlenecks
1. `DATE_FORMAT` — Bare numeric dates (e.g. `"05/06"`, `"10-11-12"`: DD/MM vs MM/DD).
2. `CURRENCY_LOCALE` — Unqualified currency symbols (e.g. `"$"`, `"£"`, `"kr"`: USD vs AUD vs CAD).
3. `DECIMAL_SEPARATOR` — Period vs comma in numeric expressions (e.g. `"1.000"`).
4. `NUMERIC_SCALE` — Billion vs Milliard, short vs long scale.
5. `MEASUREMENT_UNITS` — Imperial vs metric defaults (e.g. `"gallons"`, `"degrees"`).
6. `NAME_ORDER` — Given/family name ordering ambiguity across cultures.
7. `ADDRESS_FORMAT` — Postal code / administrative division structure.

### B. Enterprise & Domain-Specific Bottlenecks
8. `TEMPORAL_DEIXIS` — Time-relative references (`"last quarter"`, `"next Friday"`, `"recently"`, `"YTD"`).
9. `SCOPE_UNSPECIFIED` — Under-specified entity references (`"the report"`, `"the account"`, `"our policy"`).
10. `VERSION_COLLISION` — Software/document version ambiguity (`"how to configure SSO"` across v2 vs v3).
11. `JURISDICTION` — Tax, legal, or compliance rules varying by state/country (`"sales tax rate"`).
12. `AGGREGATION_TYPE` — Financial/metric definitions (`"revenue"`: gross vs net vs recognized).
13. `QUANTIFIER_METRIC` — User/usage counts (`"active users"`: DAU vs MAU vs registered).
14. `PERMISSION_SCOPE` — Answers dependent on caller authorization level.
15. `COMPOUND_QUERY` — Multi-part questions in a single turn.
16. `ANAPHORA_UNBOUND` — Pronouns without thread context (`"it"`, `"they"`).

### C. Cross-Lingual & Structural Bottlenecks
17. `FORMALITY_TV` — Second-person pronoun collapses (e.g. English *"you"* → Spanish *tú/usted*).
18. `SUBJECT_DROP` — Zero-anaphora in pro-drop languages (Japanese, Spanish, Italian).
19. `GENDER_AGREEMENT` — Neutral terms collapsing gender-marked target equivalents.
20. `NUMBER_PLURALITY` — Dual/collective forms lacking singular/plural specification.
21. `HONORIFIC_LEVEL` — Register divergence across polite vs casual target forms.
22. `SCRIPT_VARIANT` — Simplified vs Traditional, Cyrillic vs Latin transliteration.
23. `CALENDAR_SYSTEM` — Gregorian vs Hijri vs Solar Hijri vs Bikram Sambat dates.
24. `CODE_SWITCHING` — Mixed language/script boundary collisions.

---

## 3. Model Architecture & Output Schema

### 3.1 Neural Token-Level Head

```
Input Tokens: [CLS] How do I calculate sales tax for last quarter ? [SEP]
                    │                  │         └───┬────┘
                    │                  │        Span 2: TEMPORAL_DEIXIS (tokens [6, 7])
                    │                  └────────────────Span 1: JURISDICTION (token 4)
                    ▼
          ┌───────────────────────────────────┐
          │   Encoder (XLM-RoBERTa / DeBERTa) │  (hidden_size = 1024)
          └─────────────────┬─────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  Token Classification Head       Global Abstain Head
  Linear(1024 → N_categories + 1) Linear(1024 → 1)
  (Per-token logits)              (Global clear probability)
```

### 3.2 Output Data Contract (`SpanGateDecision`)

```python
from typing import Literal, TypedDict
import numpy.typing as npt
import numpy as np

AmbiguityCategory = Literal[
    "DATE_FORMAT", "CURRENCY_LOCALE", "DECIMAL_SEPARATOR", "NUMERIC_SCALE",
    "MEASUREMENT_UNITS", "NAME_ORDER", "ADDRESS_FORMAT", "TEMPORAL_DEIXIS",
    "SCOPE_UNSPECIFIED", "VERSION_COLLISION", "JURISDICTION", "AGGREGATION_TYPE",
    "QUANTIFIER_METRIC", "PERMISSION_SCOPE", "COMPOUND_QUERY", "ANAPHORA_UNBOUND",
    "FORMALITY_TV", "SUBJECT_DROP", "GENDER_AGREEMENT", "NUMBER_PLURALITY",
    "HONORIFIC_LEVEL", "SCRIPT_VARIANT", "CALENDAR_SYSTEM", "CODE_SWITCHING"
]

class AmbiguousSpan(TypedDict):
    token_start: int
    token_end: int
    text: str
    categories: list[AmbiguityCategory]
    confidence: float

class SpanGateDecision(TypedDict):
    behavior: Literal["ANSWER", "CLARIFY", "ALTERNATIVES"]
    abstain: bool
    spans: list[AmbiguousSpan]
    global_clear_prob: float
    latency_ms: float
    fallback_used: bool
```

---

## 4. Integration with Graph & Clarification Agent

### 4.1 Closed-Loop Clarification

When `SpanGateDecision["behavior"] == "CLARIFY"`:

Instead of emitting a generic prompt (*"Could you clarify your question?"*), the **Clarification Agent** receives structured span context:

```json
{
  "span_text": "last quarter",
  "category": "TEMPORAL_DEIXIS",
  "detected_options": ["Q1 2026 (Calendar)", "Q4 2025 (Fiscal)"]
}
```

The agent formulates a **targeted, ≤25-word question**:
> *"Are you asking about Fiscal Q4 2025 or Calendar Q1 2026?"*

### 4.2 Verification & Abstaining

If no token span exceeds confidence threshold $\tau_{\text{span}} = 0.50$, the global abstain head flags `abstain = True` and routes directly to `ANSWER` without invoking dialogue clarification routines.

---

## 5. Implementation Roadmap

1. **Synthetic Synthetic/Distant Supervision Dataset Generation**: Annotate 2,000 production turns across the 24 categories using regex patterns, entity linkers, and synthetic templates.
2. **Token Head Adapter**: Add `TokenSpanHead` to `app/gate/centerdistill.py`.
3. **Clarification Agent Update**: Update `app/graph/clarification_agent.py` to ingest `AmbiguousSpan` metadata.
