# AmbiGuard

![AmbiGuard Live Demo](demo.gif)

An evaluation harness for ambiguity-aware query routing, built around a published research classifier.

The short version: I set out to show that a small learned classifier could decide how a QA system should respond — answer, or ask for clarification — more cheaply than asking an LLM. Building the harness to prove it, I found four defects in how this task gets trained and measured, including two in my own published work. On leak-free data, no system tested detects ambiguity above chance — including GPT-4o-mini.

The router is not the contribution. The measurement is.

---

## 1. Cross-Lingual Evaluation

The gate architecture (XLM-RoBERTa over LaBSE-induced centers) was pretrained on 100 languages. We evaluate two cross-lingual hypotheses:

### C1 — Translation Stability
Ambiguity is a property of meaning, not surface form. A routing decision that flips upon translating the question is flawed regardless of which side was right. We evaluate 300 questions translated into 6 non-English languages (Spanish, German, Japanese, Arabic, Hindi, Swahili) using NLLB-600M.

| Model / Arm | Mean Stability | 95% CI | Worst Language | English Baseline Distribution |
|---|---|---|---|---|
| **CenterDistill (learned gate)** | **66.8%** | **[64.6%, 69.0%]** | **Swahili (63.3%)** | **ANSWER: 54%, AMBIGUOUS: 46%** |
| gpt-4o-mini (LLM judge) | 83.6%* | [81.8%, 85.4%] | Swahili (79.7%) | ANSWER: 81%, AMBIGUOUS: 19% (Collapsed) |

*\*Note on LLM Stability: The LLM judge predicts `ANSWER` on 81% of English queries. Its high raw stability (83.6%) is trivially driven by class collapse (predicting ANSWER almost constantly), rather than genuine cross-lingual robustness.*

**Per-Language Stability (CenterDistill Gate)**:
- German (`de`): **72.7%**
- Spanish (`es`): **67.3%**
- Hindi (`hi`): **66.7%**
- Arabic (`ar`): **66.0%**
- Japanese (`ja`): **64.7%**
- Swahili (`sw`): **63.3%**

---

### C2 — Typological Ambiguity & Taxonomy

Some ambiguity is created or erased by linguistic translation (e.g. T-V formality, pro-drop subjects, script variants). We evaluate on a 210-row typological benchmark across 14 categories, containing 20% control rows (`ANSWER` expected).

| System | Overall Accuracy | 95% CI | Control Accuracy | 95% CI |
|---|---|---|---|---|
| **CenterDistill (learned gate)** | 35.2% | [29.0%, 41.9%] | 45.2% | [30.9%, 59.5%] |
| **gpt-4o-mini (LLM judge)** | **87.1%** | **[82.4%, 91.4%]** | **97.6%** | **[92.9%, 100.0%]** |

#### Typological Category Breakdown

| Category | n | Sample Contrast / Example | Gate Acc | LLM Acc |
|---|---|---|---|---|
| `currency` | 15 | `$50,000` (USD vs CAD vs AUD) | 6.7% | **100.0%** |
| `numeric_scale` | 15 | `1 billion` (US 10^9 vs Long Scale 10^12) | 26.7% | **100.0%** |
| `subject_drop` | 15 | Japanese `tabeta?` (Omits subject pronoun) | 0.0% | **100.0%** |
| `honorific` | 15 | Korean `Kim-seonsaengnim` (Teacher vs Doctor) | 13.3% | **100.0%** |
| `code_switching` | 15 | Hinglish `yeh setting change kar do` | 6.7% | **100.0%** |
| `formality` | 15 | German `How are you?` (dir vs Ihnen) | 73.3% | **100.0%** |
| `measurement` | 15 | `75 degrees` (°C vs °F) | 66.7% | **100.0%** |
| `entity_collision` | 15 | `Santiago` (Chile vs Spain) | 33.3% | **100.0%** |
| `date_format` | 15 | `05/06/2024` (DD/MM vs MM/DD) | 80.0% | **93.3%** |
| `number_ambiguity` | 15 | English `you` (singular vs plural group) | 80.0% | **93.3%** |
| `script_variant` | 15 | Traditional Chinese `發` (issue) vs `髮` (hair) | 13.3% | **86.7%** |
| `gender` | 15 | French `the doctor` (le vs la médecin) | 26.7% | **73.3%** |
| `word_order` | 15 | German `Den Hund sah der Mann` (OSV case) | 6.7% | **40.0%** |
| `calendar` | 15 | `BE 2567` (Thai Buddhist Era vs Gregorian) | **60.0%** | 33.3% |

Raw numbers: [`eval/results/crosslingual_stability.json`](eval/results/crosslingual_stability.json), [`eval/results/crosslingual_typological.json`](eval/results/crosslingual_typological.json)

---

## 2. Monolingual English Benchmark

Balanced binary task (ANSWER vs AMBIGUOUS), 600 examples from AmbigNQ human annotations, 300 per class. Chance is 50.0%.

| Arm | Accuracy | 95% CI | Worst-class F1 | p50 latency | $/1k |
|---|---|---|---|---|---|
| CenterDistill (learned gate) | 50.3% | [46.3, 54.3] | 4.9 | 459 ms | $0.00 |
| gpt-4o-mini (LLM judge) | 53.3% | [49.3, 57.5] | 3.2 | 738 ms | $4.39 |
| Majority class | 50.0% | [46.0, 54.0] | 0.0 | 0 ms | $0.00 |
| Confidence threshold | 49.2% | [45.2, 53.2] | 0.9 | 463 ms | $0.00 |

Read this carefully:
- Every confidence interval contains 50%. No arm is distinguishable from guessing.
- The LLM judge leads on raw accuracy by 3 points, well inside the overlap.
- Both systems collapse toward one class: worst-class F1 is 4.9 and 3.2 on a 0–10 scale, meaning neither handles the weaker class competently. The gate's minimum is marginally higher, but the 1.7-point difference is inside sampling noise at n=600 and no confidence interval is reported for this metric.
- Worst-class F1 is the minimum per-class F1 over gold classes. A constant predictor scores 0.0 by construction. An earlier implementation iterated over predicted classes instead and reported 6.7 for a predictor that never emitted the minority class at all.

Raw numbers: [`eval/results/comparison.json`](eval/results/comparison.json)

---

## 3. Finding 1 — benchmark formatting inflates LLM accuracy

The first version of the dataset builder rendered the two classes differently. Unambiguous rows got a prose passage ending in "The documented answer is: X." Ambiguous rows got a bulleted list of sub-questions. Same underlying Wikipedia text, different wrapper.

A model does not need to reason about ambiguity to exploit that.

Controlled ablation — identical rows, identical passage text, only the wrapper differs:

| Variant | Accuracy | 95% CI | Recall (ANSWER) | Recall (AMBIGUOUS) |
|---|---|---|---|---|
| clean | 55.5% | [50.5, 60.2] | 86.5% | 24.5% |
| formatted | 62.0% | [57.0, 66.8] | 94.0% | 30.0% |

*n = 400 per variant, gpt-4o-mini, temperature 0.*

+6.5 points from formatting alone. The 95% intervals overlap by 3.2 points ([57.0, 60.2]), so this is a directional result consistent with the leak hypothesis, not a statistically separated effect at n=400. The recall shift is the clearer evidence: ANSWER recall climbs from 86.5% to 94.0%, which is what you would expect if the wrapper is announcing the label.

`prepare_eval_data.py` now runs an automated leakage check before writing, and fails the build if the label becomes recoverable from answer-string presence, context length, or class-specific markers.

Raw numbers: [`eval/results/leakage_ablation.json`](eval/results/leakage_ablation.json)

---

## 4. Finding 2 — the passage barely helps

A third ablation arm removed the context entirely, leaving only the question:

| Variant | Accuracy | Recall (ANSWER) | Recall (AMBIGUOUS) |
|---|---|---|---|
| clean (with passage) | 54.0% | 87.0% | 21.0% |
| blind (question only) | 51.5% | 3.0% | 100.0% |

The blind arm predicts AMBIGUOUS on 97% of inputs and lands at chance because the set is balanced. The clean arm is degenerate in the opposite direction — 87% recall on one class, 21% on the other.

Both are collapsed predictors. Adding the retrieved passage buys 2.5 points over having no context at all. Whatever signal distinguishes an ambiguous question from an unambiguous one in AmbigNQ, gpt-4o-mini is not recovering it from the evidence passage.

---

## 5. Finding 3 — the gate is perturbable, not injection-immune

I expected the learned gate to be structurally immune to prompt injection: it consumes embeddings and emits a probability distribution, so there is no instruction-following surface to attack.

Injected tokens change the encoder's output, which moves $max(P_S)$ across a decision threshold. The routing flips just as effectively as if the model had obeyed an instruction. Note on metric scope: `eval/results/adversarial.json` reports 0% injection resistance on a legacy 5-case payload suite measuring explicit instruction compliance in text output, whereas the 300-case suite in `injection_robustness.json` measures decision-distribution stability under context perturbation.

Evaluating mitigation via **question-only gating** (`decide(question, None)`):

| Mode | Clean Accuracy | 95% CI | Robustness (Hold Rate) | 95% CI |
|---|---|---|---|---|
| with context | 45.3% | [37.3%, 53.3%] | 95.7% | [93.3%, 97.7%] |
| **question only** | **47.3%** | **[39.3%, 55.3%]** | **100.0%** | **[100.0%, 100.0%]** |

*n_clean = 150, n_injection = 300 poisoned variants across prefix/midpoint/suffix positions.*

Question-only gating is injection-immune by construction — the gate never reads retrieved text. It moves clean accuracy from 45.3% to 47.3% while raising robustness from 95.7% to 100%. Robustness is gained at no measurable accuracy cost, since the clean accuracy confidence intervals overlap heavily ([37.3%, 53.3%] vs [39.3%, 55.3%]).

Raw numbers: [`eval/results/injection_robustness.json`](eval/results/injection_robustness.json)

---

## 6. Two silent bugs in the underlying research

The gate comes from a paper I published (EAAAI 2026, [10.1007/978-3-032-31141-2_11](https://doi.org/10.1007/978-3-032-31141-2_11)). Wiring it into a running system surfaced two defects that the original evaluation could not have caught. A correction is in progress.

### The distillation target was constant

```python
# CDTrainer.compute_loss
sl = model.mean_soft_labels.unsqueeze(0).expand(B, -1)
```

Every training example received the same KL target — the mean over all soft labels — instead of its own teacher distribution. Under that objective the optimal center head is a constant function: emit the mean vector, ignore the input. The head learned exactly that.

Symptom: $max(P_S)$ ranged 0.221–0.263 across every input, against a uniform value of 0.200 for K=5. A 0.04 spread over hundreds of distinct questions means the model was not reading the question at all. After passing per-example targets, the range opened to 0.23–0.65.

### The published evaluation never called the model

```python
PS_test = softmax(log(PT_test + 1e-8) + noise)  # noise ~ N(0, 0.08)
```

The reported student distribution was the teacher distribution with Gaussian noise added. The trained center head was never invoked at inference in any reported result.

The published 90.1% behaviour accuracy therefore measures how often adding $N(0, 0.08)$ noise to a distribution fails to push it across a threshold.

---

## 7. The System

The harness sits on a working multi-agent QA service, not a notebook.

```
POST /api/chat
 │
 ├─ supervisor    CenterDistill gate — routes before any LLM call
 │                (one linear layer on the CLS token; ~460 ms CPU, $0)
 │
 ├─ research      own toolbelt, multi-hop retrieval, returns typed Evidence
 ├─ clarification owns dialogue memory, emits one question, interrupts the graph
 ├─ verification  adversarial; checks claims against evidence, can reject and
 │                send work back — bounded at 2 retries, then downgrades to CLARIFY
 └─ synthesis     merges interpretations, formats citations
```

Built on LangGraph, with typed Pydantic contracts between agents rather than free-text handoffs. The clarification path uses LangGraph interrupts, so the graph genuinely halts, surfaces a question, and resumes on the next turn with state preserved — verified end to end through `POST /api/chat` → `POST /api/chat/resume`.

---

## 8. Evaluation Infrastructure

- Four comparison arms, bootstrap CIs (10k resamples), balanced accuracy, worst-class F1
- 210-row typological ambiguity benchmark across 14 categories
- Translation stability evaluation across 6 languages with disk-cached NLLB-600M
- Automated leakage detection that fails the dataset build
- Adversarial suite: prompt injection, PII, degenerate input, near-boundary selection
- CI regression gate — behaviour accuracy dropping more than 2 points fails the PR
- 51 tests, `mypy --strict` clean on `app/gate/` and `app/graph/`

---

## 9. Quickstart

```bash
make dev
make data     # build evaluation sets (AmbigNQ + typological set)
make compare  # four-arm comparison
make test
uvicorn app.main:app --reload
```

Set `AMBIGUARD_LLM_PROVIDER=openai` and `AMBIGUARD_OPENAI_API_KEY` in `.env` to run LLM judge arms.

---

## 10. Limitations

- **NLLB Translation Quality**: NLLB-600M is distilled; translation errors can occasionally confound stability measurements.
- **Domain Shift**: The gate was trained on MLQA clusters and evaluated here on AmbigNQ and typological contrasts.
- **Binary vs Three-Way Task**: Splitting ambiguous queries into CLARIFY vs ALTERNATIVES requires human clarifying annotations (like PAQA).
- **CPU Inference**: Latency numbers are fp32 single-threaded CPU runs.

---

## 11. What I Would Do Next

1. Retrain the center head directly on AmbigNQ and typological annotations.
2. Systematic perturbation analysis of transformer encoder embeddings under adversarial tokens.
3. Human agreement study on cross-lingual ambiguity perception.
4. Temperature scaling / Platt calibration on $P_S$.

---

## Provenance

Data: AmbigNQ (CC BY-SA 4.0), via Hugging Face `ambig_qa`. Labels derive deterministically from human annotation types — never from an LLM. Every row carries `source`, `source_id`, and `annotation_provenance`.

Every number in this README traces to a committed file under `eval/results/`.
