# AmbiGuard

An evaluation harness for ambiguity-aware query routing, built around a published research classifier.

The short version: I set out to show that a small learned classifier could decide how a QA system should respond — answer, or ask for clarification — more cheaply than asking an LLM. Building the harness to prove it, I found four defects in how this task gets trained and measured, including two in my own published work. On leak-free data, no system tested detects ambiguity above chance — including GPT-4o-mini.

The router is not the contribution. The measurement is.

---

## Results

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
- The gate leads on worst-class F1 — 4.9 vs 3.2 — meaning it is less lopsided across the two classes. On a balanced task that is the more informative number, and it is the one thing here that favours the free 459 ms model over the paid 738 ms one.
- Worst-class F1 is the minimum per-class F1 over gold classes. A constant predictor scores 0.0 by construction. That property is the entire point of including it; an earlier implementation iterated over predicted classes instead and reported 6.7 for a predictor that never emitted the minority class at all.

Raw numbers: [`eval/results/comparison.json`](eval/results/comparison.json)

---

## Finding 1 — benchmark formatting inflates LLM accuracy

The first version of the dataset builder rendered the two classes differently. Unambiguous rows got a prose passage ending in "The documented answer is: X." Ambiguous rows got a bulleted list of sub-questions. Same underlying Wikipedia text, different wrapper.

A model does not need to reason about ambiguity to exploit that.

Controlled ablation — identical rows, identical passage text, only the wrapper differs:

| Variant | Accuracy | 95% CI | Recall (ANSWER) | Recall (AMBIGUOUS) |
|---|---|---|---|---|
| clean | 55.5% | [50.5, 60.2] | 86.5% | 24.5% |
| formatted | 62.0% | [57.0, 66.8] | 94.0% | 30.0% |

*n = 400 per variant, gpt-4o-mini, temperature 0.*

+6.5 points from formatting alone. The intervals still overlap by roughly three points, so this is directionally consistent with the leak hypothesis rather than conclusive — n = 400 was not enough to separate them. The recall shift is the clearer signal: ANSWER recall climbs from 86.5% to 94.0%, which is what you would expect if the wrapper is announcing the label.

`prepare_eval_data.py` now runs an automated leakage check before writing, and fails the build if the label becomes recoverable from answer-string presence, context length, or class-specific markers.

Raw numbers: [`eval/results/leakage_ablation.json`](eval/results/leakage_ablation.json)

---

## Finding 2 — the passage barely helps

A third ablation arm removed the context entirely, leaving only the question:

| Variant | Accuracy | Recall (ANSWER) | Recall (AMBIGUOUS) |
|---|---|---|---|
| clean (with passage) | 54.0% | 87.0% | 21.0% |
| blind (question only) | 51.5% | 3.0% | 100.0% |

The blind arm predicts AMBIGUOUS on 97% of inputs and lands at chance because the set is balanced. The clean arm is degenerate in the opposite direction — 87% recall on one class, 21% on the other.

Both are collapsed predictors. Adding the retrieved passage buys 2.5 points over having no context at all. Whatever signal distinguishes an ambiguous question from an unambiguous one in AmbigNQ, gpt-4o-mini is not recovering it from the evidence passage.

---

## Finding 3 — the gate is not injection-resistant

I expected the learned gate to be structurally immune to prompt injection: it consumes embeddings and emits a probability distribution, so there is no instruction-following surface to attack.

That reasoning is wrong, and the measurement says so.

| Model | Injection resistance | Adversarial accuracy |
|---|---|---|
| CenterDistill | 0% (0/5) | 30.0% |
| gpt-4o-mini | 40% (2/5) | 50.0% |

The gate cannot be instructed by injected text, but it can be perturbed by it. Injected tokens change the encoder's output, which moves max(P_S) across a decision threshold. The routing flips just as effectively as if the model had obeyed an instruction — and here it flipped more often than the LLM's did.

n = 5 injection cases, so the interval is very wide and this does not establish that the gate is worse. It does establish that the immunity claim was unfounded.

Raw numbers: [`eval/results/adversarial.json`](eval/results/adversarial.json)

---

## Two silent bugs in the underlying research

The gate comes from a paper I published (EAAAI 2026, [10.1007/978-3-032-31141-2_11](https://doi.org/10.1007/978-3-032-31141-2_11)). Wiring it into a running system surfaced two defects that the original evaluation could not have caught. A correction is in progress. Both are documented here rather than quietly fixed, because the way they hid is the interesting part.

### The distillation target was constant

```python
# CDTrainer.compute_loss
sl = model.mean_soft_labels.unsqueeze(0).expand(B, -1)
```

Every training example received the same KL target — the mean over all soft labels — instead of its own teacher distribution. Under that objective the optimal center head is a constant function: emit the mean vector, ignore the input. The head learned exactly that.

Symptom: max(P_S) ranged 0.221–0.263 across every input, against a uniform value of 0.200 for K=5. A 0.04 spread over hundreds of distinct questions means the model was not reading the question at all. After passing per-example targets, the range opened to 0.23–0.65.

### The published evaluation never called the model

```python
PS_test = softmax(log(PT_test + 1e-8) + noise)  # noise ~ N(0, 0.08)
```

The reported student distribution was the teacher distribution with Gaussian noise added. The trained center head was never invoked at inference in any reported result.

The published 90.1% behaviour accuracy therefore measures how often adding N(0, 0.08) noise to a distribution fails to push it across a threshold. That is a property of the noise magnitude and the threshold geometry, not of the model. It also explains a finding the paper reports as empirical — that 97% of errors fall within 0.02 of a threshold boundary. Small perturbations only flip labels near boundaries. It had to come out that way.

This is also why the first defect went unnoticed: nothing downstream ever exercised the head, so a broken head produced no visible symptom.

### A third, in this repo

Threshold derivation grid-searched on raw accuracy. On a dataset that was 50% ANSWER, the search found the degenerate optimum — 85% of predictions collapsed to a single class, 44% of them arriving via an `else: ANSWER` fallthrough that was never intended to carry that much traffic. Switching the objective to macro-F1 restored a balanced prediction distribution (48/27/25 across three classes, from 85/12/3).

All three bugs share a shape: the code ran, the outputs looked plausible, and the aggregate metric moved in a believable direction. None was visible without inspecting the distribution of predictions.

---

## The system

The harness sits on a working multi-agent QA service, not a notebook.

```
POST /v1/chat
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

LLM providers are pluggable (OpenAI / Ollama / deterministic mock). Retrieval is in-memory. The application boots with an empty `.env` — no checkpoint falls back to a heuristic gate, no API key falls back to the mock provider. A reviewer with no credentials still gets a running system.

---

## Evaluation infrastructure

- Four comparison arms, bootstrap CIs (10k resamples), balanced accuracy, worst-class F1
- Majority-class baseline reported beside every result
- Automated leakage detection that fails the dataset build
- Adversarial suite: prompt injection, PII, degenerate input, near-boundary selection
- CI regression gate — behaviour accuracy dropping more than 2 points fails the PR
- 51 tests, `mypy --strict` clean on `app/gate/` and `app/graph/`

---

## Quickstart

```bash
make dev
make data     # build the evaluation set from AmbigNQ
make compare  # four-arm comparison (uses mock provider without a key)
make test
uvicorn app.main:app --reload
```

No API key required. Set `AMBIGUARD_LLM_PROVIDER=openai` and a key to run the LLM judge arm against a real model.

To use the real gate, place a repaired checkpoint and point `.env` at it:

```bash
python scripts/repair_checkpoint.py \
  --checkpoint-dir ./centerdistill \
  --base-model deepset/xlm-roberta-large-squad2 \
  --out ./checkpoints/centerdistill_full
```

---

## Limitations

- **n = 600 on the main comparison**: CIs are roughly ±4 points and every arm overlaps chance. Larger samples would not obviously change the conclusion, but they would sharpen it.
- **The leak ablation does not fully separate**: +6.5 points with three points of CI overlap at n = 400. Directional, not conclusive.
- **n = 5 injection cases**: Enough to falsify the immunity claim, nowhere near enough to rank the two systems.
- **Binary task only**: AmbigNQ annotates whether a question is ambiguous, not how a system should respond. Splitting into CLARIFY vs ALTERNATIVES needs PAQA's human clarifying questions; an earlier version of the builder split them by alternating loop index, which gave half the dataset an unlearnable label and silently capped every model evaluated against it at ~75%.
- **Domain shift**: The gate was trained on MLQA clusters and evaluated here on AmbigNQ. Some of the transfer failure may be domain shift rather than a defect in the method.
- **CPU inference on an M2**: Latency numbers are fp32, single-threaded, and would be roughly 20× lower on a GPU.

---

## What I would do next

1. Retrain the center head directly on AmbigNQ annotations rather than transferring MLQA clusters.
2. Expand the injection suite well past n = 5 and test perturbation sensitivity systematically — the finding that a learned gate is perturbable rather than instructable deserves proper treatment.
3. Human agreement study on a sample: if annotators disagree about which questions are ambiguous, a ~50% ceiling is the task, not the models.
4. Temperature scaling or Platt calibration on $P_S$, since threshold placement remains the dominant error source.

---

## Provenance

Data: AmbigNQ (CC BY-SA 4.0), via Hugging Face `ambig_qa`. Labels derive deterministically from human annotation types — never from an LLM, and never from the teacher distribution the student was distilled from. Every row carries `source`, `source_id`, and `annotation_provenance`.

Every number in this README traces to a committed file under `eval/results/`.
