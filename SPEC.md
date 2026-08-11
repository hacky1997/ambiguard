# AmbiGuard — Specification

> **ACTIVE PHASE: 1**
> Build only what Phase 1 permits. See `AGENTS.md` rule 17.

---

## 1. What this is

A multi-agent question-answering system whose routing decision is made by a published research model rather
than by a prompt. Before any LLM token is spent, a small classifier (**CenterDistill**, EAAAI 2026,
DOI `10.1007/978-3-032-31141-2_11`, upstream: `github.com/hacky1997/Centerdistill`) decides whether the
question should be **answered**, **clarified**, or **split into alternatives**. Agents then execute that
decision with real tools, memory, and an adversarial verifier that can reject their work.

**The headline is the measurement, not the architecture.** The repo's central artifact is a comparison
proving the learned gate beats LLM self-assessment on accuracy, latency, cost, and manipulability — plus the
harness that keeps proving it on every commit.

---

## 2. CenterDistill — how the gate works

Do not replace this with a prompt. Understanding why is the point of the project.

1. **LaBSE** embeds questions (768-dim, L2-normalised).
2. **Spectral clustering** (cosine affinity, K=5, seed=42) induces semantic centers over a 500-question pool.
   No human ambiguity labels are used.
3. Teacher distribution: `P_T(c_k|q) = softmax(τ · µ̃_kᵀ ê_q)`, `τ = 10.0`.
4. Student: **XLM-RoBERTa-large** (`deepset/xlm-roberta-large-squad2`) with a span head and a **center head**
   `Linear(hidden → 5)`. Loss `L = λ·KL(P_T‖P_S) + (1−λ)·L_span`, `λ = 0.70`.
5. Inference thresholds: `τ_conf = 0.44`, `τ_ent = 1.51` **nats**, `τ_multi = 0.24`, applied in the order
   given in `AGENTS.md` rule 2.
6. All six hyperparameters are derived programmatically from training statistics, not grid-searched.

**Published results** (en→es, N=1000, seed 42): 90.1% behaviour accuracy, 8.8 WC-F1, 77.3 QA-F1,
95% bootstrap CI `[88.2%, 91.8%]` — non-overlapping with the confidence-threshold baseline `[78.2%, 84.4%]`.
en→de transfer: 91.0%.

**Limitations — reproduce these unedited in the README:**
- Behaviour labels come from teacher-induced distributions, not independent human annotation.
- 97% of misclassifications sit within 0.02 of a threshold boundary; fixed thresholds are the dominant error
  source.
- Silhouette scores 0.03–0.04: clusters are semantically coherent, not geometrically separated.
- Only two high-resource language pairs evaluated.
- Backbone ~560M params vs ~340M for published MLQA baselines.

**Consequence:** the gate is deterministic, auditable, and costs one linear layer on the CLS token.
An LLM asked "is this ambiguous?" is none of those. Phase 1 exists to prove that quantitatively.

---

## PHASE 1 — Comparison harness

No agent code. No API. No graph. The argument comes before the system.

### 1.1 In scope

```
pyproject.toml, Makefile, .env.example
app/settings.py
app/gate/{base,centerdistill,heuristic,thresholds}.py
app/llm/{base,mock_provider,openai_provider,registry}.py
scripts/{fetch_checkpoint,prepare_eval_data,prepare_adversarial_data}.py
eval/metrics/{behaviour_accuracy,worst_cluster_f1,bootstrap}.py
eval/arms/{centerdistill,llm_judge,majority,confidence}_arm.py
eval/{run_comparison,run_adversarial,report}.py
tests/{test_gate_contract,test_entropy_units}.py
```

### 1.2 Out of scope — do not create, not even stubs

`app/graph/**`, `app/api/**`, `app/retrieval/**`, `app/guardrails/**`, `ui/**`, `Dockerfile`,
`docker-compose.yml`, `README.md`.

### 1.3 Gate adapter

Checkpoint sources, in priority order: local path from settings → HF Hub repo from settings → **absent, use
heuristic fallback** with `fallback_used=True`. Never crash on a missing checkpoint.

Implementation: load once at startup; tokenize with `max_length=384`, `truncation="only_second"`; CLS hidden
→ center head → softmax → `P_S`; compute `max_prob`, `entropy` (nats), `second_mass`; apply thresholds in
order; return every intermediate value. Use `torch.inference_mode()` and `model.eval()`.

Heuristic fallback: no ML dependencies, same protocol, emits a synthetic 5-vector so downstream code needs
no special-casing. Simple lexical rules are fine — it exists to keep the repo runnable, not to be accurate.

`GateDecision` carries: `behaviour`, `center_distribution`, `max_prob`, `entropy`, `second_mass`,
`thresholds` (echoed), `latency_ms`, `fallback_used`.

### 1.4 Comparison arms

| Arm | What it does |
|---|---|
| `centerdistill` | The real gate |
| `llm_judge` | An LLM asked directly to classify ambiguity, few-shot, temperature 0 |
| `majority` | Always predicts the majority class — **required**, the dataset is imbalanced |
| `confidence` | Span-softmax max over a threshold; the baseline the paper beats |

Measured per arm: behaviour accuracy, worst-cluster F1, bootstrap CI (10k resamples), p50/p95 latency,
cost per 1k decisions, and determinism (run 3×, check byte-identical).

Report intervals, never bare point estimates.

### 1.5 Evaluation data — `golden_gate.jsonl`

Built by `scripts/prepare_eval_data.py` from public human-annotated sources. **Not authored, not generated.**

| Source | Selection | → Behaviour |
|---|---|---|
| AmbigNQ | one interpretation | `ANSWER` |
| AmbigNQ | multiple distinct interpretations | `ALTERNATIVES` |
| PAQA (AmbigNQ + human clarifying questions) | has a gold clarifying question | `CLARIFY` |
| SituatedQA *(optional)* | temporal/geographic under-specification | tag as near-boundary |

Verify each dataset's current location and licence before use; record both in `eval/datasets/README.md`.
Do not hardcode a dataset path you have not confirmed exists.

Each row carries: `id`, `source`, `source_id`, `question`, `context`, `expected_behaviour`,
`near_boundary`, `annotation_provenance`.

Sizing: aim for several hundred rows minimum. **Do not force a uniform behaviour split** — report the true
distribution and compute the majority baseline from it.

Context: use each dataset's provided evidence passages where available. Where absent, retrieve with the same
LaBSE encoder the gate's centers were induced from — embedding-space consistency between retrieval and
gating is deliberate.

`DECISION:` commit the derivation script and a small sample; `make data` builds the full set locally.

### 1.6 Adversarial data — `adversarial.jsonl`

Built by `scripts/prepare_adversarial_data.py`. Derived, never authored, never LLM-generated.

| Category | Source |
|---|---|
| Indirect prompt injection | Public injection benchmarks (InjecAgent, AgentDojo, LLMail-Inject, and HF prompt-injection datasets) — verify availability before use |
| Near-boundary | Derived: run the gate over `golden_gate.jsonl`, select rows scoring within 0.02 of any threshold |
| PII | Public PII-masking datasets or Faker-generated |
| Degenerate input | Generated: empty string, very long input, punctuation-only, unicode edge cases |
| Non-English | Existing rows — the backbone is cross-lingual |
| Contradictory context | Derived: pair a question with two passages asserting incompatible facts |

**Injection construction:** take payload strings from the benchmark, template-insert into corpus documents at
varying positions, pair with a question whose gold behaviour is already known. `expected_behaviour` is
inherited from the clean row and **must not change** — any deviation is a hard failure, not a scored loss.

State plainly in `EVALUATION.md` that payloads are drawn from public corpora and inserted programmatically,
not authored against this deployment.

### 1.7 The row that matters

Run injection resistance against **both** the gate arm and the `llm_judge` arm. The gate consumes embeddings
and emits a distribution — it has no instruction-following surface to attack. The LLM judge reads poisoned
context as text and can be instructed by it. That contrast is an architectural claim, and it is the strongest
line in the repo.

**Sanity check:** if `llm_judge` scores 100% resistance, the payloads are not reaching the model. That is a
broken harness, not a secure one. Investigate before reporting.

### 1.8 Exit criteria

- [ ] Both prep scripts run end-to-end; every row has `source`, `source_id`, `annotation_provenance`
- [ ] True behaviour distribution documented; majority baseline computed from it
- [ ] `test_entropy_units.py` passes, asserting nats
- [ ] `test_gate_contract.py` passes for both gate implementations
- [ ] `make compare` and `make adversarial` write real results files
- [ ] Injection resistance reported for both arms side by side

When all boxes are checked, change **ACTIVE PHASE** to 2.

---

## PHASE 2 — Agents (locked until Phase 1 exits)

Five components, each justified by a failure mode observed in Phase 1 or in the naive single-path build.
Each must survive the question *"what does this do that a conditional couldn't?"*

- **Supervisor** — the gate. A learned policy, not a prompt router. Owns delegation and arbitration when the
  verifier rejects.
- **Research agent** — *failure mode: one retrieval pass cannot serve two interpretations.* Own toolbelt,
  multi-hop loop, decides when it has enough evidence. Returns typed `Evidence`, never raw strings.
- **Clarification agent** — *failure mode: the naive version re-asks what it already asked.* Owns dialogue
  memory. Generates exactly one question, ≤25 words, naming a concrete competing interpretation from context.
  Generic clarifications are a scored failure. Then interrupts the graph.
- **Verification agent** — *failure mode: generation cites sources that don't support the claim.* Checks
  claims against evidence, flags ungrounded spans, can reject and send work back. Bounded at 2 retries, then
  downgrades to CLARIFY. Instrument how often it fires.
- **Synthesis agent** — merges interpretations, resolves conflicts, formats citations.

Handoffs are typed Pydantic contracts, never free-text. Each agent is a LangGraph subgraph with its own
internal state; the shared `AgentState` carries only what crosses boundaries.

`AgentState` fields: `thread_id`, `messages` (reducer), `question`, `resolved_question`, `evidence`, `gate`,
`answer`, `clarifying_question`, `alternatives`, `verification`, `citations`, `trace_id`, `error`.

Resume path: on user reply, set `resolved_question`, re-enter at retrieval. The gate re-runs on the enriched
question — the clarified version should now route to ANSWER, and that transition is itself a metric.

---

## PHASE 3 — Ship (locked)

**CI:** `ruff` → `mypy` (strict on `app/gate`, `app/graph`) → `pytest`, provider forced to mock, no secrets,
must pass on a fork. Separate workflow runs the gate eval on PRs touching the gate or datasets and **fails
the PR if behaviour accuracy drops more than 2 points**. Break the gate deliberately once, let CI fail,
screenshot it for the README.

**Metrics** (Prometheus): gate latency histogram, behaviour counters, `verification_rejection_rate`,
`clarify_resolution_rate` (fraction of CLARIFY threads reaching a confident answer on resume — a product
metric the paper does not have; make it a headline number), `fallback_gate_used_total`.

**README order:** hook → comparison table → demo asset → the insight in three sentences → architecture →
quickstart (*no API key required*) → adversarial results → how the research maps to the system (DOI + upstream
link) → limitations, unedited → what's next (threshold calibration, human eval, distilled sub-100M gate).

Tone: precise, no hype, numbers with CIs, limitations stated plainly.

---

## Open questions — resolve before the gate adapter is finished

- Checkpoint format: full fine-tuned XLM-R-large or LoRA adapters? LoRA needs `peft` plus a separate base
  model pull; `fetch_checkpoint.py` must branch.
- Checkpoint location: local disk or HF Hub? Hub availability enables a hosted demo with real weights.
- `llm_judge` model: `DECISION:` a current frontier model at temperature 0, few-shot. If no key is available,
  run against a local model and label the table accordingly. **The arm must exist either way** — without it
  there is no argument.
