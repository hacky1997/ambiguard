# The Journey, and the Design It Produced

> One document: where this started, every turn it took, why each turn happened, and the
> architecture the evidence actually supports. Written so the reasoning is auditable —
> including the parts where the reasoning was wrong.

---

# PART I — THE JOURNEY

## Where it started

Productionize a published research classifier — a distilled XLM-R gate that routes
questions to ANSWER / CLARIFY / ALTERNATIVES — into a multi-agent QA system, as a
portfolio artifact.

That was the whole scope. It was achievable in roughly a day.

## Turn 1 — the artifact did not work

Wiring the gate into a running system exposed two defects that the original evaluation
could not have surfaced.

**The distillation target was constant.** Every training example received the same KL
target — the mean over all soft labels — instead of its own teacher distribution. The
optimal head under that objective is a constant function, and that is what it learned:
`max(P_S)` spanned 0.221–0.263 across every input, against 0.200 uniform for K=5.

**The evaluation never called the model.** The student distribution used at inference was
constructed as `softmax(log(P_T) + noise)` — the teacher plus Gaussian noise. The trained
head was never invoked. This is why the first defect went unnoticed: nothing downstream
exercised the head, so a degenerate head produced no symptom.

Retraining with per-example targets opened the range to 0.23–0.65. The gate began working.

*What this cost: the headline number the project was built around.*

## Turn 2 — the benchmark did not work either

With a functioning gate, evaluation moved to AmbigNQ human annotations. Nothing beat
chance. Three independent stress tests, each removing a suspected artifact:

| Removed | LLM result |
|---|---|
| class-dependent context formatting | 70.8% → 53.3% |
| a CLARIFY/ALTERNATIVES split assigned by loop index | still ~chance |
| serialised Python repr rebuilt as prose | 53.3% → 53.3% |

The third is decisive. Context length halved, the question was removed from 52–58% of
rows where it had been duplicated inside its own evidence, every serialisation marker
stripped — and accuracy did not move. **A model reading the passage would respond to the
passage becoming readable.**

Supporting evidence: a blind arm with no context at all scored 51.5% against 54.0% with
context. The evidence contributes 2.5 points.

## Turn 3 — ten attempts to find signal

Each was a different mechanism. All returned null.

| Attempt | Outcome |
|---|---|
| confidence-gated cascade | corr(confidence, correct) = +0.005 |
| class-gated cascade | 84% cheaper, −17.5 points accuracy |
| binary reframing | judge advantage *widened* +22 → +39.5 |
| ensemble with the LLM | gate uniquely correct on 3.0% |
| temperature calibration | best T = 1.09, i.e. no miscalibration |
| multilingual centroid rebuild | diagnostics showed flips too large; not run |
| translation displacement as detector | AUC 0.569 [0.506, 0.632] |
| cross-lingual disagreement | AUC 0.407, below chance |
| answer-diversity teacher | AUC 0.49–0.55 after tie correction |
| log-probability judge (CLAM method) | AUC 0.578 [0.515, 0.641] |

## Turn 4 — the measurement apparatus was also wrong

Seven bugs, found by inspecting distributions rather than trusting aggregates:

1. non-transitive clustering mislabelled as single-link, order-dependent
2. `_auc` without tie averaging — inflated a 0.52 statistic to **0.879**
3. comparing a new lower CI bound to a baseline *point* estimate, not its upper bound
4. `'ALTERNATIVES' == 'AMBIGUOUS'` string comparison producing a meaningless 3-class metric
5. selecting the "best" statistic by AUC without a nonzero-rate floor — a 0.7% hedge rate scored 0.972
6. threshold grid search on raw accuracy, finding the majority-class collapse
7. worst-class F1 iterating over predicted rather than gold classes — scored a constant predictor 6.7

Bug 2 is the one worth carrying forward as a finding in its own right: **tie-heavy
discrete statistics silently inflate AUC**, and ambiguity statistics are frequently
discrete.

## Turn 5 — where detection does work

A hand-built typological set — ambiguity created or erased by translation — produced the
first non-null contrast:

**gpt-4o-mini: 87.1% on the typological set, 53.3% on AmbigNQ.** Same model, same binary
framing, 34-point gap.

Per-category, the gate splits cleanly:

| Works (≥60%) | Fails (≤33%) |
|---|---|
| date_format 80.0% | entity_collision 33.3% |
| number_ambiguity 80.0% | gender 26.7% |
| formality 73.3% | numeric_scale 26.7% |
| measurement 66.7% | script_variant 13.3% |
| calendar 60.0% | honorific 13.3% |
| | word_order 6.7% |
| | currency 6.7% |
| | code_switching 6.7% |
| | subject_drop 0.0% |

And hand-written regex beats the pooled gate in 6 of 8 comparable categories — 100% vs 0%
on subject_drop, 86.7% vs 6.7% on currency.

## Turn 6 — two dead ends, honestly closed

**Cross-lingual stability.** Routing decisions survive translation only 66.8% of the time
across six languages, against 83.6% for the LLM. But the LLM predicts one class on 81% of
English rows — a near-constant predictor is stable by construction. Drift is directional:
78% of flips go ANSWER → ALTERNATIVES.

**ASPI attack-surface reduction.** The hypothesis — a router that clarifies less reduces
injection exposure — could not be tested. ASPI's premise requires a model with ~2%
baseline injection susceptibility. Every model available sat at 35–85%, and enabling
their prompt guard made it *worse* (85% → 100% ASR, utility 55% → 15%). Untestable, not
falsified.

## The three causes, separated

Twelve nulls, three distinct reasons, and conflating them would be the last mistake:

**Cause A — the task was undetectable.** AmbigNQ's label records what an annotator found
while searching Wikipedia. That is a fact about a search process, not about the query.
Six attempts hit this wall.

**Cause B — the instrument was faulty.** Seven measurement bugs. Some nulls were partly
artefacts of the harness, not properties of the data.

**Cause C — external constraints.** Frontier model access, PAQA annotations, human
annotators. Walls, not failures.

## The drift, named

The project was finished after Turn 1. Everything after was searching for a positive
result, and the search kept aiming at the benchmark where detection is impossible — even
after Turn 5 showed where it is possible.

**The signal that should have redirected it:** the 34-point gap between the same model on
text-marked versus annotation-derived ambiguity. That appeared at Turn 5 and six more
experiments were aimed at AmbigNQ afterward.

---

# PART II — WHAT THE EVIDENCE ACTUALLY SUPPORTS

## The diagnosis nobody stated until now

The gate is not weak. **Its output space is wrong.**

It has 5 slots. Those slots came from spectral clustering of 500 English questions by
*topic similarity*. Ambiguity has at least 14 mechanically distinct kinds. The head is
being asked to express "this is a currency ambiguity" as a distribution over five topic
clusters that were never about ambiguity.

Three consequences follow, and all three are visible in the measurements:

- **Pooling destroys the evidence.** Triggers are token-local: a bare `$` at position 7,
  an unqualified `05/06`. CLS pooling averages them away. Regex 100% vs pooled gate 0% on
  subject_drop is the signature.
- **One head cannot serve incompatible mechanisms.** Lexical markers are visible in the
  surface form. Referential gaps require dependency structure. Locale ambiguity requires
  metadata the text does not contain. No shared representation covers all three.
- **One teacher cannot supervise all of them.** Spectral clustering knows about topic. It
  does not know about dates, pronouns, or locales.

## The design: heterogeneous multi-head distillation

Each head learns from **the teacher that actually knows its subproblem**.

```
                          query tokens  (+ locale metadata when present)
                                 │
                          XLM-R encoder
                                 │
        ┌──────────────┬─────────┴────────┬──────────────┐
        ▼              ▼                  ▼              ▼
   LEXICAL        REFERENTIAL          LOCALE         SCOPE
   token spans    attention over       metadata +     in-domain /
                  sequence             text          abstain
        │              │                  │              │
   teacher:       teacher:            teacher:       teacher:
   regex          coref / parser      locale rules   agreement of
   (100% on       (spaCy, stanza)     + explicit     the three
   dates, etc.)                       markers        heads
```

**Multi-label, not multi-class.** "How much did I spend last quarter in $" fires LEXICAL
(currency) and REFERENTIAL (scope of "spend") simultaneously. The existing 5-way softmax
cannot express that; independent binary heads can.

**The scope head is the contribution.** It predicts whether *any* head is competent. When
none is, the system emits ABSTAIN and escalates. Every published ambiguity detector
reports accuracy on what it covers; none reports what fraction it cannot judge.

## The question this design must answer first

**If regex is the teacher and regex already scores 100%, why is a student better than
just running the regex?**

There are exactly three defensible answers, and each must be *measured*, not asserted:

1. **Generalisation to unseen surface forms** — regex matches patterns; a student should
   generalise to forms the pattern list does not contain.
2. **Multilingual transfer** — regex rules are per-language; XLM-R is not. A student
   trained on English lexical markers should transfer to Spanish and Hindi.
3. **Composition** — regex fires per-rule; a student can weigh multiple weak signals.

**If none of the three holds, the honest answer is to ship the regex.** That is Phase 0,
and it comes before any training.

---

# PART III — EDGE CASES, ENUMERATED IN ADVANCE

Written now, not discovered late. Each has a stated detection method and mitigation.

### E1 — Student cannot exceed a rule-based teacher on the training distribution

**Risk:** highest in the document. Distilling regex into a network reproduces regex.
**Detect:** hold out *surface forms*, not rows. Train on `DD/MM`, test on `DD.MM.YYYY`,
`YYYY年MM月`, `Reiwa 3`. Train on `$`, test on `kr`, `₨`, `R$`.
**Mitigate:** if the student does not beat the teacher on held-out forms with a paired CI
excluding zero, **stop and ship the regex.**
**This is Phase 0 and it can end the project in a day.**

### E2 — Head disagreement with no arbitration rule

**Risk:** LEXICAL says ambiguous, REFERENTIAL says clear. Undefined behaviour.
**Mitigate:** disagreement is not an error, it is a signal. Any head firing above
threshold → AMBIGUOUS, with the firing head's category attached. Disagreement *rate* is a
reported metric and feeds the scope head.

### E3 — Teacher quality is not uniform

**Risk:** regex is ~100% on dates; a coref model on short queries is closer to 60–70%.
Distilling a 65% teacher gives a ≤65% student, and averaging heads hides it.
**Detect:** measure every teacher against the gold set *before* distillation. Publish the
table.
**Mitigate:** any head whose teacher scores below ~75% does not ship — it routes to
ABSTAIN instead. A weak head is worse than no head.

### E4 — Locale metadata absent at inference

**Risk:** the LOCALE head is trained with locale tags; most benchmarks strip them and
production sometimes lacks them.
**Mitigate:** train with metadata dropout (30% of examples masked). At inference, absent
metadata is itself informative — an unlocalised query is *more* ambiguous. Report LOCALE
head performance with and without metadata separately.

### E5 — Per-head class imbalance

**Risk:** each head sees ~10% positives for its own category. A head that always predicts
negative scores 90%.
**Detect:** per-head balanced accuracy and worst-class F1, never raw accuracy. The
worst-class F1 implementation must iterate over **gold** classes — this was bug 7.
**Mitigate:** class-weighted loss per head; report the trivial-baseline score beside every
number.

### E6 — Template leakage across splits

**Risk:** slot-filled data means the same template appears in train and test with
different fillers. Row-level splitting leaks the pattern and produces a fake result.
**Mitigate:** split by **slot value and template ID jointly**. Assert in the validator
that no template ID crosses the split boundary. Fail the build otherwise.

### E7 — Synthetic data teaches the generator, not the phenomenon

**Risk:** the model learns the construction rules rather than ambiguity.
**Detect:** a second evaluation on *naturally occurring* queries mined by regex recall
from public multilingual corpora (MASSIVE, MTOP, Mintaka), with spans verified by
inspection rather than generated.
**Mitigate:** if performance on natural data collapses relative to synthetic, report both
numbers and state the gap. Do not report only the synthetic figure.

### E8 — Negative controls insufficient

**Risk:** a detector that fires on every date scores well on a set where every date is
ambiguous.
**Mitigate:** ≥25% controls per category — ISO dates, currencies with ISO codes, formality
fixed by context, quantities with explicit units. Control accuracy is reported separately
and **below 50% voids the headline number.**

### E9 — Heads are not independent, but are trained as if they are

**Risk:** a shared encoder means gradient from one head degrades another. Classic
multi-task interference.
**Detect:** train each head alone; compare to joint training. If joint is worse, the
interference is real.
**Mitigate:** gradient scaling per head, or freeze the encoder and train heads
independently. Report both configurations.

### E10 — The research asset disappears

**Risk:** discarding the 5-way center head removes the published model from the system —
the thing this project exists to showcase.
**Mitigate:** keep it as a fifth head with its own measured contribution. If it adds
nothing above the other four in an ablation, that is a legitimate finding and belongs in
the write-up rather than being hidden.

### E11 — Compute

**Risk:** XLM-R-large + 4 heads on an M2 is slow; joint training may not fit.
**Mitigate:** start from XLM-R-**base**. If the base model reaches similar accuracy, that
is a better result — smaller, faster, cheaper. Report both if both run.

### E12 — Threshold tuning on the test set

**Risk:** the failure mode of this entire project. Every threshold in every earlier
experiment was fitted somewhere it should not have been.
**Mitigate:** all thresholds derived on **dev only**, recorded in the results file before
the test run, and never adjusted afterward. The scope selection in §Phase 1 is frozen
against pre-existing numbers.

### E13 — Selection on noise

**Risk:** `calendar` beats the LLM by 27 points at n=15 — two examples. Choosing it as
in-scope because of that repeats the 0.879-AUC error.
**Mitigate:** category inclusion requires ≥60% accuracy **and** n ≥ 100 after expansion.
`calendar` is excluded at this stage and revisited only after expansion.

### E14 — Per-head evaluation looks fine, system evaluation does not

**Risk:** each head is good; arbitration is bad; end-to-end is worse than any component.
**Mitigate:** report head-level *and* system-level metrics. The system number is the
headline. A coverage-accuracy curve shows the operating points rather than one threshold.

---

# PART IV — PHASED PLAN, WITH KILL CRITERIA

| # | Deliverable | Kill criterion |
|---|---|---|
| **0** | Regex teacher measured on held-out **surface forms** | Student later fails to beat this with a paired CI excluding zero → **ship the regex, stop** |
| **1** | Teacher quality table — regex, coref, locale rules, each vs gold | Any teacher < 75% → that head does not ship |
| **2** | Slot-filled data, ~200/category in-scope, ≥25% controls, split by template ID | Validator fails on control ratio or template leakage → fix before proceeding |
| **3** | Heads trained independently, encoder frozen | Any head below its teacher → investigate before joint training |
| **4** | Joint training, gradient scaling | Joint worse than independent → ship independent, report interference |
| **5** | Scope head + abstention calibrated on dev | Precision < 85% at abstention < 40% → report the curve, not a point |
| **6** | Full arm comparison: regex, pooled gate, multi-head, gpt-4o-mini, always-abstain | — |
| **7** | Natural-data evaluation (E7) | Collapse vs synthetic → report both, state the gap |

**Phase 0 first. It can end the project in a day, and that is the point.**

## What a positive looks like, written before the experiment

> On N lexically-marked ambiguity categories, the multi-head gate reaches **X%** balanced
> accuracy against gpt-4o-mini's **Y%**, abstaining on **Z%** of queries, at **$0.00** and
> **W ms** against $4.39/1k and 738 ms. It beats a regex teacher by **D points** on
> held-out surface forms, paired CI excluding zero, and transfers to **L** languages
> without per-language rules.

**Requires:** X within ~10 points of Y; Z < 40%; D > 0 with CI excluding zero; control
accuracy > 70%.

**Every failure mode is reportable:**

- regex matches the student → the contribution is the scoping method and the taxonomy
- abstention > 60% → the scope is too narrow; report the coverage-accuracy curve
- natural data collapses → the model learned the generator; state it
- joint training interferes → ship independent heads; that is a real architectural finding
- the 5-way head adds nothing → the published asset is superseded; say so

None of these requires pretending, and each is a sentence someone could defend in an
interview.

---

## The through-line

The gate was never weak. It had five topic clusters where it needed a dozen mechanisms,
one teacher where it needed four, and a pooled vector where the evidence was token-local.
Every null in Part I is consistent with that, and the design in Part II is what the
evidence supports rather than what would have been convenient.

The measurement discipline that produced twelve nulls is the same discipline that would
make a positive believable. That is the argument for the whole journey, and it is why the
nulls stay in the write-up.
