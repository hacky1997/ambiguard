# AmbiGuard — Key Empirical Findings

## Headline Finding

No system tested beats chance on AmbigNQ — a finding that survived the removal of a surface formatting leak (+6.5% artifact shift), a fabricated label split, and malformed context passages.

## The Two-Benchmark Contrast

- **Typological Benchmark** (GPT-4o-mini / Surface Rules): **87.1% / 72.4%** accuracy
- **AmbigNQ Benchmark** (GPT-4o-mini / CenterDistill Gate): **53.3% / 50.5%** accuracy (Chance = 50.0%)

**The 33.8-point gap**: Ambiguity is readily detectable when it lives in the text (explicit lexical, structural, or locale markers), but unlearnable when it lives in external annotation processes.

## Supporting Evidence & Diagnostic Nulls

1. **Answer-Side Semantic Clustering**:
   - Sampled answers to ambiguous questions cluster as tightly as answers to unambiguous ones (mean $k$ gap = 0.002 at 10 samples: ambiguous 1.084 vs unambiguous 1.082).
   - Rescoring with transitive linkage (`single_link`, `complete_link`, `average_link`) confirms that high AUCs at $\text{thr} \ge 0.40$ are tie-breaking artifacts driven by $>88\%$ of rows collapsing to $k=1$. Honest thresholds ($0.20–0.35$) degrade with sample count (AUC 0.591–0.647 at 10 samples). Models commit to a single reading rather than reflecting question-side ambiguity.

2. **Typological Holdout Generalization**:
   - 2:1 Dev/Holdout evaluation (140 dev / 70 holdout rows) shows no performance collapse (70.0% dev vs 77.1% holdout, $-7.1\%$ drop), confirming surface rules generalize across unseen lexical variations within categories.

3. **Center Manifold Dispersion & Geometry**:
   - Manifold dispersion (`dispersion (rank-norm)`) achieves AUC **0.571** (`95% CI [0.527, 0.618]`), yielding a directional $+0.047$ gain over entropy baseline (`0.524`). Paired bootstrap difference CI spans zero ($[-0.009, +0.108]$).
   - Pairwise center distances span 0.9032 to 1.0132 (relative spread 0.113): learned centers occupy near-orthogonal positions without distance manifold structure.

4. **Routing Policy & Threshold Defect**:
   - `second_mass` yields AUC **0.425** ($95\%\text{ CI } [0.380, 0.473]$, paired $\Delta$ CI $[-0.163, -0.031]$), which is statistically below chance.
   - Under derived thresholds, the `ALTERNATIVES` branch unconditionally triggers on 37–48% of traffic using a statistic anti-correlated with ambiguity.

5. **Data Saturation & Formatting Leakage**:
   - 48% of open-domain QA rows produce near-uniform center probabilities ($\max P < 0.30$), setting a hard ceiling on scalar threshold policies over arbitrary trivia queries.
   - Surface formatting markers accounted for a $+6.5\%$ artifact shift in raw accuracy; context repairs improved behavioral consistency without altering binary classification limits.

## Methodological Limitations

- **Typological Benchmark**: Dataset labels are single-author and unvalidated by independent human annotators.
- **AmbigNQ**: Labels reflect teacher-induced distributions and specific annotator interpretations rather than observable model behavior.
