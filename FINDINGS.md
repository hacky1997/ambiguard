# AmbiGuard — Key Empirical Findings

## Key Findings

1. **Answer-Diversity Null Result**:
   - Rescoring answer diversity with transitive linkage methods (`single_link`, `complete_link`, `average_link`) across 5 and 10 samples revealed that high AUCs ($\ge 0.80$) at distance thresholds $\ge 0.40$ are tie-breaking artifacts driven by $>88\%$ of rows collapsing to $k=1$ (mean $k$ gap between classes = 0.002 at 10 samples: ambiguous 1.084 vs unambiguous 1.082).
   - At honest thresholds ($0.20-0.35$), performance degrades with more samples (AUC 0.591–0.647 at 10 samples). Sampled answers to ambiguous questions cluster as tightly as unambiguous ones because models commit to a single interpretation.

2. **Typological Benchmark Holdout & Rule Generalization**:
   - On the 210-row typological ambiguity benchmark across 14 categories, a surface regex baseline achieves 72.4% overall accuracy.
   - 2:1 Dev/Holdout evaluation (140 dev / 70 holdout rows) shows no performance collapse (70.0% dev vs 77.1% holdout, $-7.1\%$ drop), proving that structural rules generalize across unseen lexical variations within categories.
   - Ambiguity is detectable when structurally or lexically marked in text (e.g., entity collisions, date formats, subject drops), but non-surface categories (`word_order`, `calendar`, `honorifics`) leave a 27.6% error gap.
   - **Limitation**: Typological dataset labels are single-author and unvalidated by independent human annotators.

3. **Center Manifold Dispersion & Geometry**:
   - Manifold dispersion (`dispersion (rank-norm)`) achieves an AUC of **0.571** (`95% CI [0.527, 0.618]`), yielding a directional $+0.047$ gain over entropy baseline (`0.524`). On paired bootstrap difference analysis, the paired $\Delta$ CI spans zero ($[-0.009, +0.108]$), indicating directional gain without statistical separation at $n=600$.
   - The learned center directions occupy near-orthogonal positions (pairwise distances 0.9032–1.0132, relative spread 0.113), meaning the model's centers lack manifold distance structure to exploit.

4. **Policy Threshold Limitations**:
   - `second_mass` yields an AUC of **0.425** ($95\%\text{ CI } [0.380, 0.473]$, paired $\Delta$ CI $[-0.163, -0.031]$), which is statistically below chance.
   - The ALTERNATIVES branch fires on 37–48% of traffic using `second_mass`, a statistic measured at AUC 0.425 — reliably anti-correlated with ambiguity. Whether that routing is wrong cannot be determined with binary labels; testing it requires PAQA sub-type annotations.

5. **Data Saturation & Formatting Leakage**:
   - 48% of open-domain QA rows produce near-uniform probabilities ($\max P < 0.30$), setting a hard ceiling on scalar threshold policies over arbitrary trivia queries.
   - Leakage ablation confirmed that surface formatting markers account for a $+6.5\%$ artifact shift in raw accuracy, while repaired context normalisation improves model behavior consistency without altering binary classification boundaries.
