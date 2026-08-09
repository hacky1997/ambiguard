# AmbiGuard — Key Empirical Findings

## Key Findings

1. **Center Manifold Dispersion & Geometry**:
   - Manifold dispersion (`dispersion (rank-norm)`) achieves an AUC of **0.571** (`95% CI [0.527, 0.618]`), yielding a directional $+0.047$ gain over entropy baseline (`0.524`). On paired bootstrap difference analysis, the paired $\Delta$ CI spans zero ($[-0.009, +0.108]$), indicating directional gain without statistical separation at $n=600$.
   - The learned center directions occupy near-orthogonal positions (pairwise distances 0.9032–1.0132, relative spread 0.113), meaning the model's centers lack manifold distance structure to exploit.

2. **Policy Threshold Limitations**:
   - `second_mass` yields an AUC of **0.425** ($95\%\text{ CI } [0.380, 0.473]$, paired $\Delta$ CI $[-0.163, -0.031]$), which is statistically below chance.
   - The ALTERNATIVES branch fires on 37–48% of traffic using `second_mass`, a statistic measured at AUC 0.425 — reliably anti-correlated with ambiguity. Whether that routing is wrong cannot be determined with binary labels; testing it requires PAQA sub-type annotations.

3. **Data Saturation & Formatting Leakage**:
   - 48% of open-domain QA rows produce near-uniform probabilities ($\max P < 0.30$), setting a hard ceiling on scalar threshold policies over arbitrary trivia queries.
   - Leakage ablation confirmed that surface formatting markers account for a $+6.5\%$ artifact shift in raw accuracy, while repaired context normalisation improves model behavior consistency without altering binary classification boundaries.
