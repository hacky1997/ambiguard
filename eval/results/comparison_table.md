## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (600 samples)

| Arm | Beh. Acc | Bal. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|---|
| CenterDistill | 0.503 | 0.503 | [0.463, 0.543] | 4.9 | 459 ms | 519 ms | $0.00 | yes |
| LLM judge (gpt-4o-mini) | 0.533 | 0.533 | [0.493, 0.575] | 3.2 | 738 ms | 1127 ms | $4.39 | yes |
| Majority class (ANSWER) | 0.500 | 0.500 | [0.460, 0.540] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.492 | 0.492 | [0.452, 0.532] | 0.9 | 463 ms | 561 ms | $0.00 | yes |

*Generated: 2026-08-06T18:41:34Z*
