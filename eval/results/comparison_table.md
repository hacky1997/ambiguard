## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (600 samples)

| Arm | Beh. Acc | Bal. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|---|
| CenterDistill | 0.503 | 0.503 | [0.463, 0.543] | 5.1 | 450 ms | 498 ms | $0.00 | yes |
| LLM judge (gpt-4o-mini) | 0.533 | 0.533 | [0.493, 0.573] | 6.4 | 761 ms | 1112 ms | $4.39 | yes |
| Majority class (ANSWER) | 0.500 | 0.500 | [0.460, 0.540] | 6.7 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.492 | 0.492 | [0.452, 0.532] | 0.9 | 466 ms | 585 ms | $0.00 | yes |

*Generated: 2026-08-06T16:29:27Z*
