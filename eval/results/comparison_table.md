## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (600 samples)

| Arm | Beh. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|
| CenterDistill | 0.402 | [0.362, 0.442] | 2.7 | 442 ms | 569 ms | $0.00 | yes |
| LLM judge (gpt-4o-mini) | 0.708 | [0.672, 0.745] | 3.2 | 779 ms | 1112 ms | $1.83 | yes |
| Majority class (ANSWER) | 0.500 | [0.460, 0.540] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.257 | [0.222, 0.292] | 0.0 | 477 ms | 659 ms | $0.00 | yes |

*Generated: 2026-08-06T14:47:04Z*
