## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (600 samples)

| Arm | Beh. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|
| CenterDistill | 0.502 | [0.462, 0.542] | 1.1 | 500 ms | 701 ms | $0.00 | yes |
| LLM judge (gpt-4o-mini) | 0.722 | [0.685, 0.758] | 3.7 | 751 ms | 1037 ms | $1.83 | yes |
| Majority class (ANSWER) | 0.500 | [0.460, 0.540] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.257 | [0.222, 0.292] | 0.0 | 487 ms | 696 ms | $0.00 | yes |

*Generated: 2026-08-06T12:38:20Z*
