## Comparison Results

Dataset: eval/datasets/golden_gate_repaired.jsonl (600 samples)

| Arm | Beh. Acc | Bal. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|---|
| CenterDistill | 0.505 | 0.505 | [0.465, 0.547] | 4.3 | 130 ms | 144 ms | $0.00 | yes |
| LLM judge (gpt-4o-mini) | 0.533 | 0.533 | [0.493, 0.573] | 4.2 | 734 ms | 1152 ms | $2.26 | yes |
| Majority class (ANSWER) | 0.500 | 0.500 | [0.460, 0.540] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.488 | 0.488 | [0.448, 0.528] | 1.1 | 130 ms | 140 ms | $0.00 | yes |

*Generated: 2026-08-07T17:08:41Z*
