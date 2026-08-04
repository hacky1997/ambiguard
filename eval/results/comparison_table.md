## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (120 samples)

| Arm | Beh. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|
| CenterDistill | 0.258 | [0.183, 0.342] | 0.0 | 481 ms | 603 ms | $0.00 | yes |
| LLM judge (mock-v1) | 0.350 | [0.267, 0.433] | 3.2 | 0 ms | 0 ms | $0.00 | yes |
| Majority class (ANSWER) | 0.500 | [0.408, 0.592] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.250 | [0.175, 0.325] | 0.0 | 608 ms | 769 ms | $0.00 | yes |

*Generated: 2026-08-04T18:24:13Z*
