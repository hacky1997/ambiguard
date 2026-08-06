## Comparison Results

Dataset: eval/datasets/golden_gate.jsonl (600 samples)

| Arm | Beh. Acc | 95% CI | WC-F1 | p50 lat. | p95 lat. | $/1k | Det. |
|---|---|---|---|---|---|---|---|
| CenterDistill | 0.290 | [0.255, 0.327] | 2.1 | 443 ms | 598 ms | $0.00 | yes |
| CenterDistill (calibrated) | 0.502 | [0.462, 0.542] | 1.1 | 455 ms | 629 ms | $0.00 | yes |
| LLM judge (mock-v1) | 0.298 | [0.262, 0.335] | 2.8 | 0 ms | 0 ms | $0.00 | yes |
| Majority class (ANSWER) | 0.500 | [0.460, 0.540] | 0.0 | 0 ms | 0 ms | $0.00 | yes |
| Confidence threshold (τ=0.44) | 0.257 | [0.222, 0.292] | 0.0 | 463 ms | 707 ms | $0.00 | yes |

*Generated: 2026-08-06T11:45:23Z*
