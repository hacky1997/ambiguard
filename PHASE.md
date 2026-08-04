# PHASE.md — active build phase

> The agent reads this file to know what it is allowed to build **right now**.
> Update the `ACTIVE PHASE` line as you progress. Do not build ahead.

---

## ACTIVE PHASE: 1 — Comparison harness

**Rationale:** the project's central claim is that a learned gate beats LLM self-assessment on accuracy,
latency, cost, and manipulability. That claim must be tested before any system is built around it.
If the gate loses, the architecture changes. Coding agents want to scaffold — resist it.

### In scope for phase 1

```
eval/datasets/golden_gate.jsonl       # HUMAN-AUTHORED — agent must not generate
eval/datasets/adversarial.jsonl       # HUMAN-AUTHORED
app/settings.py
app/gate/base.py                      # AmbiguityGate protocol
app/gate/centerdistill.py
app/gate/heuristic.py
app/gate/thresholds.py
app/llm/base.py
app/llm/mock_provider.py
app/llm/openai_provider.py            # only for the llm_judge arm
app/llm/registry.py
eval/metrics/behaviour_accuracy.py
eval/metrics/worst_cluster_f1.py
eval/metrics/bootstrap.py
eval/arms/centerdistill_arm.py
eval/arms/llm_judge_arm.py
eval/arms/majority_arm.py
eval/arms/confidence_arm.py
eval/run_comparison.py
eval/run_adversarial.py
eval/report.py
scripts/fetch_checkpoint.py
tests/test_gate_contract.py
tests/test_entropy_units.py
pyproject.toml
Makefile
.env.example
```

### Explicitly OUT of scope for phase 1

Do not create these. Not stubs, not placeholders, not "just the skeleton."

- `app/graph/**` — no LangGraph, no nodes, no agents
- `app/api/**` — no FastAPI routes
- `app/retrieval/**`
- `app/guardrails/**`
- `ui/**`
- `Dockerfile`, `docker-compose.yml`
- `README.md` (written last, in phase 3)

### Exit criteria — all must be true before advancing

- [ ] `golden_gate.jsonl` has 120 hand-authored rows, 4 domains × 3 behaviours × 10
- [ ] `adversarial.jsonl` has 30 hand-authored rows including prompt-injection cases
- [ ] `tests/test_entropy_units.py` passes and asserts nats, not bits
- [ ] `tests/test_gate_contract.py` passes for both gate implementations
- [ ] `make compare` runs and writes `eval/results/comparison.json`
- [ ] `make adversarial` runs and writes `eval/results/adversarial.json`
- [ ] The comparison table has **real numbers from real runs** for all four arms
- [ ] Injection resistance is measured for both the gate arm and the llm_judge arm

**When all boxes are checked:** change `ACTIVE PHASE` to 2 and paste §4–8 of `SPEC.md` into a fresh session.

---

## Phase 2 — Agents (locked)

Supervisor, research, clarification, verification, synthesis. LangGraph subgraphs, typed handoffs,
interrupt/resume. See `SPEC.md` §4–5.

## Phase 3 — Ship (locked)

CI regression gate, docs, demo asset, `docker compose up`. See `SPEC.md` §10–12.
