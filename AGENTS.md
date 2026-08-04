# AGENTS.md

Binding rules for all work in this repo. These are the constraints that, when violated, produce code that
runs, looks correct, and is wrong.

## Correctness

1. **Entropy in nats.** `-(p * np.log(p)).sum()`. Never `log2`. `tau_ent` is a nats threshold; using log2
   misroutes every query, raises no error, and produces plausible output. A contract test asserts this.

2. **Threshold order is fixed:**
   ```
   if max_prob > tau_conf:        ANSWER
   elif second_mass > tau_multi:  ALTERNATIVES
   elif entropy > tau_ent:        CLARIFY
   else:                          ANSWER
   ```
   Do not reorder or "simplify".

3. **Boots with an empty `.env`.** No checkpoint → heuristic gate. No API key → mock provider. No vector DB
   → in-memory retrieval. A clone that fails to start is a failed project.

4. **Heuristic-gate output is never reported as CenterDistill output.** When `fallback_used=True`, every
   surface says so.

## Evidence

5. **Never fabricate a number.** Every value in any results table must come from a committed
   `eval/results/*.json` produced by an actual run. No script run → table stays empty. Do not fill in
   plausible-looking placeholders.

6. **Ground truth is external human annotation or deterministic derivation. Never an LLM, never the teacher.**
   - Never label ambiguity by prompting an LLM. The comparison pits the gate against an LLM judge; LLM
     labels hand that arm the win by construction.
   - Never evaluate against MLQA behaviour labels. They are teacher-induced and the same teacher trained
     the student — that measures distillation fidelity, not correctness. MLQA is permitted only as an
     adapter sanity check, reported as *teacher-agreement*, never as accuracy.
   - Every dataset row carries `source`, `source_id`, `annotation_provenance`. A row without provenance is
     a bug.

7. **Do not invent dataset identifiers, payload strings, or example rows.** If a source dataset is not
   downloaded, stop and say so.

## State and graph

8. Never invent state keys. Add the field to `AgentState` first, then use it.
9. Nodes return partial dicts. No in-place mutation.
10. `messages` is the only reducer field. Everything else is last-write-wins.
11. `resolved_question` takes precedence over `question` downstream when set.
12. Verification retries bounded at 2, then escalate to CLARIFY. Never an unbounded loop.

## Code

13. No bare `except:`. Catch specific exceptions, log with context, write `state["error"]`. The graph always
    reaches its terminal node.
14. `mypy --strict` passes on `app/gate/` and `app/graph/`.
15. Config lives in `app/settings.py`. No magic constants elsewhere.
16. Commit at each milestone, message naming it.

## Scope

17. **Build only the active phase.** The active phase is declared at the top of `SPEC.md`. If a file belongs
    to a later phase, do not create it — not even a stub.
18. When cutting scope: UI → ALTERNATIVES path → external vector DB → web-search tools → synthesis agent.
    Never cut: comparison harness, adversarial suite, CI regression gate, verification agent.
19. If something is underspecified, choose the option that keeps the repo runnable without credentials and
    leave a `# DECISION:` comment. Do not stall.
