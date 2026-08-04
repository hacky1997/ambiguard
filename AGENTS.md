# AGENTS.md — always-in-context rules

These rules apply to every task in this repo. They are short on purpose: they are the constraints that,
when forgotten, produce code that **runs, looks correct, and is silently wrong**.

## Non-negotiable

1. **Entropy is computed in NATS.** `-(p * np.log(p)).sum()`. Never `log2`.
   `tau_ent = 1.51` is a nats threshold. Using log2 misroutes every query by a factor of `ln 2`,
   raises no error, and produces plausible output. There is a contract test for this — do not delete it.

2. **Threshold evaluation order is ANSWER → ALTERNATIVES → CLARIFY.**
   ```
   if max_prob > tau_conf:        ANSWER
   elif second_mass > tau_multi:  ALTERNATIVES
   elif entropy > tau_ent:        CLARIFY
   else:                          ANSWER      # conservative default
   ```
   Reordering changes results. Do not "simplify" this.

3. **The app must boot with a completely empty `.env`.** Every external dependency degrades gracefully:
   no checkpoint → heuristic gate; no API key → mock LLM provider; no Qdrant → in-memory retrieval.
   If a reviewer clones this repo and it fails to start, the project has failed.

4. **Never present heuristic-fallback numbers as CenterDistill results.** Not in logs, not in the README,
   not in any results table. When `fallback_used=True`, every downstream surface must say so.

5. **Never fabricate evaluation numbers.** Every value in a results table must trace to a committed
   `eval/results/*.json` produced by an actual run. If a script has not been executed, the table stays empty.

6. **Do not generate the golden dataset with an LLM.** `eval/datasets/golden_gate.jsonl` is hand-authored by
   the human. Evaluating a router against LLM-authored labels is circular. If those files are missing,
   stop and say so — do not fill them in.

## State and graph

7. **Never invent state keys.** If a node needs new data, add the field to `AgentState` in
   `app/graph/state.py` first, then use it.

8. **Nodes return partial dicts. No in-place mutation of state.**

9. `messages` is the only reducer field (`Annotated[list[dict], add]`). Everything else is last-write-wins.

10. **`resolved_question` takes precedence over `question`** everywhere downstream when it is set.

11. **Verification retries are bounded at 2**, then escalate by downgrading the response to CLARIFY.
    Never an unbounded reject/retry loop.

## Code quality

12. **No bare `except:`.** Catch specific exceptions, log with context, write to `state["error"]`.
    The graph must always reach the terminal node.

13. **Type hints everywhere.** `mypy --strict` must pass on `app/gate/` and `app/graph/`.

14. **Config lives in `app/settings.py`** (pydantic-settings). No magic constants anywhere else.

15. **Commit at every milestone boundary**, message naming the milestone. The commit history gets read.

16. If something is genuinely underspecified, take the option that keeps the repo runnable **without
    credentials**, and leave a `# DECISION:` comment explaining the choice. Do not ask and stall.

## Scope discipline

17. **Do not build ahead of the current phase.** Check `PHASE.md` for the active phase. If a file belongs
    to a later phase, do not create it — even if it seems obviously needed.

18. When time is constrained, cut features in this order: UI → ALTERNATIVES path → Qdrant → web-search
    tools → synthesis agent. **Never cut:** the comparison harness, the adversarial suite, the CI
    regression gate, the verification agent.
