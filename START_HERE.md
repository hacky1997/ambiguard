# Setup kit — placement and usage

## Files and where they go

```
your-repo/
├── AGENTS.md      ← always-in-context rules (the silent-bug guardrails)
├── PHASE.md       ← scope gate; update ACTIVE PHASE as you progress
└── SPEC.md        ← rename AMBIGUARD_SPEC_V2.md to this
```

All three at repo root. Antigravity picks up `AGENTS.md` automatically. If it doesn't, add its contents
to the project's custom-instructions field.

## First session — exact prompt

Do **not** paste the whole spec. Feed phase 1 only, or the agent will start scaffolding `app/graph/`
and treat the comparison harness as later work.

```
Read AGENTS.md and PHASE.md first — both are binding.

We are in PHASE 1 only. Read SPEC.md sections 0 through 3 and ignore sections 4 onward
for now; those are locked.

Task: set up the project skeleton for phase 1 exactly as listed in PHASE.md "in scope".
Create nothing outside that list — no graph, no API, no Dockerfile, not even stubs.

Start with pyproject.toml, app/settings.py, and app/gate/ (base, heuristic, thresholds).
Stop after those and show me what you have before continuing.
```

That last line matters. Both Claude and Gemini Pro will otherwise produce forty files in one shot and
you'll spend longer reviewing than writing.

## Session hygiene

- **Fresh session per milestone.** Long sessions drift; the middle of a long spec gets lost.
  Each new session: `AGENTS.md` + `PHASE.md` + only the relevant `SPEC.md` section.
- **Model split:** Claude for `app/gate/` and the state contracts (silent-bug territory, precision matters).
  Gemini Pro for boilerplate, config, packaging. Either handles the eval scripts.
- **After any eval work:** verify every number in a results table traces to a committed
  `eval/results/*.json`. Fabricated tables are the most common failure and the most damaging one.

## The two things you do yourself

1. **`golden_gate.jsonl` — 120 rows, by hand.** Two evenings. Non-negotiable: LLM-generated labels
   evaluating a router is circular reasoning and an interviewer will catch it. Side benefit — authoring
   these is the best interview prep available, because you'll internalize exactly where the gate is weak.

2. **`adversarial.jsonl` — 30 rows, by hand.** Especially the prompt-injection cases. These produce the
   single strongest row in the comparison table: the LLM-judge arm is manipulable by instructions hidden in
   retrieved context; the gate structurally is not.

## Checkpoint — resolve before day 3

Two open questions branch `scripts/fetch_checkpoint.py`:

- Full fine-tuned XLM-R-large (~2.2 GB) or LoRA adapters? LoRA needs `peft` plus a separate base-model pull.
- Local disk only, or already on HF Hub? Hub availability enables an HF Spaces demo running real weights.

Until resolved, the heuristic gate keeps everything runnable — but the comparison table cannot be published
with fallback numbers.
