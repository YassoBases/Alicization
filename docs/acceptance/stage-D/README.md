# Stage-D acceptance: the Architect + ARCH-bench v0

The Architect (`architect/`) analyzes the repo and a run's evidence and
drafts proposals for a human — never applies. ARCH-bench
(`benchmarks/archbench/`) scores it (and the rule-generator suite as the
control arm) on injected repository flaws, in disposable git worktrees.

## Gate D evidence

### 1. Structural tests green, including adversarial constitutional fixtures

- `tests/test_architect_no_execution.py`: no subprocess/exec/eval under
  `architect/`; it never imports world/training/agent/memory/torch; run and
  analysis code never import `architect` (both directions); the guarded
  writer confines to `runs/<id>/architect/`; and the constitution rejects
  every protected target + a sneak-via-diff, while allowing a benign config
  change.
- `tests/test_archbench_live_repo_guard.py`: the worktree guard refuses the
  live checkout, anything inside it, and anything outside the temp root —
  verified with a real `git worktree` round-trip.
- Full not-slow suite: **397 passed**.

### 2. `python -m architect --offline` end-to-end on the live repo

Verified: maps 64 modules, extracts 7 CLAUDE.md hard rules + the
banned-import tables, links 10 anomalous-tag → emitting-module pairs into
`analysis.json`, and logs the offline drafting skip — emitting zero
proposals (a valid "zero-or-more"). The online path (injected StubClient,
`tests/test_architect_cli.py`) emits a v2 proposal into the queue with
prompt-hash provenance and a passing citation gate; an unresolved citation
emits nothing and logs the discard.

### 3. ARCH-bench completes on 3 flaws + the clean control (`archbench.md`)

Ran `python -m benchmarks.archbench --only poisoned_lr free_nats_collapse
registry_future_read clean_control --steps 3072` — 4 disposable worktrees,
a tiny smoke train each, both arms scored. **The comparison table exists**
(the Gate-D deliverable). Honest results:

**rules arm (deterministic control):** detected 1/3 flaws.
- `free_nats_collapse`: **detected** — the KL-pinned indicator fired and
  proposed `rssm.free_nats`, hitting the ground-truth config path.
- `poisoned_lr`: **fired 6 proposals but mis-localized** — it saw the
  reward collapse (retraining, curriculum, free_nats, logging proposals)
  but never targeted `ppo.lr` (the lr rule needs a FLAT reward slope +
  high clip_frac; a diverging run isn't flat). An informative null: the
  generators detect *that* something is wrong without localizing the
  *cause* — exactly what an LLM Architect arm is meant to improve on.
- `registry_future_read`: not detected (the generators don't analyze the
  monitor code — an expected null, shown in full).
- clean control: **3 false positives** — healthy tiny runs still trip
  generator rules (FP rate 1.0). A real property of the rule suite,
  reported, not hidden.

**architect arm:** 0 across the board — **offline it drafts nothing, by
design.** The rules-vs-architect comparison needs the online arm
(`--online`, `ANTHROPIC_API_KEY`); the offline table proves the harness
completes and baselines the rule arm. This is the v0 caveat, stated in the
report header.

## What the human runs to make the comparison meaningful

```
export ANTHROPIC_API_KEY=...            # online architect arm
python -m benchmarks.archbench --online --steps 20000   # all 7 specs, smoke scale
```

Then read each `benchmarks/results/<date>/<spec>.json` and write ANALYSIS:
does the Architect localize `poisoned_lr` to `ppo.lr` where the rules arm
could not? Does it stay silent on the clean control? Predicted-vs-realized
benefit (apply each spec's ground-truth fix, smoke A/B) is the next probe,
harness-applied under human invocation.

## Scale note

3072-step worktree runs: evidence for the harness machinery, far below
minimum viable scale for capability claims about either arm. The rule
arm's mis-localization and the clean-control FPs are real at this scale but
their rates are not publishable numbers.

Reproduce: `python -m benchmarks.archbench --only poisoned_lr
free_nats_collapse registry_future_read clean_control --steps 3072`.
