# Stage-C acceptance: evidence plane, schema v2, evaluation ladder

Stage C turned the proposal machinery into a provenance-carrying, tiered
pipeline: one shared read-only evidence plane (C1), a versioned proposal
schema with provenance and intervention classes (C2), the researcher's
agenda unified into that queue (C3), and a two-tier evaluation ladder (C4).

## Gate C evidence

### 1. Evaluation ladder — ≥10 tier-0 evaluations (`ladder_run.txt`)

`python scripts/verify_ladder.py` seeds a run with 11 distinct config knobs
(+ one experiment proposal) and runs `experiments.runner.run_ladder`:

- **11 tier-0 config-knob proposals** smoke-A/B evaluated, each marked
  `evaluation=smoke_ab`, each scoring the two independent metrics
  (`wm_loss`, `reward`) beside its own criterion.
- The `rssm.free_nats` knob judged on `sleep/kl` is flagged
  `[TAUTOLOGICAL]` — the criterion the knob mechanically moves (the
  proposal_quality ANALYSIS caveat, made machine-visible).
- The **experiment proposal was left `pending`** (tier-0 touches only
  `intervention_class=config`; experiments are the human-gated tier-1 path).
- Degenerate/absent controls produced **NaN, never astronomical numbers**
  (the PPO knobs' `wm_loss` is NaN because tiny PPO runs log no wm-loss
  tag — the guard degrading honestly, not inventing a number). At quick
  scale the reward benefits are ~0: this validates the machinery, not an
  architecture effect.

### 2. proposal_quality battery — green end-to-end on v2 (`proposal_quality_v2_summary.md`, `proposals.csv`)

`python -m experiments.batteries.proposal_quality --scale quick --seeds 3`
ran clean through the schema-v2 + evidence-plane migration. Every emitted
proposal is a valid v2 record carrying `provenance.{evidence_bundle_hash,
generator_id}` and an inferred `intervention_class` (the `hyperparameter`
knob → `config`, the rest → `experiment`). The result is the same shape as
the pre-v2 stage-7f run — ledger +72.97 control-sd benefit on the free_nats
A/B, `logs_only` silent (healthy runs give the raw-log rules nothing to
flag) — so **the migration changed the plumbing, not the numbers**. The
free_nats/KL result carries the same validity caveat stage-7f flagged; the
ladder's tautology flag now records it structurally.

### 3. Structural + dashboard tests

- `evidence/` is scanned by `tests/test_proposals_no_execution.py` (same
  no-exec + import bans as proposals/researcher); `tests/test_evidence.py`
  covers the logs_only strip, deterministic bundle hashing, snapshot
  injection/fallback, and code refs.
- Dashboard tests green: `test_dashboard_researcher.py` reads the unified
  queue (researcher experiment proposals ranked by `provenance.agenda_score`,
  generator proposals excluded); `test_dashboard_proposals.py` unchanged.
- Full not-slow suite: **349 passed** after C4.

## Reproduce

```
python scripts/verify_ladder.py                 # >=10 tier-0 smoke-A/B
python scripts/verify_agenda.py --run-dir <fixture>   # agenda -> queue (C3)
python -m experiments.batteries.proposal_quality --scale quick --seeds 3
pytest -q -m "not slow"
```

## Note on scale

All numbers here are quick-scale (CPU, tiny/short runs): evidence for the
pipeline machinery, below minimum viable scale for architecture claims
(the MIN_VIABLE_SCALE contract from stage-A applies). The ladder's tier-0
is a cheap screen by design; tier-1's fuller A/B (human-approved, longer
window) is the one whose numbers inform decisions.
