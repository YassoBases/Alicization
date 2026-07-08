# The proposal layer

Proposals are DATA, NEVER CODE (CLAUDE.md Hard rules): the agent's
diagnostics generate recommendations; a human reviews them blind and
executes every approved change by hand. Structural tests
(`tests/test_proposals_no_execution.py`) ban execution machinery, restrict
imports, and confine writes to `runs/<id>/proposals/`.

## Schema (`proposals/schema.py`, `schema_version: 2`)

| field | meaning |
|-------|---------|
| `id`, `type`, `created_tick`, `run_id` | identity; `type` from the 10-entry enum |
| `source` | OPEN string (v2), validated non-empty — the ledger-vs-logs control condition (`ledger`/`logs_only`) plus stage-D `architect`; BLINDED everywhere until `status=evaluated`. Blinding keys off status, never off a known-source list |
| `intervention_class` | `config` \| `experiment` \| `architecture` (v2): what KIND of change. Knob proposals are `config` (tier-0 auto-A/B eligible); everything else `experiment`; the Architect emits `architecture` |
| `provenance` | `{evidence_bundle_hash, generator_id}` for rule generators, `+ {prompt_hash, model_id}` for LLM-drafted ones — reproducibility (standing rule) |
| `artifacts` | run-relative paths to attached data files (e.g. an UNAPPLIED diff); never absolute, never containing `..` |
| `rationale` | templated text citing specific records (`tb:<tag>@step=`, `competence:…`, or `code:<path>@<sha>#Lx-Ly`) |
| `expected_benefit` | `{metric, direction, magnitude_estimate}` |
| `confidence` | [0,1]; heuristic at first, recalibrated from binned hit rates once ≥20 evaluated |
| `supporting_observations` | log/code-record refs — validated against real records in the acceptance |
| `estimated_cost` | `{human_hours, gpu_hours}` |
| `risks`, `success_criteria` | criteria: `{metric, threshold, eval_window_ticks}` — the proposal is judged against its OWN criteria |
| `status` | pending → (approve/reject/postpone/partial/modify) → approved/… → evaluated; legal transitions enforced (`tests/test_review_state_machine.py`) |
| `decision` | timestamp, note, optional 1–5 usefulness rating, human diff for modifications |
| `proposed_change` | optional machine-readable knob `{config_path, new_value}` — enables A/B evaluation; absent for non-knob proposals |
| `linked_experiment_id`, `realized_benefit` | filled by the evaluation run |

`Proposal.from_json` reads v1 records and migrates them (fills `provenance`/
`artifacts` empty, infers `intervention_class` = `config` if a
`proposed_change` knob is present else `experiment`).

## Generators (`proposals/generator.py`) — trigger conditions

Every generator runs on BOTH evidence variants each pass (`ledger` full
diagnostics; `logs_only` with all Ledger scalars + competence stripped).
Dedup by hash(type, target) survives restarts; per-type rate limiting;
every FIRED/SUPPRESSED decision logged with its reason.

| generator | ledger trigger | logs_only trigger |
|-----------|----------------|-------------------|
| retraining | region degrading with wm-loss ratio >1.5× AND replay coverage <2% | reward <70% of peak AND thinnest-region visitation <2% |
| training_schedule | ≥2 regions mid-adaptation with positive learning progress | positive reward slope + skewed visitation entropy (<75% of uniform) |
| hyperparameter | KL pinned at the free-nats floor (`rssm/kl` or `sleep/kl`) → halve free_nats; else flat reward + clip_frac >0.2 → lr/3 | the lr rule only |
| memory_policy | stale-trip rate >50/1k (cites ECE) | never (memory telemetry IS Ledger data) |
| checkpoint_schedule | participation-ratio dip below 25% of max → halve interval | loss/total NaN near-miss or >6σ jump |
| evaluation | reliability-ECE slope >0.002/rollout → run memory_reliability battery | reward variance doubling between run halves → full battery |
| logging_change | reward coefficient of variation >3 with no decomposing scalar → name the missing scalar | same rule (different suggested scalar) |
| compute_budget | sleep windows ending <90% of grad-step budget → set budget to observed mean | same (grad_steps is a raw log) |
| dataset_extension, visualization | schema-supported stubs; never fire | — |

## Blind-review protocol (`python -m review`)

`list`/`show` hide `source` until a proposal is evaluated — in the CLI and
the dashboard alike — so the reviewer cannot favor a variant. Approval
emits a ticket (`experiments/tickets/<id>.md`) whose exact command a HUMAN
runs; `python -m experiments.runner --ticket <id>` performs the evaluation
and writes `realized_benefit` back (which is what unblinds the record).
Decisions append to an immutable `decisions.jsonl`. Rejected and evaluated
are terminal for decisions; a rejected recommendation coming back is
measured (repeated-after-denial), not re-decided.

## Realized-benefit evaluation rules (Section 17)

- **Preferred: A/B.** When `proposed_change` exists: a seeded control run
  vs the identical run with the knob applied;
  `benefit = (mean(M_treated) − mean(M_control)) / std(M_control)`.
- **Fallback: pre/post with drift correction** (marked
  `evaluation=pre_post`): post-window mean minus the pre-trend
  extrapolation, normalized by pre std — a continuing trend earns zero.
- **Success-criteria hit**: did M cross the proposal's own threshold within
  its own window. **Confidence ECE**: 10-bin, reported with counts (don't
  over-read below ~50 evaluated). **Acceptance rate**: explicitly weak —
  reviewer behavior in numerator and denominator; never headline alone.
  **Usefulness (1–5)** reported alongside benefit; rated-useful-but-
  no-benefit is an interesting cell, not an error.

The flagship comparison lives in
`experiments/batteries/proposal_quality.py`; its hypothesis (ledger beats
logs_only on benefit and calibration) is stated in every report header and
answered honestly either way.
