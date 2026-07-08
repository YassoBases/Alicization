# Deferred work

Each item: one-line spec + what blocks it. Level-6 competence tracking is
NOT here — it is Stage 7's P8.1, no longer deferred.

- **k=100 forecaster horizon** — add 100 to `ledger.horizons` as a default
  (the code already accepts arbitrary horizons; the full battery sweeps it).
  Blocked by: base-scale training runs long enough that 100-tick futures
  accumulate meaningful signal (smoke-scale tuple stores are too small).

- **Counterfactual module with snapshot ground-truth re-rolls** — estimate
  "what would have happened under action b" by restoring a world snapshot,
  stepping the alternative, and training a counterfactual head against the
  re-rolled truth. Blocked by: an experimenter-side re-roll harness that
  keeps the ground-truth strictly evaluation-only (same discipline as
  `training/attribution_eval.py`).

- **Agent-state graph** — persist the (position, plan, intero-bucket)
  transition graph as a queryable artifact for the dashboard. Blocked by:
  arbiter-mode runs long enough for the graph to be non-trivial; a
  dashboard page to render it.

- **NCA core** — a neural-cellular-automaton core as a third
  `agent.core` option (locality-constrained persistent state). Blocked by:
  the GRU/RSSM comparison batteries finishing at base scale first, so the
  third core has stable baselines to be judged against.

- **Multi-env scaling** — true parallel world stepping (subprocess or
  vectorized numpy worlds) to lift the CPU sps ceiling. Blocked by: nothing
  conceptually; deferred until base-scale runs become the bottleneck (the
  determinism/state-hash tests must survive the parallelization).

- **LLM readout layer** — verbalize Ledger evidence into proposal
  rationales ("MOVE_E success collapsed at tick 41k; body model refreshed;
  suggest recalibration run"). DATA ONLY, per the proposal-layer rule in
  docs/safety_scope.md — a human executes every approved change. Blocked
  by: the proposal-layer plumbing itself (queue + review UI), and the
  safety_scope data-not-code rule is a precondition, not an afterthought.
