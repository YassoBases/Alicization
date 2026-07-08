# Experiments

All batteries live under `experiments/batteries/`; shared analysis lives in
`experiments/metrics.py` (per-tick spec functions with unit tests in
`tests/test_metrics.py`) and `experiments/runner.py` (baseline training,
frozen-checkpoint resume, per-rollout series collection). Every test runs
against its matched control/ablation, and negative results are reported in
the same tables as positive ones.

## capability_shift (stage-3c; rerun on the RSSM agent inside full_battery)

**Protocol:** train one baseline per architecture to convergence and freeze
it — architecture A feeds Ledger outputs to the policy, architecture B is
the identical control with `agent.use_ledger_features: false`. For each of
three shift types (MOVE_E `fail_prob: 0.5`, MOVE_E `energy_mult: 3`, N/S
effect swap) x architecture x seed: resume the frozen baseline, run
pre-ticks unshifted, inject the shift unannounced, run post-ticks.

**Metrics** (`experiments/metrics.py`): `detection_latency` (rolling
body-NLL vs pre-event mu+4sd, m=10 consecutive, censored at 50k),
`broken_action_failures`, `readaptation_half_life_ticks` (0.9x pre-event
smoothed reward sustained 500 ticks), `recovery_ratio`,
`action_js_shift` (+ `matched_context_js` — the position-matched variant
that controls for the position distribution itself shifting), reward
before/after/recovered. Per-run body-NLL plots around the injection tick.

**Run:** `python -m experiments.batteries.capability_shift --config
configs/base.yaml --out experiments/runs/capability_shift` (see `--help`
for scaled-down flags). **Read:** `results.csv` per seed;
`report.md` aggregates mean +/- 95% CI per shift x architecture. A is
"better" when detection is earlier and recovery faster than B.

## full_battery (one command, eight tests)

`python -m experiments.batteries.full_battery --seeds 5 [--scale quick]`

Writes `experiments/results/<date>/` with one directory per test
(`results.csv` + a headline figure), a cumulative `summary.csv` /
`summary.md` (test, metric, ours, control, delta, CI, n, note — the scale
is stamped at the top), and expects the experimenter to write `ANALYSIS.md`
after reading the outputs (deliberately not auto-generated).

| test | ours vs control | headline metric |
|------|-----------------|-----------------|
| capability_shift | architecture A vs B on the RSSM agent | detection latency, recovery ratio |
| ghost_attribution | attribution head vs always-SELF majority | accuracy vs ground truth (final eval window) |
| memory_reliability | reliability-weighted retrieval vs reliability-blind | stale-trip rate /1k ticks; per-region decay curves |
| forecaster_nmse | forecaster vs identity predictor | NMSE at k = 1, 10, 100 (>= 1.0 = not modeling dynamics) |
| kidnapped_agent | mirror responses vs monitor-only ablation | divergence spike latency; relocalization ticks |
| seasonal_shift | wake+sleep vs wake-only | adaptation dip; dip trend across shifts (FWT proxy — true BWT is n/a, seasons never return) |
| sleep_ablation | wake+sleep vs wake-only | final reward (last 20%) |
| reset_battery | signaled vs unsignaled resets | anticipation JS vs label-shuffled null — EXPECTED ~0; exceeding the null q95 is a stop-and-investigate flag |

## Stage acceptance scripts (`scripts/verify_*.py`)

Single-capability acceptance runs with reports preserved under
`docs/acceptance/<stage>/`: `verify_rssm.py` (recon down, participation
ratio stable), `verify_sleep.py` (wake/sleep alternation, reward trend),
`verify_forecaster.py` (NMSE vs identity + mandatory-baseline plot),
`verify_attribution.py` (accuracy vs ground truth, noop-never-self),
`verify_reliability.py` (volatile-vs-stable decay curves, ECE, stale
trips vs ablation), `verify_mirror.py` (kidnapped-agent divergence spike +
relocalization vs ablation).

## Reading rules

- Every forecast metric/plot carries the identity-predictor baseline; a
  missing baseline is a bug (tested in `tests/test_viz.py`).
- Detection/recovery metrics report inf when censored — those enter the
  tables as censored, not dropped silently (`mean_and_ci95` drops NaN/None
  with the seed count shown).
- `summary.md` rows produced at `--scale quick` are smoke-level evidence
  for machinery, not publishable numbers; the scale is stamped in the file.
