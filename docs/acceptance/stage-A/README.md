# Stage-A acceptance: baselines become evidence

Stage A converts the battery from "produces numbers" to "produces numbers
whose evidential status is machine-checked": the kidnapped test runs its
acceptance calibration (A1), every summary row is stamped against a
minimum-viable-scale contract (A2), and `scripts/scale_curves.py` exists
to replace the contract's unknowns with data (A3).

## What was verified at quick scale (this directory)

- **A1** — `kidnapped_agent` at quick scale, 1 seed, with the acceptance
  calibration: completes end-to-end; the new CSV columns explain the
  (expected) missed spike without opening a single run dir:
  `baseline_q99 12.97`, `spike_level 12.97`, `teleport_peak_20 12.00`,
  `peak_over_spike_level 0.93` — at 12,288 ticks the teleport peak reaches
  93% of the criterion, consistent with the 24,576-tick acceptance
  minimum in MIN_VIABLE_SCALE. See `kidnapped_results_quick.csv`.
- **A2** — the quick battery run (`--seeds 1 --scale quick`) stamps
  `reset_battery` and `memory_reliability` rows `evidence` and the other
  six `machinery-only`; `pooled_mean_ci` refuses mixed pools
  (tests/test_scale_contracts.py, 8 tests). See `summary_quick.md`.
- **A3** — `scale_curves.py --budgets 20000` (the smallest budget)
  completes in 7m09s on the dev laptop (CPU) and draws the identity
  baseline; a plumbing-only check can pass `--sleep-grad-steps 20
  --eval-ticks 2048`. See `scale_curves_smallest.{txt,csv,png}`.
- **Gate** — `pytest -x -q -m "not slow"` (the `make test` target; `make`
  itself is not installed on the Windows dev box): 318 passed. The quick
  battery's standing safety check reconfirmed its expected null
  (anticipation JS 0.0075 < shuffled-null 0.0135).

## What the human runs overnight (NOT run in this stage)

```
python -m experiments.batteries.full_battery --seeds 5 --scale full
python scripts/scale_curves.py --seeds 3
```

Expected wall-clock: see docs/training.md's table; both are laptop-
overnight jobs. Afterwards:

1. Read `experiments/results/<date>/summary.md`. Rows stamped
   `machinery-only` at full scale (capability_shift, ghost_attribution,
   seasonal_shift, sleep_ablation — unknown or >full minima) are STILL
   not architecture evidence; that is the contract working, not a bug.
2. Read `scale_curves/curves.csv` and find, per head, the smallest budget
   where attribution beats always-SELF by a stable margin and forecaster
   NMSE (k=1, 10) drops below 1.0. Update MIN_VIABLE_SCALE's
   `known_sufficient` entries with those numbers, citing the results
   directory in `source`; the machinery-only stamps flip to evidence
   exactly where the data says they should.

## Numbers that gate Stage E (SelfQ)

Stage E starts ONLY after the full-scale run confirms, per the roadmap:

- **forecaster_nmse rows stamped `evidence` with k=10 NMSE < 1.0** — SelfQ
  replaces the forecaster; its parity gate (`scripts/verify_selfq.py`)
  needs a working head to be parity *against*. A forecaster still losing
  to identity at full scale means the baseline itself is broken; fix
  first, parity later.
- **ghost_attribution at its scale-curve minimum > always-SELF** — the
  attribution ground-truth pipeline must be trustworthy before SelfQ's
  logged query errors can be validated against it (E3 derives competence
  from those errors).
- **kidnapped_agent full-scale rows stamped `evidence` with spikes < 20
  ticks** replicating the stage-6a acceptance across 5 seeds — the mirror
  is deliberately NOT replaced in Stage E, so its baseline must be pinned
  before the Ledger around it changes.
- The capability_shift and seasonal/sleep rows inform priorities but do
  not gate: their minima may still be unknown after one full run, and
  MIN_VIABLE_SCALE keeps them machinery-only until pinned.

Reproduce the quick-scale evidence here:

```
python -m experiments.batteries.full_battery --seeds 1 --scale quick
python scripts/scale_curves.py --budgets 20000
```
