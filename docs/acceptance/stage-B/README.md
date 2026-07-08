# Stage-B acceptance: sequential monitors (CUSUM frozen baseline)

**Claim under test:** the `cusum_frozen_baseline` detector catches
persistent changes that `mean_shift` provably cannot escalate on, with
zero pre-lever false alarms — closing the onset-only detectability window
documented in docs/researcher.md.

## Blind-spot construction (`verify_registry.py --blind-spot`)

Frozen-policy fixture (the lever is the only nonstationarity), window
1000, cadence 250, arm_after 500. The lever lands at tick 1600 — after
CUSUM's baseline freeze completes (arm_after + window = 1500), before
mean_shift's first armed check (arm_after + 2·window = 2500) minus one
window: mean_shift gets exactly ONE check with a clean prev-window
straddling the onset (2500); at 2750 the prev-window is contaminated,
its inflated variance collapses the statistic, and the lone weakening
"recovers" — never contradicted, by geometry, not bad luck.

**Result (3 seeds — `blind_spot_run.txt`):**

| seed | lever | mean_shift | CUSUM contradiction | pre-lever false alarms |
|------|-------|------------|---------------------|------------------------|
| 0 | 1601 | 1 weakening @2500, recovered — never contradicted | @2250 (+649) | none |
| 1 | 1601 | 1 weakening @2500, recovered — never contradicted | @2250 (+649) | none |
| 2 | 1601 | 1 weakening @2500, recovered — never contradicted | @2250 (+649) | none |

CUSUM detects in +649 ticks (budget: 6 checks = 1500), identically
across seeds (frozen policy + seeded world make the trace family tight).
Post-lever downstream contradictions on correlated actions (MOVE_W /
MOVE_N — the position distribution shifts when MOVE_E fails 90%) are
reported, expected, and post-lever only.

Note CUSUM contradicts at 2250 — BEFORE mean_shift's first armed check
even runs (2500): the single-window freeze arms a full window earlier,
which is part of the win at short horizons.

## Stage-8a mode re-verified under the new default

`verify_registry.py` (default mode, lever 4500, CUSUM default):
weakening @5000 (S=5.1), contradicted @5250 (S=9.4, lever 4501), zero
pre-lever transitions anywhere, downstream MOVE_W @5750 reported as
legitimate (`stage8a_mode_cusum.txt`). The stage-8a acceptance stays
green with the detector swapped.

## Researcher-value latency arm: censoring before/after

Same battery command, quick scale, 3 seeds, lever at 2000; the battery
file is untouched — it inherits the CUSUM default through
`build_default_hypotheses`.

| detector (run) | per-seed latency | censored |
|----------------|------------------|----------|
| mean_shift, lever 1500 (stage-8d, first run) | — / — / — | **3/3** |
| mean_shift, lever 2000 (results/20260708-2141) | 750 / 750 / censored | **1/3** |
| CUSUM, lever 2000 (this stage — `researcher_value_cusum.md`) | 750 / 750 / censored | **1/3** |

Honest reading: at this lever geometry mean_shift could already see the
change, so CUSUM matches it (750/750) rather than beating it — the win
shows where the geometry is hostile (the blind-spot table above, and the
first run's 3/3 -> this design's 1/3). Seed 0 censors under BOTH
detectors but for different reasons, and the difference matters: this
seed's policy left MOVE_E at a noisy 0.89 baseline (frozen sigma 0.31,
so each post-lever z is ~3x smaller than the other seeds'), and at the
12k-tick run end the persisted detector state shows **S = 3.88 of h = 5
and rising** — censored by RUN LENGTH, with detection 1–2 checks away.
mean_shift's seed-0 censor was a closed detectability window that no
additional run length could reopen. Quick scale's 3,072-tick horizon is
simply short for a sigma-0.31 baseline; the full-scale battery run
resolves this seed class.

## Gate B

- `verify_registry.py --blind-spot --seeds 3` passes (above).
- All registry tests green (27, including the future-bound replay-safety
  invariant parametrized over mean_shift AND cusum — the stage-8a
  future-read bug stays fixed on every path); full not-slow suite 325.
- Changes confined to researcher/, scripts/, tests/, configs/, docs/.

Reproduce:

```
python scripts/verify_registry.py --blind-spot --seeds 3
python scripts/verify_registry.py                    # stage-8a mode
python -m experiments.batteries.researcher_value --scale quick --seeds 3
```
