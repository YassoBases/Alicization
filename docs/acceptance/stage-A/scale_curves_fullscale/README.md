# Head-convergence scale curves — full run (stage-A follow-up)

`python scripts/scale_curves.py --seeds 3` over 5 log-spaced budgets
(20k → 200k). This is the first of the overnight full-scale sequence; it
pins the forecaster's and attribution head's minimum viable scales with
data, and it directly confirms the Stage-E forecaster deferral. Raw:
`curves.csv`, plot: `scale_curves.png`.

## Forecaster NMSE at k=10 (beats identity iff < 1)

| budget | seed-mean k10 NMSE | verdict |
|--------|--------------------|---------|
| 20,000 | ~17.4 | ≫ identity (useless) |
| 36,000 | ~3.8 | still > 1 |
| 63,000 | ~1.30 | at the boundary |
| 113,000 | ~0.57 (2/3 seeds < 1) | **beats identity** |
| 200,000 | ~0.57 (3/3 seeds) | **solidly beats identity** |

**The forecaster crosses NMSE < 1 between 63k and 113k** and is functional
(~0.5–0.66) by 200k. This confirms the stage-A finding that the forecaster
needs ~50k+ to converge, and it validates the Stage-E decision to treat the
smoke-scale (24k) forecaster comparison as machinery-only: at 24k the
forecaster sits at NMSE ~130 in *both* impls (non-functional), so a
SelfQ-vs-heads ratio there is not admissible parity evidence. The full-scale
`verify_selfq` (running next in the sequence) produces a meaningful
forecaster parity number precisely because 200k is past this crossing.

Follow-up: tighten `forecaster_nmse.known_sufficient` in
`experiments/batteries/full_battery.py:MIN_VIABLE_SCALE` toward ~113k (63k
is borderline) once the full sequence completes.

## Attribution accuracy vs the always-SELF majority (~0.99)

Honestly less encouraging — **bimodal and seed-fragile**. Per seed the
classifier either reaches ~0.99 or collapses to ~0.00, at every budget
including 200k (0.005 / 0.987 / 0.992). And where it works, ~0.99 only
*matches* the trivial always-SELF baseline rather than beating it. So at
these scales attribution does not cleanly earn its keep, and its training is
unstable across seeds — a real null that feeds back into the
ghost-attribution premise and its `MIN_VIABLE_SCALE` entry. Worth a
stability investigation (loss balance / pseudo-label thresholds) before
treating ghost-attribution numbers as architecture evidence.

Reproduce (resumable — skips finished points): `python scripts/scale_curves.py --seeds 3`.
