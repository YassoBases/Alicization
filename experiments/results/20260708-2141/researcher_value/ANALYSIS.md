# Researcher-value battery — analysis (quick scale, 3 seeds)

**Question:** does following the agenda reduce more uncertainty per unit
budget than uniform-random or greedy raw-uncertainty selection?

**Verdict: null at this scale, on every axis.** Reported with the same
prominence a positive would have gotten.

## 1. Uncertainty reduction: no arm beats drift (the headline)

All three arms land slightly *negative* after drift correction
(agenda −0.000029, random −0.000032, greedy −0.000035, overlapping
CI95s): 20 extra region-targeted grad steps reduce a region's ensemble
disagreement marginally *less* than the same budget of uniform replay.
Plausible mechanism: uniform sampling already covers every region while
also improving the shared encoder/dynamics trunk; hard-biasing 100% of
sampling into one region trades that global improvement for local data
the trunk has largely absorbed. At 12k-step lives with a 4-head
ensemble, region disagreement is dominated by trunk quality, not by
region-specific data starvation. A real test of the agenda needs
wake-phase execution (directed visits collecting *new* data) rather
than replaying old data — this harness only operationalizes
targeted_replay, as stated in the module docstring.

## 2. EIG calibration: no evidence, mildly negative

Spearman rho = −0.33 over 12 agenda items (predicted gain vs realized
drift-corrected reduction). With realized reductions that are
themselves ~noise (point 1), a rank correlation against them measures
little; but the honest statement is that v2's predicted_gain has **no
demonstrated positive calibration** at this scale. The scatter is in
eig_calibration.png.

## 3. Contradiction-detection latency: 750 ticks (2/3 seeds), 1 censored

Seeds 1 and 2: the MOVE_E capability hypothesis went weakening@2500 →
contradicted@2750 against a lever at 2001 — latency 750 ticks, exactly
the geometry prediction (first armed check + one cadence). Seed 0: the
MOVE_E hypothesis itself never fired (its usage pattern left the shift
statistic below threshold at the two armed checks inside the
detectability window), but the *downstream* hypothesis — MOVE_W
stability, which the lever perturbs via the position distribution —
was contradicted at 2750. The lever was detected, the targeted
hypothesis censored; per-protocol we report the censor, not the
consolation.

The first run of this battery censored on ALL seeds with a lever at
1500: mean_shift is an onset detector and its detectability window
(now-window straddling the onset, prev-window still clean) fell
entirely before the first armed check. Documented as a known monitor
limitation in docs/researcher.md; the geometry comment in
researcher_value.py has the arithmetic.

## 4. Agenda stability

Kendall tau = 0.822 between the pre- and post-execution agendas
(descriptive): execution barely moved the ranking, consistent with
point 1 (the executed items changed little).

## 5. Scale disclaimer & provenance

- quick scale (12,288-step lives, K=4 items x 20 grad steps, 3 seeds,
  CPU). The full protocol (200k lives, N≥5 seeds, K=8x100) remains
  future work; treat everything above as plumbing validation with an
  honest null, not as evidence the agenda is worthless.
- Budget equalization: the first run gave the agenda arm 7 items vs 12
  (too few region questions); top_k_cells is now widened so every arm
  executes the same K items per seed.
- Execution is offline targeted replay only; directed visits and probe
  batches need wake-phase control and are the natural next harness.
