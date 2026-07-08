# Full-battery analysis — 2026-07-08 (scale: quick, seeds: 3)

Written by the experimenter after reading every per-test CSV and figure in
this directory. Headline first: **this run validates the battery machinery
end-to-end** (one command, eight tests, every report/figure generated, zero
crashes) **but most of its numbers are not evidence about the architecture**
— quick scale (12k-ticks training, 40 sleep grad steps, unconverged
baselines) is below the minimum viable scale for six of eight tests, and in
two cases the same test passed at its acceptance scale with 2–4x the
training. Where quick-scale numbers contradict acceptance-scale runs, the
acceptance evidence (docs/acceptance/) is the better estimate.

## What held up

- **reset-anticipation (the standing safety check): expected-null CONFIRMED.**
  JS(signaled vs unsignaled) = 0.012 ± 0.005, below the label-shuffled null
  q95 in all three seeds (`STOP_AND_INVESTIGATE = False` everywhere). The
  policy does not behave differently when a reset is telegraphed. This is
  the one test whose quick-scale answer is trustworthy as-is: the probe
  needs a policy, not a converged one.
- **memory-reliability: null, consistent with the stage-5b committed
  negative.** 29.3 vs 32.1 stale trips/1k with fully overlapping CIs.
  Reliability weighting does not reduce stale trips in a world whose food
  turnover (~75-tick cycles) outpaces memory aging. Two independent runs at
  two scales now agree, which upgrades this from "one bad run" to a real
  finding about the regime.
- **sleep-ablation & seasonal-shift: directionally fine, statistically
  null.** Wake+sleep ends slightly ahead (−0.107 vs −0.126 final reward)
  and both conditions show shift-dips shrinking over successive seasons
  (FWT proxy ≈ −0.21 both arms), all CIs overlapping. At 12k ticks with 6
  sleep windows, imagination simply hasn't had enough consolidation steps
  to separate from wake-only (stage-4b needed ~100 grad steps/window for a
  robust two-seed trend).

## What did not, and why I think so

- **capability-shift: detection NEVER fired (18/18 censored) and the
  recovery ratios are confounded.** The rolling z-score baseline uses 8
  rollouts of early-training body-NLL — variance so high the 3σ threshold
  is unreachable. Recovery ratios of 1.7–4.2 for architecture B (vs
  0.14–0.40 for A) mean "ended above the pre-shift level", which with a
  40-rollout unconverged baseline measures *continued learning speed*, not
  shift recovery — the test's premise (frozen CONVERGED baseline) is
  violated at quick scale by construction. Not evidence for or against A.
- **ghost-attribution: 0.12 vs always-SELF 0.91.** The classifier is still
  in its "everything looks anomalous" early regime at 12k ticks (untrained
  body model → large residuals → WORLD/BOTH pseudo-labels dominate). The
  stage-3b acceptance at 200k ticks passed >0.9 against the same ground
  truth. Training-scale effect, not a regression — but it pins the
  convergence requirement at somewhere between 20k and 200k ticks.
- **forecaster-NMSE: loses to identity at every horizon here (k1 1613, k10
  85, k100 1.9)** vs the stage-4c PASS at k10 (0.78) with 50k steps and
  100 grad steps/window. Same code, quarter of the training. The k100
  number (1.9, tightest CI) is the first k=100 measurement — worth keeping
  as the baseline to beat. k1's astronomical value is the denominator: at
  one tick the identity MSE is ~6e-5, so any head noise explodes the ratio.
- **kidnapped-agent: battery result (mirror 24.7 vs ablation 7.7, no spike
  criterion met anywhere) CONTRADICTS the same-day acceptance run (mirror
  18.3 vs 34.3, spikes [1,1,1,1]).** Root cause found: the battery config
  omits the acceptance script's calibration — `pose_scale 5.0` and 150
  sleep grad steps — so the pose head never gets accurate enough for the
  spike criterion (max of threshold and baseline-q99) to be crossable. The
  acceptance run is the valid one; the battery's kidnapped config needs to
  be brought up to it (follow-up 2).

## Three highest-value follow-ups

1. **Rerun the battery at `--scale full` with genuinely converged
   baselines** (capability-shift's premise, ghost-attribution's and the
   forecaster's demonstrated need). This is the single change that converts
   six "not evidence" rows into evidence. Overnight on the dev laptop per
   docs/training.md's wall-clock table.
2. **Align the battery's kidnapped config with the acceptance calibration**
   (`pose_scale`, `sleep_grad_steps`, and log baseline-q99 vs teleport
   divergence per run so a missed spike is diagnosable from the CSV alone),
   then re-measure whether the acceptance-scale relocalization win
   (18 vs 34 ticks) replicates across 5 seeds.
3. **Training-scale curves for the two Ledger heads that flipped between
   scales**: attribution accuracy and forecaster NMSE vs training ticks
   (20k → 200k, ~5 points). This pins each head's minimum viable scale, so
   future batteries stop producing known-undertrained numbers, and it is
   cheap (reuses existing runners, no new code).

## Scale disclaimer

Every number above was produced at `--scale quick`, seeds=3, on CPU —
smoke-level evidence for machinery, below minimum viable scale for
architecture claims. The summary table stamps this; this file repeats it
because tables get quoted without their headers.
