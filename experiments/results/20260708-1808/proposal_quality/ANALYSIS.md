# Proposal-quality battery — analysis (quick scale, 3 seeds)

**Hypothesis under test:** ledger-sourced proposals outperform
logs-only-sourced proposals on realized benefit and calibration.

**Verdict: not decidable from this run — the logs-only arm produced zero
proposals.** What the run does establish, and its validity limits, below.

## 1. The control arm was silent (the main finding)

Both generator variants ran on every review cycle (30 ledger + 30
logs-only generator invocations per seed, per the decision log). Every
one of the 11 proposals that fired came from the ledger variant; the
logs-only variant fired **zero** and was suppressed or below-trigger on
all 90 invocations. On healthy smoke-scale runs (reward climbing to ~+5),
raw logs simply contain nothing the logs-only rules flag: no NaNs, no
reward collapse, no checkpoint gaps. The Ledger's self-model surfaces
triggers (KL at the free-nats floor, replay-coverage imbalance, region
competence dips) that are invisible in raw logs at this scale.

That is itself evidence for the *sensitivity* half of the hypothesis
(the ledger sees more), but with n_logs_only = 0 the benefit/calibration
comparison the battery exists for is undefined. A longer or deliberately
unhealthy run (injected capability_shift, LR spike) is needed to make
the control arm speak.

## 2. What the evaluated ledger proposals show — with a validity caveat

3 of 11 proposals carried a machine-readable knob (`rssm.free_nats/2`)
and were A/B-evaluated (seeded control vs treated, 10,240-step eval
runs): benefit +72.97 ± 26.59 control-sd on their success metric, hit
rate 3/3.

**Caveat, stated plainly: this hit rate is partly mechanical.** The
free_nats proposal's success criterion is the KL metric itself, and
halving free_nats lowers the KL clamp by construction. The A/B measures
a real, causal config effect (the machinery works end-to-end), but
"KL went down after lowering the KL floor" is close to tautological —
it does not show the world model got *better*. Future batteries should
score knob proposals against an independent metric (wm loss, reward)
alongside the proposal's own criterion.

The other 8 proposals (training_schedule, memory_policy, retraining)
carry no machine-readable knob; nothing was executed, so they are marked
`evaluation=not_executed` and **excluded** from benefit stats. A pre/post
around their creation tick would measure the life run's natural drift,
not causation (this exact mistake produced a meaningless −1.74 "benefit"
in an earlier draft of this battery and was removed).

## 3. Calibration

ECE 0.450 over 1 occupied bin (all three A/B proposals had confidence
0.55, all hit): the generator is *underconfident* on its free_nats
proposals — though see the caveat above about how easy that criterion
is. With n=3 in one bin this is anecdote, not calibration measurement;
Section 17's ECE needs ≥ ~50 evaluated proposals to mean anything.

## 4. Quality vs lifetime

Undefined here: all three A/B-evaluated proposals were created at the
same review tick (8192), so Pearson r(created_tick, benefit) has no
variance in x. The not-executed majority spans ticks 8192–24576 but has
no benefit values by construction.

## 5. Scale disclaimer & provenance

- quick scale (24,576-step lives, review every 8,192, 10,240-step
  evals), 3 seeds, CPU. The full protocol (500k lives, N≥5 seeds,
  scheduled human-style blind review with usefulness ratings) remains
  future work; every number above should be read as a plumbing
  validation, not a scientific result.
- The scripted reviewer approves everything on schedule and never sees
  `source` (blind by construction); acceptance rate 1.0 and usefulness
  ratings n/a are artifacts of that scripting.
- An earlier run of this battery (results/20260708-1745, deleted)
  reported benefits of ~5e11 sd: eval runs were so short that control
  series had one point and std ≈ 0. Fixed by a degenerate-control guard
  in `realized_benefit_ab` (NaN, never astronomical) plus longer eval
  runs; the guard is unit-tested.
