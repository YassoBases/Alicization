# Stage-8a acceptance: hypothesis registry + contradiction monitors

**Criterion:** a world-config lever violating a registered capability
hypothesis drives `supported -> weakening -> contradicted` with correct
timing against the lever log; no pre-lever contradictions anywhere.

**Fixture:** smoke-scale circadian run, `capability_shift` on MOVE_E
(action 2, fail_prob 0.9) starting at world tick 4500 — unannounced, from
world config. **Frozen policy** (ppo.lr = 0, rssm world/ac lr = 0): the
acceptance isolates monitor-vs-lever timing, so the lever must be the only
nonstationarity; a learning policy genuinely changes success rates while
its position distribution settles (a correct detection, but a confounded
demonstration).

**Result** (`verify_registry_replay.txt`):

- lever fired at world tick 4501 (read from the JSONL event log)
- MOVE_E `hyp-capability-success-2`: supported -> weakening @4750
  (7.4 sd) -> **contradicted @5000** (10.7 sd), threshold 3.0
- 0 pre-lever weakening wobbles; 0 pre-lever contradictions on any action
- one post-lever downstream detection: MOVE_W (action 3) contradicted
  @5250. Legitimate, not a false alarm: with MOVE_E failing 90% the
  (frozen-weights) agent's position distribution piles against the west
  edge and MOVE_W's blocked rate genuinely changes. Reported, not tuned
  away.
- contradiction_events.jsonl written per transition (consumed by the
  agenda + dashboard).

**Bugs this acceptance caught** (fixed in researcher/registry.py, kept
here for the record):

1. **Future leakage in replay**: the "now" window was unbounded above, so
   post-hoc checks before the lever read post-lever data and "detected"
   the shift early (impossible 10-sd pre-lever shifts on a frozen policy).
   Both windows are now bounded by `now_tick`.
2. **Degenerate std**: a constant all-success baseline window has std ~ 0
   and one stray sample divided to an astronomical shift; `std_floor`
   (default 0.05) sets the minimum meaningful unit of change.
3. **arm_after**: behavior-coupled monitors arm only after early policy
   settling, and both comparison windows must lie past the boundary.

Monitor geometry at smoke scale (script comment has the rationale):
window 1000, check cadence 250, arm_after 1500 — the env-0 tick axis is
`env_steps / num_envs` long, and contradiction requires two consecutive
violating checks to straddle the change.

Reproduce: `python scripts/verify_registry.py --seed 0` (or `--run-dir`
to replay an existing fixture).
