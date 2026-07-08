# The Internal Researcher (Stage 8)

The researcher layer turns run diagnostics into a **prioritized research
agenda** — a ranked list of what the agent does not understand and which
experiment would reduce that uncertainty most efficiently. Its output is
data for a human to read. It never acts, never executes experiments,
never modifies configuration.

## Structural rules (enforced by tests)

All Stage-7 proposal-layer rules apply verbatim to `researcher/`
(tests/test_proposals_no_execution.py):

- **Data, never code.** Nothing under `researcher/` may execute, import
  execution machinery, or spawn processes. A human implements everything.
- **Import bans.** `researcher/` may not import `world`, `training`,
  `agent`, `memory`, or `torch`. `ledger.competence` and the proposal
  schema are the only project imports. Anything needing the live model
  lives outside the package and is *injected* (see EIG v2 below).
- **Confined writes**: `runs/<id>/researcher/` only.
- **SCOPE RULE** (CLAUDE.md hard rule): hypotheses and questions may be
  about the agent-in-the-world and its own models ONLY. Statements
  matching reviewer / reset-schedule / training-infrastructure patterns
  are rejected in `Hypothesis.validate()` / `Question.validate()` at the
  schema level, and adversarial fixtures test the rejection
  (tests/test_registry.py).

## Hypothesis schema (researcher/registry.py)

Versioned dataclass records:

| field | meaning |
|---|---|
| `schema_version` | currently 1; loaders reject other versions |
| `statement_template` + `params` | e.g. `"success rate of action {action} is stable"` |
| `scope` | `world` \| `self_capability` \| `self_memory` \| `self_model` |
| `monitor` | `{metric_query, statistical_test, threshold, window, min_samples, arm_after?, std_floor?}` |
| `status` | `supported -> weakening -> contradicted` (+ `retired`) |
| `transitions` | full history, each with tick + evidence string + statistic |

Monitors run every sleep phase over the run's JSONL/TB store
(`registry.check_all`). Status machine: one violating check moves
supported→weakening; a second **consecutive** violation escalates to
contradicted (sticky); one clean check recovers weakening→supported.
Every transition is appended to `runs/<id>/researcher/contradiction_events.jsonl`
(consumed by the agenda and the dashboard).

### Monitor templates (auto-populated)

- **Region stationarity** (per 8x8 region): KS two-sample statistic
  between consecutive windows of regrowth intervals.
- **Capability stability** (per action): mean shift of success rate in
  prev-window std units, with `std_floor` (a constant all-success window
  must not divide one stray failure into an astronomical shift) and
  `arm_after` (behavior-coupled monitors arm only after early policy
  settling; both comparison windows must lie past the boundary).
- **Memory-decay validity / forecaster validity / calibration
  stability**: band tests on the corresponding TB tags.

Both windows are bounded above by the check tick: a post-hoc replay's
store holds the whole run, and an unbounded now-window would read the
future (this bug produced impossible pre-lever detections in the
stage-8a acceptance before it was fixed).

**Known limitation — mean_shift is an onset detector.** It violates only
while the now-window straddles the change and the prev-window is still
clean; once both windows are post-onset the statistic returns to ~0
(and while the prev-window straddles the onset, its inflated variance
suppresses the statistic). Escalation to `contradicted` therefore needs
two consecutive checks inside that transient detectability window —
check cadence, window and `arm_after` must be sized so the window spans
at least two checks, or a real change is seen once (weakening) and then
"recovers". The researcher-value battery hit exactly this at quick
scale (every seed censored) before its geometry was fixed.

**The stage-B upgrade — `cusum_frozen_baseline` (default for capability
templates).** After `arm_after`, the first armed check freezes a
baseline (mean, std with `std_floor`); each later check standardizes its
window mean against that frozen baseline and accumulates the one-sided
CUSUM statistic `S_t = max(0, S_{t-1} + (|z_t| - k_drift))`, violating
when `S_t > h_threshold` (`researcher.monitors` in configs/base.yaml:
`cusum_k_drift 0.5`, `cusum_h_threshold 5.0`; detector state persists in
`Hypothesis.monitor_state`). Because the baseline never slides, a
persistent change keeps paying into S regardless of when checks land —
there is no transient detectability window to miss. The status machine
is unchanged (one violation weakens, a second consecutive one
contradicts, sticky, one clean check recovers), but with CUSUM the
second violation is nearly automatic once S crosses h, so escalation
semantics read **"confirmed persistent"** rather than "caught twice in a
transient window". `mean_shift` remains selectable per template
(`researcher.monitors.capability_test`) as the onset-detector ablation.
Both windows stay bounded above by the check tick on every path (the
stage-8a future-read fix, regression-tested for CUSUM too).

Human hypotheses go in `researcher/hypotheses.yaml` (same schema, same
scope validation).

## Questions (researcher/questions.py)

Four types, each with evidence refs and candidate experiments from the
fixed menu `{probe_action_batch: 1.0, directed_visit: 2.0,
targeted_replay: 1.5, run_battery: 4.0}` (costs in budget units):

- `world_uncertainty` — top-k epistemic-map cells
- `capability_gap` — actions whose success rate shifted between recent
  windows (std-floored)
- `assumption_violation` — one per weakening/contradicted hypothesis
- `model_misfit` — forecaster horizons with NMSE >= 2 (worse than the
  identity predictor by construction)

Pending Stage-7 proposals enter the **same agenda** as a second
candidate stream.

## Ranking

### v1 (researcher/agenda.py) — heuristic

```
score = value x tractability x novelty_decay / cost
```

- value: normalized uncertainty (questions) or
  |magnitude_estimate| x confidence (proposals)
- tractability: learning progress in the relevant region from the
  competence report, squashed to (0.05, 1]. **Noisy-TV guard:** zero or
  negative progress floors tractability at 0.05 — an irreducibly random
  region maxes out disagreement forever and teaches nothing, so high
  uncertainty alone must never buy the top slot.
- novelty_decay: 1 / (1 + recent near-duplicate executions)
- cost: from the experiment menu / proposal estimate

Deterministic on a frozen store (sort key `(-score, id)`, tested).

### v2 (researcher/eig.py) — expected information gain

Selected by `researcher.ranker` in config (`v1` kept as the ablation).
v2 replaces the *value* term with an EIG estimate where one is
computable and records `predicted_gain` per item:

- **Directed visits** (world_uncertainty): Plan2Explore-style. The world
  model is injected as a duck-typed adapter
  (`experiments/model_adapter.py` — the torch side of the seam;
  `researcher/` stays import-clean). `EIG = region_disagreement x
  imagined_visit_reduction`, where the reduction is the *reducible
  fraction* `epistemic / (epistemic + aleatoric)` measured along
  imagined RSSM rollouts from replay states inside the region under a
  uniform random policy. Approximation: this is a first-order
  learnability estimate, not a simulated training update.
- **Probe batches** (capability_gap): expected posterior-variance (~
  Brier) reduction under a Beta(successes+1, failures+1) posterior on
  the action's success rate: `V = ab/((a+b)^2(a+b+1))`; after n more
  labels `V' ~= V (a+b+1)/(a+b+n+1)`; `EIG = V - V'`. Ignores drift
  during the probe batch.
- Everything else falls back to the v1 value term.

The noisy-TV guard **survives v2**: EIG is still multiplied by the
tractability term, so a big predicted gain the model cannot realize
(max disagreement, zero learning progress) stays floored.

`predicted_gain` is logged per executed item so the researcher-value
battery can score EIG calibration against realized reductions
(Section 21 metrics).

## Outputs

Per sleep phase, under `runs/<id>/researcher/`:

- `agenda_<tick>.json` — every item with score decomposition,
  hypothesis links, predicted gain
- `research_agenda.md` — top 10, human-readable, each with the question,
  the proposed experiment, the score decomposition, and which hypothesis
  a result would move
- `hypotheses/*.json`, `contradiction_events.jsonl`

## Three-arm value battery (P9.4, experiments/batteries/researcher_value.py)

Does following the agenda actually buy anything? Three arms under a
fixed experiment budget, N >= 5 seeds:

1. **agenda-following** — execute the top-ranked items
2. **uniform-random** — same budget, random menu items
3. **greedy raw-uncertainty** — always the highest-disagreement region
   (no tractability term; the noisy-TV control)

Metrics (Section 21, experiments/metrics.py): uncertainty reduction per
item (against a no-intervention control run's drift), competence gain
per item, contradiction-detection latency (censored at 50k),
EIG calibration (predicted vs realized, Spearman), agenda stability
(Kendall tau between consecutive agendas — descriptive only, stability
is not a virtue by itself). Nulls are reported with the same prominence
as positives.

## Acceptance evidence

- stage-8a: docs/acceptance/stage-8a — lever-driven
  supported→weakening→contradicted with correct timing, no pre-lever
  false alarms
- stage-8b: docs/acceptance/stage-8b — top agenda items point at the
  lever-affected capability; deterministic ranking
- stage-8c: docs/acceptance/stage-8c — v2 rescores with finite gains,
  differs from v1, never promotes the noisy-TV region
