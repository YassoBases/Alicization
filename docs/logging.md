# Log schema

Two sinks per run, both under `runs/<timestamp>/`:

1. **JSONL event log** — one record per tick, written by
   `training.loggers.JsonlRunLogger`.
2. **TensorBoard scalars** — written by `training.loggers.TBLogger`
   (scalars only for now).

## JSONL event log

Files rotate every 100,000 ticks (`rotate_every`), named
`events-<chunk:09d>.jsonl` with `chunk = tick // rotate_every`.
One JSON object per line:

| field     | type          | description                                              |
|-----------|---------------|----------------------------------------------------------|
| `tick`    | int           | world tick after the step (first record has tick 1)      |
| `pos`     | [int, int]    | agent position [x, y] after the step                     |
| `action`  | int           | action id 0..8 (see action table in world/engine.py)     |
| `success` | bool          | realized success flag from the engine                    |
| `intero`  | [float x 6]   | [energy, fatigue, memory_pressure, sin(tod), cos(tod), 1]|
| `reward`  | float         | per-tick reward (0.0 until drives are implemented)       |
| `events`  | list, optional| ground-truth world events this tick (omitted when empty) |

### Event records (`events[]`)

Every entry has at least `tick`, `type`, `cause`. `cause` is the ground-truth
attribution label (`"self"` or `"world"`); it exists for evaluation only and
must never enter observations or losses.

| type                     | cause | extra fields                              |
|--------------------------|-------|-------------------------------------------|
| `agent_moved`            | self/world | `pos`, `agent`, `dpos`               |
| `food_consumed`          | self/world | `pos`, `agent` (null if world)       |
| `food_regrown`           | world | `pos`                                      |
| `food_relocated`         | world | `pos` (destination), `src`, `had_food`     |
| `seasonal_shift`         | world | `patches_moved`                            |
| `mark_placed`            | self  | `pos`, `agent`                             |
| `mark_erased`            | self  | `pos`, `agent`                             |
| `capability_shift_start` | world | `action`, `fail_prob`, `energy_mult`       |
| `capability_shift_end`   | world | `action`                                   |
| `exogenous_reset`        | world | —                                          |

## TensorBoard scalars

Tag convention: `<group>/<name>`. Every scalar written by either trainer is
listed here; keep this table in sync with training/ppo.py and
training/sleep.py (grep for `"` + `/` in those files to audit).

Step axis: `PPOTrainer` logs against `global_step` (env ticks across all
envs); `CircadianTrainer` logs against `env_steps` (same unit).

### Task / PPO (`PPOTrainer.train`)

| tag | meaning |
|-----|---------|
| `reward/rollout` | sum of per-tick rewards over a rollout, mean across envs |
| `loss/policy` | PPO clipped surrogate |
| `loss/value` | (clipped) value MSE |
| `loss/total` | policy + value + entropy (+ world-model when core=rssm) |
| `entropy` | mean policy entropy |
| `approx_kl` | mean approximate KL(old, new) per update |
| `clip_frac` | fraction of ratios clipped |
| `sps` | env ticks per wall-clock second |

### World model (core=rssm; PPO logs `rssm/*` per update, sleep logs `sleep/*` per window)

| tag | meaning |
|-----|---------|
| `rssm/recon` | grid + intero reconstruction MSE |
| `rssm/kl` | KL-balanced prior/posterior loss (free-nats floored) |
| `rssm/ensemble_nll` | dynamics-ensemble next-embed NLL |
| `rssm/pose_mse` | allocentric pose head MSE (stage-6a) |
| `rssm/reward_mse` | reward head MSE (post-action-state aligned) |
| `rssm/epistemic` | mean ensemble disagreement over the rollout |
| `rssm/aleatoric` | mean predicted variance over the rollout |
| `rssm/epistemic_map_mean` | mean of the position-bucketed epistemic map (visited cells) |
| `rssm/participation_ratio` | deter-state PR (collapse detector; WARNING < 25% of running max) |
| `sleep/wm_total`, `sleep/recon`, `sleep/kl`, `sleep/pose_mse` | world-model loss terms on replay |
| `sleep/actor`, `sleep/critic` | imagination REINFORCE / critic MSE |
| `sleep/imagined_reward` | mean predicted reward over imagined trajectories |
| `sleep/grad_steps` | consolidation steps actually run this window |
| `sleep/memory_pruned` | episodic entries dropped by sleep pruning |
| `phase/sleep` | 1.0 at env-steps where a sleep window ran, else 0.0 (wake/sleep marker) |

### Ledger

| tag | meaning |
|-----|---------|
| `ledger/body_nll` (+`_ema`) | body-model dpos CE |
| `ledger/success_bce`, `ledger/success_brier` (+`_ema`) | success head BCE / Brier |
| `ledger/denergy_mse`, `ledger/denergy_mae` (+`_ema`) | denergy head regression |
| `ledger/attribution_loss` | attribution classifier CE (self-supervised labels) |
| `ledger/attribution_accuracy` (+`_ema`) | vs ground-truth cause labels (evaluation only) |
| `ledger/reliability_bce` | reliability logistic BCE on verification queue |
| `ledger/reliability_ece` | 10-bin expected calibration error over the queue |
| `ledger/reliability_verifications` | cumulative revisit verifications |
| `sleep/forecaster_nll` | forecaster Gaussian NLL (trained in sleep) |
| `sleep/forecaster_nmse_k<k>` | NMSE vs identity predictor per horizon k |

### Episodic memory / mirror

| tag | meaning |
|-----|---------|
| `memory/pressure` | mean fill fraction across env stores |
| `memory/write_rate` | gate's realized write-rate EMA (target `memory.write_rate_target`) |
| `memory/gate_threshold` | current surprise threshold |
| `memory/trips`, `memory/stale_trips`, `memory/stale_trip_rate_per_1k` | memory-guided foraging trips (arbiter mode) |
| `mirror/divergence`, `mirror/divergence_max` | decoder-vs-body implied self-state distance (cells); MONITORED, never a loss |
| `mirror/triggers` | cumulative probe/MPC trigger count |

### Text annotations

`levers/events` — one text record per lever event at its step:
`capability_shift_start/end`, `seasonal_shift`, `exogenous_reset`
(per-patch `food_relocated` is deliberately excluded — hundreds per run
drown the TB text tab; find those in the JSONL event log instead).
