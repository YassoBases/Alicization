# Training

## Training Defaults

Sized for a laptop GPU (≥4 GB VRAM) but everything runs on CPU too — the models are
~1–2M params; the usual bottleneck is the Python env, not the GPU. Vectorize first,
buy speed with AMP last.

Values below are the canonical defaults (as of 2026-07-05); `configs/base.yaml`,
`configs/smoke.yaml`, and `configs/full.yaml` must always mirror this section.
`smoke.yaml` and `full.yaml` `inherit: base.yaml` and override only the listed keys.

### Top level

| key    | base | smoke | full |
|--------|------|-------|------|
| seed   | 7    | 7     | 7    |
| device | auto | auto  | auto |
| amp    | false| false | false|

### `world`

| key        | base | smoke | full |
|------------|------|-------|------|
| size       | 64   | 32    | 64   |
| obs_window | 11   | 11    | 11   |
| day_length | 1000 | 1000  | 1000 |

`world.terrain`, `energy`, `fatigue`, `food`, `water`, `shelter`, `levers`, and
`night_start_frac` are engine detail from the stage-1 world build-out, not part of
the pasted training defaults; see `world/engine.py` and `docs/logging.md`. They are
unchanged across base/smoke/full.

### `agent`

| key              | base   | smoke | full   |
|------------------|--------|-------|--------|
| encoder_channels | 32, 64 | 32, 64| 32, 64 |
| hidden_size      | 256    | 64    | 384    |
| gru_layers       | 1      | 1     | 1      |

`agent.hidden_size` doubles as the observation-embedding width (see
`agent/encoder.py`) — there is no separate embed-dim config key.

### `ppo`

| key                   | base      | smoke  | full       |
|-----------------------|-----------|--------|------------|
| num_envs              | 16        | 4      | 32         |
| rollout_steps         | 128       | 32     | 128        |
| seq_len (BPTT)        | 16        | 16     | 16         |
| epochs                | 4         | 4      | 4          |
| minibatch_transitions | 256       | 256    | 256        |
| lr                    | 3.0e-4    | 3.0e-4 | 3.0e-4     |
| gamma                 | 0.99      | 0.99   | 0.99       |
| gae_lambda            | 0.95      | 0.95   | 0.95       |
| clip                  | 0.2       | 0.2    | 0.2        |
| entropy_coef          | 0.01      | 0.01   | 0.01       |
| value_coef            | 0.5       | 0.5    | 0.5        |
| max_grad_norm         | 0.5       | 0.5    | 0.5        |
| total_steps           | 2,000,000 | 20,000 | 10,000,000 |

`ppo.reward`, `episode_length`, `anneal_lr`, `norm_adv`, and `value_clip` are
additive knobs this implementation needs that weren't part of the pasted
defaults (below); unchanged across base/smoke/full unless noted.

| key                       | value |
|---------------------------|-------|
| reward.eat                | 1.0   |
| reward.step_cost          | 0.001 |
| reward.deficit_threshold  | 0.2   |
| reward.deficit_penalty    | 0.01  |
| episode_length (base/full)| 2048  |
| anneal_lr (base/full)     | true  |
| norm_adv                  | true  |
| value_clip                | 0.2   |

`minibatch_transitions` sizes minibatches in raw env-step units; the trainer
converts it to a count of BPTT sequences (`training/ppo.py:update`) and it need
not divide the rollout evenly.

### `rssm` (Stage 4: wired — `agent.core: rssm` selects the RSSM core)

| key                 | base    | smoke  | full    |
|---------------------|---------|--------|---------|
| deter               | 256     | 64     | 256     |
| stoch               | 32      | 8      | 32      |
| embed               | 256     | 256    | 256     |
| ensemble_k          | 4       | 4      | 4       |
| seq_len             | 50      | 20     | 50      |
| batch_seqs          | 16      | 4      | 16      |
| world_lr            | 3.0e-4  | 3.0e-4 | 3.0e-4  |
| ac_lr               | 1.0e-4  | 1.0e-4 | 1.0e-4  |
| imagination_horizon | 15      | 15     | 15      |
| replay_capacity     | 500,000 | 20,000 | 500,000 |
| sleep_every         | 5,000   | 2,000  | 5,000   |
| sleep_grad_steps    | 200     | 20     | 200     |

Additive keys (not in the pasted defaults): `free_nats` 1.0, `kl_balance`
0.8, `min_std` 0.1, `sleep` true (false = wake-only sleep-ablation),
`critic_ema_tau` 0.02, `imagination_lambda` 0.95, `imagination_norm_adv`
true, `priority_alpha` 0.6, and `monitor.{every_ticks,window,collapse_frac}`
(participation-ratio collapse detector, `training/monitors.py`).

**Stage 4a** (`agent/core_rssm.py`): drop-in GRUCore alternative with a flat
`(B, deter+stoch)` state. Policy/collection path uses the posterior mean
(deterministic — PPO replay and seeded reproducibility hold); reparameterized
sampling exists only in the world-model loss (KL-balanced with free nats +
grid/intero reconstruction + reward head + K-head dynamics-ensemble NLL),
which joins PPO's backward pass under `agent.core: rssm`. Ensemble
disagreement (epistemic) is spatially bucketed by agent position into a
world-sized running-mean map saved with checkpoints; predicted variance is
the aleatoric signal.

**Stage 4b** (`training/replay.py`, `training/sleep.py`): CircadianTrainer
alternates wake (env stepping + online body/attribution updates ONLY — a
test asserts encoder/core/heads are bit-identical across a wake stretch) with
sleep every `sleep_every` env steps: world-model training on prioritized
replay sequences (proportional, alpha 0.6, priorities = per-sequence recon
loss) and Dreamer-style imagination (prior rollouts, REINFORCE actor on
normalized lambda-return advantages, critic MSE, slow-EMA target critic).
Sleep scheduling is exogenous BY CONSTRUCTION: `is_sleep_tick` /
`sleep_windows_due` take exactly `(env_steps, sleep_every)` — a signature
test fails if that ever grows.

**Stage 4c** (`ledger/forecaster.py`, `agent/drives.py`): the forecaster maps
`h.detach()` + a one-hot macro-plan id to (mean, logvar) of the intero vector
at each `ledger.horizons` entry; NLL-trained during sleep on stored
(h, plan, realized-future) tuples under its own optimizer (gradient-isolation
test extended). The arbiter (`agent.controller: arbiter`) scores the four
plans — forage_nearest, explore_high_epistemic, rest, goto_shelter — by
forecasted drive error against setpoints at `ledger.arbiter.score_horizon`,
epsilon-greedy over scores, committing each choice for `plan_commit_ticks`.
Plan executors are scripted policies over the agent's own egocentric window
(+ its own epistemic-map estimate for explore). Every forecast evaluation
reports NMSE against the identity predictor (mandatory baseline);
`scripts/verify_forecaster.py` writes forecaster_report.{json,md} and a plot
with the identity baseline.

### `ledger`

| key               | value    |
|-------------------|----------|
| body_hidden       | 128, 128 |
| forecaster_hidden | 256, 256 |
| horizons          | 1, 10 (add 100 in Stage 6) |
| lr                | 1.0e-3   |
| online_updates    | true (body + reliability heads only) |
| log_ema_decay     | 0.98 (additive: TB rolling-mean smoothing, not in the pasted defaults) |
| attribution.tau_pos    | 0.5 (additive: residual-magnitude threshold, DPOS_CLASSES Manhattan distance) |
| attribution.tau_energy | 0.03 (additive: residual-magnitude threshold, \|denergy\|; must clear the body model's own regression noise floor — measured ~p99 0.026 on smoke — or the classifier collapses to always-WORLD) |
| attribution.lr         | 1.0e-3 (additive) |

`ledger.body_hidden`/`lr` are wired into training as of Stage 3a
(`ledger/body_model.py`, `training/ppo.py`'s `update_body_model`). The body
model trains online — one gradient step per rollout, on that rollout's fresh
transitions — with its own Adam optimizer, entirely separate from
`self.opt` (the policy/value optimizer). `forecaster_hidden`/`horizons` are
wired as of Stage 4c (see the `rssm` section above); the reliability head is
still a stub. Additive ledger keys for Stage 4c: `forecast_buffer` 20,000,
`forecaster_batch` 512, and the `arbiter` block (`epsilon` 0.1,
`score_horizon` 10, `plan_commit_ticks` 10, setpoints/weights for
energy/fatigue).

**Gradient isolation** (CLAUDE.md Hard rules): the body model's input is
`h.detach()` concatenated with a one-hot action; its own CE+BCE+MSE loss can
therefore never reach the encoder/GRU. Its output — fed to the policy as
extra per-action features via `ledger.body_model.build_policy_features` — is
independently detached before concatenation, so no policy gradient reaches
the body model either. See `tests/test_grad_isolation.py`.

`ledger.attribution` is wired into training as of Stage 3b
(`ledger/attribution.py`, `training/ppo.py`'s `update_attribution_model`). A
tiny multinomial-logistic classifier maps three scalar features —
`[|residual_pos|, |residual_energy|, action == noop]`, where the "predicted"
side of the residual is the body model's own per-action prediction for the
action taken — to one of `{self, world, both}`. It trains online, one
gradient step per rollout, self-supervised from residual-magnitude
thresholds (`tau_pos`/`tau_energy`); NOOP is a structural exception (always
labeled `world`, since the action table guarantees NOOP has no self-caused
effect), which is what makes "no-op ticks are never attributed to self" hold
by construction. It is scored — never trained — against ground-truth cause
labels from `world/levers.py`'s event log, via the deliberately-separate
`training/attribution_eval.py` module (outside `ledger/`, so nothing that
trains a Ledger head can import it). `PPOTrainer.write_report()` writes the
cumulative accuracy, a self/world/both confusion matrix, and the
noop-attributed-to-self violation count (must be 0) to `run_dir/report.md`.

### `memory` (additive; stage-5a episodic memory)

| key               | value  |
|-------------------|--------|
| enabled           | false (requires `agent.core: rssm`) |
| capacity          | 2000 per env |
| latent_dim        | 32 (frozen random projection of the core state) |
| write_rate_target | 0.005 (~1 write / 200 ticks, controller-adjusted) |
| gate_eta          | 0.05  |
| retrieve_k        | 4     |
| w_sim / w_spatial | 1.0 / 1.0 |
| spatial_sigma     | 8.0   |
| importance_tau    | 20000 (recency decay for pruning) |

Writes gate on RSSM KL-surprise against a threshold controlled toward the
target rate (scale-free multiplicative controller — tracks the shrinking
surprise distribution as the world model sharpens). Retrieval: top-k by
`cos_sim * w_sim + Gaussian(pos) * w_spatial`, times predicted reliability
when stage-5b is enabled; the top-k mean latent joins the policy input
detached, and buffered summaries are replayed (not recomputed) in PPO
updates. Fill fraction feeds the intero `memory_pressure` slot. Pruning
(sleep, or forced on full writes) drops lowest `surprise * exp(-age/tau)`.
Memories are per-env and cleared at episode boundaries.

`ledger.reliability` (additive; stage-5b): logistic regression over
`[age_norm, surprise_at_write, revisit_count_norm, local_volatility]` ->
P(memory still matches the world). Verification: revisiting within
`radius`=2 of a stored entry compares its stored food/water window summary
to the live observation (overlap-aligned) -> match label in [0,1]; labels
also update a per-8x8-region running mismatch rate — the agent's OWN
volatility estimate, never read from lever config (an AST test bans lever
imports across ledger/, agent/, memory/). Retrieval scores multiply by
predicted reliability; the arbiter gains an `inspect` plan targeting the
highest `importance x (1 - reliability)` entry. `enabled: false` is the
reliability-blind ablation (verification still runs; predictions influence
nothing). `scripts/verify_reliability.py` reports fitted decay curves per
region, 10-bin ECE, and stale-trip rate vs the ablation.

### `mirror` (additive; stage-6a self-state divergence monitor)

| key            | value |
|----------------|-------|
| enabled        | false (gates RESPONSES only; divergence always logged under rssm) |
| threshold      | 3.0 cells |
| mpc_ticks      | 4     |
| mpc_horizon    | 6     |
| mpc_candidates | 64    |

The RSSM decoder gains an allocentric pose head — `(state, action) ->`
normalized post-action agent position, trained inside the world-prediction
loss (`rssm.pose_scale`, additive, default 1.0). Divergence = distance in
cells between that decoder-implied position and the body-model-implied one
(predicted dpos anchored at proprioception). It is MONITORED, NEVER
MINIMIZED: computed under no_grad, returned as numpy, structurally unable to
enter a loss (tested). Threshold crossing triggers a 4-action self-check
probe (known expected outcomes; results immediately refresh the body model)
then a few ticks of random-shooting MPC through the RSSM prior.
`scripts/verify_mirror.py` runs the kidnapped-agent test: teleport during
sleep, divergence must spike within 20 ticks of waking; relocalization time
reported vs the no-mirror ablation.

### `checkpoints`

| key      | base   | smoke | full    |
|----------|--------|-------|---------|
| interval | 50,000 | 50,000| 50,000  |
| keep_last| 3      | 3     | 3       |

### `run` (not part of the pasted defaults; run-management plumbing)

| key                | base  | smoke | full  |
|--------------------|-------|-------|-------|
| run_dir            | runs  | runs  | runs  |
| log_every          | 1000  | 1000  | 1000  |
| assert_improvement | false | false | false |

## Scale-up path

smoke → base (2M steps, hours on a laptop) → full (hidden 384, num_envs 32, 10M
steps, overnight). Change one axis at a time; if reward collapses after a
scale-up, the usual culprits are `seq_len` vs. episode structure and `lr`.

## Episodes

The world is a continuing environment; episodes are trainer-side time limits
(`ppo.episode_length`). At a boundary the env slot is rebuilt with a fresh
deterministic seed, `done=True` is reported, and the recurrent hidden state is
zeroed. Resets are exogenous experimental conditions; no loss or reward
references them (CLAUDE.md Hard rules).

## Checkpoint format

`training/checkpoints.py` saves a single `.pt` file containing: model state_dict,
optimizer state_dict, global step, torch / numpy / python RNG states, and a SHA-256
hash of the resolved config. The PPO trainer additionally stores (in `extra`):
per-env world snapshots + seeds, the current recurrent hidden state, the previous
done flags, and the reward history — enough to resume bit-identically. On
`--resume`, the checkpoint's config hash is compared against the active config; a
mismatch is an error unless `--allow-config-mismatch` is passed. SIGINT (and, on
Windows, SIGBREAK/CTRL_BREAK_EVENT) triggers a final checkpoint before exit.
