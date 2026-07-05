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

### `rssm` (Stage 4 — not yet wired into training)

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
`self.opt` (the policy/value optimizer). `forecaster_hidden`, `horizons`, and
`online_updates` are not yet wired (reliability/forecaster heads are still
stubs).

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
