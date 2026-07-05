# Training

> **PLACEHOLDER VALUES.** The Training Defaults below are provisional and were not
> supplied by the project owner yet. When the canonical values are provided, update
> this table and regenerate `configs/base.yaml` / `configs/smoke.yaml` /
> `configs/full.yaml` from it. Configs must always mirror this section.

## Training Defaults

### Algorithm (recurrent PPO)

| key                | base    | smoke  | full      |
|--------------------|---------|--------|-----------|
| total_env_steps    | 5000000 | 40000  | 50000000  |
| num_envs           | 8       | 2      | 16        |
| rollout_length     | 256     | 64     | 256       |
| seq_len (BPTT)     | 32      | 16     | 32        |
| episode_length     | 2048    | 256    | 2048      |
| lr                 | 3.0e-4  | 2.0e-3 | 3.0e-4    |
| anneal_lr          | true    | false  | true      |
| gamma              | 0.99    | 0.99   | 0.997     |
| gae_lambda         | 0.95    | 0.95   | 0.95      |
| clip_range         | 0.2     | 0.2    | 0.2       |
| update_epochs      | 4       | 2      | 4         |
| num_minibatches    | 8       | 2      | 8         |
| entropy_coef       | 0.01    | 0.005  | 0.01      |
| value_coef         | 0.5     | 0.5    | 0.5       |
| value_clip         | 0.2     | 0.2    | 0.2       |
| max_grad_norm      | 0.5     | 0.5    | 0.5       |
| norm_adv           | true    | true   | true      |
| amp_bf16           | false   | false  | false     |

### Baseline reward (training/reward.py, `ppo.reward`)

| key               | value |
|-------------------|-------|
| eat               | 1.0   |
| step_cost         | 0.001 |
| deficit_threshold | 0.2   |
| deficit_penalty   | 0.01  |

### Model

| key                 | base  | smoke | full  |
|---------------------|-------|-------|-------|
| core                | gru   | gru   | gru   |
| core_hidden         | 256   | 64    | 512   |
| obs_embed_dim       | 256   | 32    | 256   |
| encoder_channels    | 32,64 | 8,16  | 32,64 |

### Run management

| key                | base    | smoke | full    |
|--------------------|---------|-------|---------|
| seed               | 0       | 0     | 0       |
| device             | auto    | auto  | auto    |
| checkpoint_every   | 100000  | 10000 | 250000  |
| keep_last          | 5       | 2     | 10      |
| log_every          | 1000    | 500   | 1000    |
| assert_improvement | false   | true  | false   |

The smoke config also overrides the world food density
(`num_patches: 400`, `regrow_interval_range: [20, 60]`) so the reward trend is
visible above eat-event Poisson noise within its tiny step budget.

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
mismatch is an error unless `--allow-config-mismatch` is passed. SIGINT triggers a
final checkpoint before exit.
