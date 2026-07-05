# Training

> **PLACEHOLDER VALUES.** The Training Defaults below are provisional and were not
> supplied by the project owner yet. When the canonical values are provided, update
> this table and regenerate `configs/base.yaml` / `configs/smoke.yaml` /
> `configs/full.yaml` from it. Configs must always mirror this section.

## Training Defaults

### Algorithm (PPO)

| key                | base    | smoke  | full      |
|--------------------|---------|--------|-----------|
| total_env_steps    | 5000000 | 20000  | 50000000  |
| num_envs           | 8       | 2      | 16        |
| rollout_length     | 256     | 64     | 256       |
| lr                 | 3.0e-4  | 3.0e-4 | 3.0e-4    |
| anneal_lr          | true    | false  | true      |
| gamma              | 0.99    | 0.99   | 0.997     |
| gae_lambda         | 0.95    | 0.95   | 0.95      |
| clip_range         | 0.2     | 0.2    | 0.2       |
| update_epochs      | 4       | 2      | 4         |
| num_minibatches    | 8       | 2      | 8         |
| entropy_coef       | 0.01    | 0.01   | 0.01      |
| value_coef         | 0.5     | 0.5    | 0.5       |
| max_grad_norm      | 0.5     | 0.5    | 0.5       |

### Model

| key                 | base | smoke | full |
|---------------------|------|-------|------|
| core                | gru  | gru   | gru  |
| core_hidden         | 256  | 64    | 512  |
| obs_embed_dim       | 128  | 32    | 256  |
| encoder_channels    | 16,32| 8,16  | 32,64|

### Run management

| key               | base    | smoke | full    |
|-------------------|---------|-------|---------|
| seed              | 0       | 0     | 0       |
| device            | cpu     | cpu   | cpu     |
| checkpoint_every  | 100000  | 10000 | 250000  |
| keep_last         | 5       | 2     | 10      |
| log_every         | 1000    | 500   | 1000    |

## Checkpoint format

`training/checkpoints.py` saves a single `.pt` file containing: model state_dict,
optimizer state_dict, global step, torch / numpy / python RNG states, and a SHA-256
hash of the resolved config. On `--resume`, the config hash of the checkpoint is
compared against the active config; a mismatch is an error unless
`--allow-config-mismatch` is passed.
