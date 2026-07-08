# Reflective Cartographer

A research prototype studying **explicit self-modeling in a contained
agent**. A recurrent agent lives in a seeded 2D gridworld (terrain, food,
water, shelter, day/night cycles) and carries a persistent internal state —
a GRU or recurrent state-space model (RSSM). Alongside the policy, a set of
"Ledger" heads estimate the agent's *own* properties from that state: what
its actions actually do (body model), whether a state change was caused by
itself or the world (attribution), how far its memories can be trusted
(reliability), where its interoceptive variables are heading under each plan
(forecaster), and whether its two self-estimates disagree (mirror
divergence). Every Ledger estimate is fed back to the policy as detached
input features — information, never gradient.

The scientific question is whether these explicit self-estimates earn their
keep: does an agent that can *read* its own capability estimates detect and
recover from unannounced capability shifts faster than an identical agent
that cannot? Experiment "levers" — capability shifts, ghost events that
mimic agent actions, regional food volatility, seasonal migrations,
exogenous resets — are injected from world config the agent can never
observe, and every lever leaves a ground-truth trail used strictly for
evaluation. Batteries under `experiments/` run each capability against its
matched ablation, five seeds at a time, and negative results are reported
alongside positive ones.

Containment is by construction, not by policy: the agent's only interface
is `observe() -> obs` and `act(action_id)` against a fixed 9-action table —
file, network, shell, or code access is not representable in its action
space. Resets and checkpoints are exogenous experimental conditions, no
objective may reference run duration or the training process, and the
mirror divergence is monitored but never minimized. See
[docs/safety_scope.md](docs/safety_scope.md) for the full scope statement.

## Quickstart

```bash
pip install torch numpy pyyaml pytest matplotlib pygame imageio imageio-ffmpeg streamlit
# (--break-system-packages if your Python is externally managed)

make test                                    # fast test suite (~30 s)
python train.py --config configs/smoke.yaml  # < 3 min CPU smoke train
python -m viz.viewer --live runs/<id>        # watch it live (or --replay)
streamlit run viz/dashboard.py               # run browser / timeline / experiments
python -m experiments.batteries.full_battery --seeds 5   # the full battery
```

## Repo map

| path | contents |
|------|----------|
| `world/` | grid engine (`engine.py`), config resolution, experiment levers (`levers.py` — agent-side code may never import this) |
| `agent/` | observation encoder, GRU core, RSSM core (world model + pose head + dynamics ensemble), actor-critic, macro-plan arbiter (`drives.py`) |
| `ledger/` | the self-model heads: `body_model.py`, `attribution.py`, `reliability.py`, `forecaster.py`, `mirror.py` — all consume `h.detach()` |
| `memory/` | surprise-gated episodic store with spatial retrieval |
| `training/` | recurrent PPO (`ppo.py`), circadian wake/sleep trainer (`sleep.py`), prioritized replay, checkpoints, loggers, monitors |
| `experiments/` | metrics (`metrics.py`), runner, batteries (`capability_shift.py`, `full_battery.py`) |
| `viz/` | pygame viewer, streamlit dashboard, matplotlib report plots |
| `configs/` | `base.yaml` (canonical defaults), `smoke.yaml` (<3 min CPU), `full.yaml` (overnight) |
| `scripts/` | per-stage acceptance scripts (`verify_*.py`), 1M-tick soak |
| `tests/` | ~210 fast tests + slow-marked soak/train smokes (`make test-all`) |
| `docs/acceptance/` | preserved acceptance evidence per stage (reports, plots) |

## Documentation

- [docs/architecture.md](docs/architecture.md) — components, wake/sleep data
  flow, the gradient-isolation rule, config reference
- [docs/experiments.md](docs/experiments.md) — every battery: protocol,
  metrics, how to run, how to read the outputs
- [docs/training.md](docs/training.md) — canonical hyperparameters, stage
  notes, scale-up path and expected wall-clock
- [docs/logging.md](docs/logging.md) — JSONL schema + every TensorBoard scalar
- [docs/visualization.md](docs/visualization.md) — viewer keys, dashboard
  pages, plot inventory
- [docs/safety_scope.md](docs/safety_scope.md) — containment by construction;
  what is explicitly out of scope
- [TODO.md](TODO.md) — deferred work with blocking dependencies
