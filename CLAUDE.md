# Reflective Cartographer

Research prototype: a sandboxed 2D gridworld agent with a persistent recurrent core and
an explicit internal-state model (the "Ledger") that estimates the agent's own
capabilities, memory reliability, and uncertainty, and feeds those estimates to the
policy as inputs.

## What this is / is not
- A contained simulation. The agent's only interface is observe() -> obs and
  act(action_id) against a fixed action table. No file, network, shell, or code access
  is representable in the action space.
- Resets/checkpoints are exogenous experimental conditions. No objective, reward, or
  loss may reference run duration, reset timing, or the training process itself.
- Use engineering vocabulary: capability_shift (not injury/damage), checkpoint/restore
  (not survival), interoceptive variables (not emotions).

## Hard rules
- GRADIENT ISOLATION: every Ledger head consumes h.detach(). Ledger losses NEVER
  backprop into the core RSSM/GRU. Core trains only on world-prediction + task loss.
- Determinism: world stepping is seeded and reproducible; same seed + same actions =>
  identical state hash. Tests enforce this.
- All experiment levers (capability_shift, ghost events, region volatility, resets)
  are injected by world config the agent cannot observe.
- Every metric that evaluates a forecast reports a baseline (identity predictor).

## Stack & conventions
- Python 3.11, PyTorch, numpy. pip install --break-system-packages if needed.
- Config via YAML in configs/ (base.yaml, smoke.yaml, full.yaml). No magic numbers in
  code; everything tunable lives in config.
- Logging: JSONL per-tick event log + TensorBoard scalars. Log schema in docs/logging.md.
- Tests: pytest, in tests/. Smoke tests must finish < 3 min on CPU.
- Type hints everywhere; dataclasses for configs and messages.

## Commands
- Run smoke training: python train.py --config configs/smoke.yaml
- Run tests: pytest -x -q
- Live viewer: python -m viz.viewer --run runs/latest
- Dashboard: streamlit run viz/dashboard.py