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
- PROPOSALS ARE DATA, NEVER CODE: nothing under proposals/ or review/ may execute,
  import execution machinery, or write outside runs/<id>/proposals/ and
  experiments/tickets/. A human implements every approved change by hand. Tests
  enforce this (tests/test_proposals_no_execution.py).
- The simulated continuity metric NEVER couples to process lifetime, shutdown, or
  researcher actions. It is a dependent variable inside the sandbox — computed from
  run logs only, present in no loss, reward, or policy input. Tests enforce this.
- SCOPE RULE (researcher/): hypotheses and questions may be about the
  agent-in-the-world and its own models ONLY. Statements about the reviewer, reset
  schedule, or training infrastructure are rejected at the schema level and tested
  with adversarial fixtures. All proposal-layer structural rules apply verbatim to
  researcher/ (data-not-code, confined writes, no world/training imports).

## Stack & conventions
- Python 3.11, PyTorch, numpy. pip install --break-system-packages if needed.
- Config via YAML in configs/ (base.yaml, smoke.yaml, full.yaml). No magic numbers in
  code; everything tunable lives in config.
- Logging: JSONL per-tick event log + TensorBoard scalars. Log schema in docs/logging.md.
- Tests: pytest, in tests/. Smoke tests must finish < 3 min on CPU.
- Type hints everywhere; dataclasses for configs and messages.

## Commands
- Run smoke training: python train.py --config configs/smoke.yaml
- Print a resolved config: python train.py --config configs/smoke.yaml --dry-run
- Resume from checkpoint: python train.py --config <cfg> --resume <ckpt.pt>
- Run tests: pytest -x -q (or make test)
- Soak test (1M-tick random agent): python scripts/soak.py (or make soak)
- Live viewer: python -m viz.viewer --live runs/<id>  (scrub a finished run: --replay runs/<id>; render video: --replay runs/<id> --record out.mp4)
- Dashboard: streamlit run viz/dashboard.py