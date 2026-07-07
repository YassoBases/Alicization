"""Stage-4b acceptance: circadian wake/sleep smoke run.

Checks: (1) wake and sleep genuinely alternate (>= 2 sleep windows ran with
grad steps), (2) reward/rollout trends up (first-vs-last rolling mean over
wake stretches), (3) all logged metrics finite.

Usage: python scripts/verify_sleep.py [--steps N] [--sleep-grad-steps N]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sleep-grad-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    if args.steps is not None:
        cfg["ppo"]["total_steps"] = args.steps
    if args.sleep_grad_steps is not None:
        cfg["rssm"]["sleep_grad_steps"] = args.sleep_grad_steps
    cfg["run"]["assert_improvement"] = False

    run_dir = create_run_dir(args.run_root)
    print(f"run dir: {run_dir}")
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    trainer.train()

    hist = trainer.reward_history
    k = max(2, len(hist) // 5)
    first, last = float(np.mean(hist[:k])), float(np.mean(hist[-k:]))
    slept = [m for m in trainer.sleep_metrics_history if m["sleep/grad_steps"] > 0]
    print(f"wake stretches: {len(hist)}  sleep windows (with grad steps): {len(slept)}")
    print(f"reward/rollout rolling mean: first {first:+.4f} -> last {last:+.4f}")
    for m in trainer.sleep_metrics_history:
        assert all(math.isfinite(v) for v in m.values()), f"non-finite sleep metric: {m}"

    assert len(slept) >= 2, "wake/sleep did not alternate"
    assert last > first, f"reward did not trend up: {first:+.4f} -> {last:+.4f}"
    print("verify_sleep OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
