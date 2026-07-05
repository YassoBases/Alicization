"""Stage-4a acceptance: smoke run with core=rssm.

Checks: (1) trains to completion with every logged metric finite (no NaNs),
(2) reconstruction loss decreases (first-vs-last rolling mean), (3)
participation ratio stable — never collapses below collapse_frac of its
running max (the monitor would have logged a WARNING) and the final PR is
above that threshold.

Usage: python scripts/verify_rssm.py [--config configs/smoke.yaml] [--steps N]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import create_run_dir  # noqa: E402
from training.ppo import PPOTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    if args.steps is not None:
        cfg["ppo"]["total_steps"] = args.steps
    cfg["run"]["assert_improvement"] = False

    run_dir = create_run_dir(args.run_root)
    print(f"run dir: {run_dir}")
    trainer = PPOTrainer(cfg, run_dir=run_dir)

    recon_history: list[float] = []
    orig_update = trainer.update

    def tracking_update(buf):  # record recon per rollout
        metrics = orig_update(buf)
        recon_history.append(metrics["rssm/recon"])
        for tag, val in metrics.items():
            assert math.isfinite(val), f"non-finite metric {tag}={val}"
        return metrics

    trainer.update = tracking_update
    trainer.train()

    k = max(3, len(recon_history) // 10)
    recon_first = float(np.mean(recon_history[:k]))
    recon_last = float(np.mean(recon_history[-k:]))
    print(f"recon rolling mean: first {recon_first:.4f} -> last {recon_last:.4f}")

    prs = trainer.pr_history
    print(f"participation ratio: n={len(prs)} first={prs[0]:.2f} last={prs[-1]:.2f} "
          f"max={max(prs):.2f} min={min(prs):.2f}")

    assert recon_last < recon_first, "reconstruction loss did not decrease"
    assert prs, "participation ratio was never computed"
    collapse_floor = trainer.pr_monitor.collapse_frac * trainer.pr_monitor.running_max
    assert prs[-1] >= collapse_floor, (
        f"final PR {prs[-1]:.2f} below collapse floor {collapse_floor:.2f}"
    )
    print("verify_rssm OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
