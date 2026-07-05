"""Attribution-head acceptance check (Stage 3b).

Runs a real PPOTrainer training run with the ghost_events lever active and
verifies: (1) steady-state attribution accuracy vs. ground truth > 0.9 (mean
of the last ~20% of rollouts — the classifier starts randomly initialized and
needs some self-supervised warm-up, so a whole-run cumulative average would
conflate that warm-up noise with converged performance; this mirrors how
body_nll/reward trends are judged elsewhere in this project, via first-vs-last
rolling means rather than a whole-run average), and (2) zero
no-op-attributed-to-self violations over the WHOLE run (this one holds from
the start by construction, no warm-up needed).

Usage: python scripts/verify_attribution.py [--ticks 200000] [--ghost-rate 0.02]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import create_run_dir  # noqa: E402
from training.ppo import PPOTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=200_000)
    parser.add_argument("--ghost-rate", type=float, default=0.02)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["world"]["levers"] = {"ghost_events": {"rate": args.ghost_rate}}
    cfg["ppo"]["total_steps"] = args.ticks
    cfg["run"]["assert_improvement"] = False

    run_dir = create_run_dir(args.run_root)
    print(f"run dir: {run_dir}")
    trainer = PPOTrainer(cfg, run_dir=run_dir)
    trainer.train()

    t = trainer.attr_tracker
    hist = trainer.attribution_accuracy_history
    n_tail = max(1, len(hist) // 5)
    steady_state_accuracy = float(np.mean(hist[-n_tail:])) if hist else float("nan")

    print(f"ticks scored (whole run): {t.total}")
    print(f"attribution accuracy (whole-run cumulative): {t.accuracy:.4f}")
    print(f"attribution accuracy (steady-state, last {n_tail} rollouts): {steady_state_accuracy:.4f}")
    print(f"noop-attributed-to-self violations (whole run): {t.noop_self_violations}")
    print("confusion (rows=ground truth, cols=predicted; self/world/both):")
    for row in t.confusion:
        print(f"    {row}")
    print(f"report: {trainer.write_report()}")

    assert steady_state_accuracy > 0.9, (
        f"steady-state attribution accuracy {steady_state_accuracy:.4f} <= 0.9"
    )
    assert t.noop_self_violations == 0, (
        f"{t.noop_self_violations} no-op ticks were attributed to self"
    )
    print("verify_attribution OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
