"""Stage-7d acceptance: the continuity composite on wake-only vs wake+sleep.

Trains the two conditions (same seed), computes the composite + components
per eval window, and reports means with 95% CIs — components ALWAYS shown
alongside the composite. The acceptance requires a defensible, CI-qualified
difference report (direction is reported, not assumed).

Usage: python scripts/verify_continuity.py [--steps N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.continuity import COMPONENTS, compare_runs  # noqa: E402
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_7d")
    parser.add_argument("--window-ticks", type=int, default=1024,
                        help="eval window in ENV-0 ticks (the JSONL stream "
                             "is one env's; the config default of 4096 "
                             "yields a single window on smoke-scale runs)")
    parser.add_argument("--reuse", nargs=2, default=None,
                        metavar=("SLEEP_RUN", "WAKE_RUN"),
                        help="compare two existing run dirs instead of training")
    args = parser.parse_args()

    if args.reuse:
        dirs = {"wake+sleep": Path(args.reuse[0]), "wake-only": Path(args.reuse[1])}
    else:
        dirs = {}
        for name, sleep in (("wake+sleep", True), ("wake-only", False)):
            cfg = load_config(args.config)
            cfg["seed"] = args.seed
            cfg["agent"]["core"] = "rssm"
            cfg["agent"]["controller"] = "arbiter"  # forecaster NMSE component
            cfg["memory"]["enabled"] = True         # reliability ECE component
            cfg["rssm"]["sleep"] = sleep
            cfg["ppo"]["total_steps"] = args.steps
            cfg["ppo"]["episode_length"] = 100_000
            cfg["run"]["assert_improvement"] = False
            run_dir = Path(create_run_dir(args.run_root))
            print(f"--- {name}: {run_dir}")
            CircadianTrainer(cfg, run_dir=run_dir).train()
            dirs[name] = run_dir

    cfg = load_config(args.config)
    cont_cfg = cfg.get("continuity", {}) or {}
    results = compare_runs(
        list(dirs.values()),
        weights=cont_cfg.get("weights"),
        window_ticks=args.window_ticks or cont_cfg.get("window_ticks", 4096),
    )
    name_of = {Path(d).name: label for label, d in dirs.items()}

    print(f"\n{'condition':<12} {'composite':>22}  components (mean per window)")
    for run_id, r in results.items():
        mean, ci = r.mean_ci()
        comp_txt = "  ".join(
            f"{c}={np.nanmean(r.components[c]):+.3f}" for c in COMPONENTS
        )
        print(f"{name_of[run_id]:<12} {mean:+.4f} +/- {ci:.4f}  {comp_txt}")
        assert np.isfinite(mean), f"{run_id}: composite not finite"

    labels = list(dirs)
    m0 = results[dirs[labels[0]].name].mean_ci()
    m1 = results[dirs[labels[1]].name].mean_ci()
    print(f"\ndifference ({labels[0]} - {labels[1]}): {m0[0] - m1[0]:+.4f}")
    print("verify_continuity OK (direction reported, not assumed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
