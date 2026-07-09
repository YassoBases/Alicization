"""Stage-A3: head-convergence curves — attribution accuracy and forecaster
NMSE vs training budget.

Motivation (results/20260708-1311/ANALYSIS.md follow-up 3): attribution and
the forecaster both flipped between quick scale (12k ticks: 0.12 accuracy,
NMSE >> 1) and their acceptance scales (200k: >0.9; 50k/100: k10 0.78).
This script pins each head's minimum viable scale WITH DATA at ~5
log-spaced budgets, so MIN_VIABLE_SCALE contracts (full_battery.py) stop
carrying unknowns and future batteries stop producing known-undertrained
numbers.

One circadian run per (budget, seed) measures BOTH heads:
- ghost lever (rate 0.02) gives attribution its ground-truth labels;
  accuracy is scored over a FINAL eval window (the cumulative tracker is
  dominated by the untrained early phase — same rule as the battery), with
  the always-SELF majority baseline reported alongside.
- arbiter controller feeds the forecast tuple store; NMSE vs the identity
  predictor at k = 1, 10 exactly as the battery's forecaster sweep. The
  identity baseline (NMSE = 1.0) is drawn on the plot — a missing baseline
  is a bug (tests/test_viz.py).

Output: experiments/results/<date>/scale_curves/{curves.csv,
scale_curves.png}. Feed the results back into MIN_VIABLE_SCALE with the
source field citing this script's output directory.

Usage:
    python scripts/scale_curves.py                      # all 5 budgets
    python scripts/scale_curves.py --budgets 20000      # CI-time smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402

# ~log-spaced 20k -> 200k (ANALYSIS pinned attribution's flip somewhere in
# exactly this interval).
DEFAULT_BUDGETS = [20_000, 36_000, 63_000, 113_000, 200_000]


def run_point(config: str, budget: int, seed: int,
              sleep_grad_steps: int, eval_ticks: int,
              run_dir: Path) -> dict[str, float]:
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["agent"]["core"] = "rssm"
    cfg["agent"]["controller"] = "arbiter"     # feeds the forecast tuples
    cfg["ledger"]["horizons"] = [1, 10]
    cfg["ppo"]["total_steps"] = budget
    cfg["ppo"]["episode_length"] = 100_000
    cfg["rssm"]["sleep_grad_steps"] = sleep_grad_steps
    cfg["world"]["levers"] = {
        "ghost_events": {"rate": 0.02, "kinds": ["push", "consume_food"]}}
    cfg["run"]["assert_improvement"] = False
    cfg.setdefault("checkpoints", {})["interval"] = 10**12  # measurement run: no ckpt

    t = CircadianTrainer(cfg, run_dir=run_dir)
    t.train()
    inner = t._inner

    # Attribution over a final eval window (head keeps training online;
    # the policy/core are NOT updated here — Ledger heads own their
    # optimizers, gradient isolation intact).
    tracker = inner.attr_tracker
    pre_correct, pre_total = tracker.correct, tracker.total
    pre_self = sum(tracker.confusion[0])
    rollout_ticks = cfg["ppo"]["rollout_steps"] * cfg["ppo"]["num_envs"]
    for _ in range(max(8, eval_ticks // rollout_ticks)):
        buf = inner.collect_rollout()
        inner.update_body_model(buf)
        inner.update_attribution_model(buf)
    d_total = max(1, tracker.total - pre_total)
    acc = (tracker.correct - pre_correct) / d_total
    majority = (sum(tracker.confusion[0]) - pre_self) / d_total

    # Forecaster NMSE vs identity at k = 1, 10 (battery recipe).
    out: dict[str, float] = {"attribution_acc": float(acc),
                             "always_self_baseline": float(majority)}
    batch = t.tuple_store.batch(t.forecaster.num_plans, t.device)
    for k in (1, 10):
        if batch is None:
            out[f"nmse_k{k}"] = float("nan")
            continue
        with torch.no_grad():
            fc = t.forecaster(batch["h"], batch["plan"])
        target = batch["future"][k]
        mse_f = (fc[k][0] - target).pow(2).mean().item()
        mse_i = (batch["intero_now"] - target).pow(2).mean().item()
        out[f"nmse_k{k}"] = mse_f / mse_i if mse_i > 0 else float("inf")
    return out


def plot(rows: list[dict], path: Path) -> None:
    budgets = sorted({r["budget"] for r in rows})

    def series(key: str) -> list[float]:
        return [float(np.mean([r[key] for r in rows if r["budget"] == b]))
                for b in budgets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(budgets, series("attribution_acc"), "o-", label="attribution")
    ax1.plot(budgets, series("always_self_baseline"), "s--",
             label="always-SELF baseline")
    ax1.set_xscale("log"); ax1.set_xlabel("training budget (env steps)")
    ax1.set_ylabel("accuracy vs ground truth")
    ax1.set_title("Attribution convergence"); ax1.legend()

    for k, marker in ((1, "o-"), (10, "s-")):
        ax2.plot(budgets, series(f"nmse_k{k}"), marker, label=f"k={k}")
    ax2.axhline(1.0, color="k", ls="--", label="identity baseline (NMSE = 1)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("training budget (env steps)"); ax2.set_ylabel("NMSE")
    ax2.set_title("Forecaster convergence"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--budgets", type=int, nargs="*",
                        default=DEFAULT_BUDGETS)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--sleep-grad-steps", type=int, default=100,
                        help="full-battery consolidation budget (stage-4c)")
    parser.add_argument("--eval-ticks", type=int, default=6_144)
    # RESUMABLE: a FIXED default out dir (not dated) + per-point result cache,
    # so re-running the SAME command after a shutdown skips finished points
    # and continues. Each point is short (~one training run), so an
    # interruption costs at most the point in flight.
    parser.add_argument("--out", default="experiments/results/scale_curves")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cache = out_dir / "points"
    cache.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for budget in args.budgets:
        for seed in range(args.seeds):
            point_file = cache / f"b{budget}_s{seed}.json"
            if point_file.exists():
                print(f"=== budget {budget} seed {seed} (cached, skip) ===")
                rows.append(json.loads(point_file.read_text(encoding="utf-8")))
                continue
            print(f"=== budget {budget} seed {seed} ===")
            r = run_point(args.config, budget, seed, args.sleep_grad_steps,
                          args.eval_ticks, out_dir / f"runs/b{budget}_s{seed}")
            row = {"budget": budget, "seed": seed, **r}
            point_file.write_text(json.dumps(row), encoding="utf-8")
            rows.append(row)
            print(f"    acc {r['attribution_acc']:.3f} "
                  f"(always-SELF {r['always_self_baseline']:.3f})  "
                  f"nmse k1 {r['nmse_k1']:.3g} k10 {r['nmse_k10']:.3g}")
            with open(out_dir / "curves.csv", "w", newline="",
                      encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            plot(rows, out_dir / "scale_curves.png")
    print(f"curves: {out_dir / 'curves.csv'}")
    print("Feed the flip points back into MIN_VIABLE_SCALE "
          "(experiments/batteries/full_battery.py) citing this directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
