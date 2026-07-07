"""Stage-5b acceptance: memory reliability on a volatile+stable world.

World: region_volatility lever relocates food patches in the LEFT half every
``--volatility-interval`` ticks; the right half stays stable. Two runs share
the seed: reliability enabled, and the reliability-blind ablation
(ledger.reliability.enabled=false — verification still runs so the learned
curves are comparable, but predictions influence nothing).

Checks / report:
  1. Learned decay differs between regions: fitted reliability-vs-age curves
     at the two measured region-volatility estimates (left vs right half).
  2. Stale-trip rate (memory-guided food trips arriving at empty cells, per
     1k ticks) with reliability < ablation. If it is NOT lower, that is a
     REPORTABLE RESULT — logged prominently, not tuned away.
  3. 10-bin ECE calibration of the reliability model.

Outputs in the reliability run's dir: reliability_report.{json,md},
reliability_curves.png.

Usage: python scripts/verify_reliability.py [--steps N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def build_cfg(args: argparse.Namespace, reliability_enabled: bool) -> dict:
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    cfg["agent"]["controller"] = "arbiter"
    cfg["memory"]["enabled"] = True
    cfg["ledger"]["reliability"]["enabled"] = reliability_enabled
    size = cfg["world"]["size"]
    half = size // 2 - 1
    cfg["world"]["levers"] = {
        "region_volatility": {
            "regions": [
                {"rect": [0, 0, half, size - 1], "interval": args.volatility_interval}
            ]
        }
    }
    # Long episodes: memory (cleared per episode) needs time to age + verify.
    cfg["ppo"]["episode_length"] = args.episode_length
    cfg["ppo"]["total_steps"] = args.steps
    # Sparse food (memory-guided trips only matter when food is not visible
    # everywhere; smoke default is ~40% coverage) with FAST regrowth, so
    # mismatch labels are dominated by the lever's relocations (left half)
    # rather than by the agent's own eating everywhere.
    cfg["world"]["food"] = {
        "num_patches": args.food_patches,
        "regrow_interval_range": [30, 80],
    }
    if args.sleep_grad_steps is not None:
        cfg["rssm"]["sleep_grad_steps"] = args.sleep_grad_steps
    cfg["run"]["assert_improvement"] = False
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--episode-length", type=int, default=8192)
    parser.add_argument("--volatility-interval", type=int, default=75)
    parser.add_argument("--food-patches", type=int, default=48)
    parser.add_argument("--sleep-grad-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_5b")
    args = parser.parse_args()

    results = {}
    trainers = {}
    for name, enabled in (("reliability", True), ("ablation", False)):
        cfg = build_cfg(args, enabled)
        run_dir = Path(create_run_dir(args.run_root))
        print(f"--- {name} run: {run_dir}")
        trainer = CircadianTrainer(cfg, run_dir=run_dir)
        trainer.train()
        rate = 1000.0 * trainer.stale_trip_count / max(1, trainer._arbiter_ticks)
        results[name] = {
            "run_dir": str(run_dir),
            "trips": trainer.trip_count,
            "stale_trips": trainer.stale_trip_count,
            "arbiter_ticks": trainer._arbiter_ticks,
            "stale_trip_rate_per_1k": rate,
            "verifications": trainer._inner.reliability.n_verifications,
        }
        trainers[name] = trainer
        print(f"    trips={trainer.trip_count} stale={trainer.stale_trip_count} "
              f"rate/1k={rate:.3f} verifications={results[name]['verifications']}")

    rel_trainer = trainers["reliability"]
    model = rel_trainer._inner.reliability
    assert model is not None and model.n_verifications > 0, "no verifications happened"

    # Region-volatility estimates: measured mismatch EMA per half of the map.
    vol = model.volatility
    half_cols = vol.grid.shape[1] // 2
    seen_l = vol.counts[:, :half_cols] > 0
    seen_r = vol.counts[:, half_cols:] > 0
    vol_left = float(vol.grid[:, :half_cols][seen_l].mean()) if seen_l.any() else 0.0
    vol_right = float(vol.grid[:, half_cols:][seen_r].mean()) if seen_r.any() else 0.0

    ages, curve_left = model.decay_curve(vol_left, max_age=args.episode_length)
    _, curve_right = model.decay_curve(vol_right, max_age=args.episode_length)
    ece, ece_rows = model.calibration_ece()

    run_dir = Path(results["reliability"]["run_dir"])
    report = {
        "seed": args.seed, "steps": args.steps,
        "volatility_interval": args.volatility_interval,
        "region_volatility_estimate": {"volatile_left": vol_left, "stable_right": vol_right},
        "decay_curves": {
            "ages": ages.tolist(),
            "volatile_left": curve_left.tolist(),
            "stable_right": curve_right.tolist(),
        },
        "ece_10bin": ece, "ece_bins": ece_rows,
        "stale_trip": results,
    }
    (run_dir / "reliability_report.json").write_text(json.dumps(report, indent=2))

    curves_differ = abs(curve_left[-1] - curve_right[-1]) > 0.05
    stale_better = (
        results["reliability"]["stale_trip_rate_per_1k"]
        < results["ablation"]["stale_trip_rate_per_1k"]
    )
    lines = [
        "# Memory-reliability report", "",
        f"- volatile left half (food relocates every {args.volatility_interval} ticks), stable right half",
        f"- measured region-volatility estimates: left {vol_left:.3f}, right {vol_right:.3f}",
        f"- verifications: {results['reliability']['verifications']}; 10-bin ECE: {ece:.4f}", "",
        "## Fitted reliability-vs-age curves (the two regions)", "",
        f"- volatile-left curve: {curve_left[0]:.3f} (age 0) -> {curve_left[-1]:.3f} (age {int(ages[-1])})",
        f"- stable-right curve: {curve_right[0]:.3f} (age 0) -> {curve_right[-1]:.3f} (age {int(ages[-1])})",
        f"- **curves {'DIFFER' if curves_differ else 'DO NOT differ'}** (see reliability_curves.png)", "",
        "## Stale-trip rate (per 1k ticks)", "",
        "| condition | trips | stale | rate/1k |",
        "|-----------|-------|-------|---------|",
    ]
    for name in ("reliability", "ablation"):
        r = results[name]
        lines.append(f"| {name} | {r['trips']} | {r['stale_trips']} | "
                     f"{r['stale_trip_rate_per_1k']:.3f} |")
    verdict = (
        "PASS: stale-trip rate with reliability < ablation"
        if stale_better else
        "REPORTABLE RESULT: stale-trip rate with reliability is NOT below the "
        "ablation. Logged, not tuned away."
    )
    lines += ["", f"**{verdict}**"]
    (run_dir / "reliability_report.md").write_text("\n".join(lines))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ages, curve_left, label=f"volatile left (vol={vol_left:.2f})")
    ax1.plot(ages, curve_right, label=f"stable right (vol={vol_right:.2f})")
    ax1.set_xlabel("memory age (ticks)"); ax1.set_ylabel("predicted reliability")
    ax1.set_title("Fitted decay curves by region"); ax1.legend(); ax1.set_ylim(0, 1)
    if ece_rows:
        ax2.bar([r["confidence"] for r in ece_rows],
                [r["accuracy"] for r in ece_rows], width=0.08, label="observed")
        ax2.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
        ax2.set_xlabel("predicted reliability"); ax2.set_ylabel("observed match rate")
        ax2.set_title(f"Calibration (ECE={ece:.3f})"); ax2.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "reliability_curves.png", dpi=120)

    print(f"curves: left {curve_left[-1]:.3f} vs right {curve_right[-1]:.3f} at max age "
          f"({'differ' if curves_differ else 'no difference'})")
    print(verdict)
    print(f"report: {run_dir / 'reliability_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
