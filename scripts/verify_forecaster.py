"""Stage-4c acceptance: forecaster NMSE vs the identity predictor.

Runs the circadian trainer in arbiter mode (agent.core=rssm,
agent.controller=arbiter), then evaluates the forecaster on the accumulated
(h, plan, realized-future) tuples: NMSE = MSE(forecast) / MSE(identity
predictor) per horizon. The identity baseline appears on every plot
(CLAUDE.md: every forecast metric reports a baseline).

Target: NMSE < 1.0 at k=10. If the forecaster does NOT beat identity, that is
a REPORTABLE RESULT — this script logs it prominently and still writes the
report; it does not tune it away or hide it (exit code stays 0 unless the run
itself is broken).

Outputs in the run dir: forecaster_report.json, forecaster_report.md,
forecaster_nmse.png.

Usage: python scripts/verify_forecaster.py [--steps N] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent.drives import NUM_PLANS, PLANS  # noqa: E402
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
    cfg["agent"]["controller"] = "arbiter"
    if args.steps is not None:
        cfg["ppo"]["total_steps"] = args.steps
    if args.sleep_grad_steps is not None:
        cfg["rssm"]["sleep_grad_steps"] = args.sleep_grad_steps
    cfg["run"]["assert_improvement"] = False

    run_dir = Path(create_run_dir(args.run_root))
    print(f"run dir: {run_dir}")
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    trainer.train()

    batch = trainer.tuple_store.batch(NUM_PLANS, trainer.device)
    assert batch is not None and batch["h"].shape[0] >= 100, (
        "too few forecast tuples accumulated to evaluate"
    )
    n_tuples = batch["h"].shape[0]
    plan_counts = batch["plan"].argmax(dim=-1).bincount(minlength=NUM_PLANS).tolist()

    with torch.no_grad():
        out = trainer.forecaster(batch["h"], batch["plan"])

    rows = []
    for k in trainer.horizons:
        target = batch["future"][k]
        mse_f = (out[k][0] - target).pow(2).mean().item()
        mse_i = (batch["intero_now"] - target).pow(2).mean().item()
        rows.append({
            "horizon": k, "mse_forecaster": mse_f, "mse_identity": mse_i,
            "nmse": mse_f / mse_i if mse_i > 0 else float("inf"),
        })

    # ------------------------------------------------------------- reporting
    report = {
        "config": args.config, "seed": args.seed,
        "env_steps": trainer.env_steps, "n_tuples": n_tuples,
        "plan_counts": dict(zip(PLANS, plan_counts)),
        "results": rows,
    }
    (run_dir / "forecaster_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Forecaster report", "",
        f"- config: `{args.config}`, seed {args.seed}, {trainer.env_steps} env steps",
        f"- {n_tuples} (h, plan, realized-future) tuples; plan usage: "
        + ", ".join(f"{p}={c}" for p, c in zip(PLANS, plan_counts)),
        "",
        "| horizon | MSE (forecaster) | MSE (identity baseline) | NMSE |",
        "|---------|------------------|-------------------------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['horizon']} | {r['mse_forecaster']:.6f} | "
            f"{r['mse_identity']:.6f} | {r['nmse']:.4f} |"
        )
    lines.append("")
    lines.append("NMSE < 1.0 means the forecaster beats the identity predictor.")

    k_max = max(trainer.horizons)
    final = next(r for r in rows if r["horizon"] == k_max)
    beats = final["nmse"] < 1.0
    verdict = (
        f"PASS: NMSE={final['nmse']:.4f} < 1.0 at k={k_max} (beats identity)"
        if beats else
        f"REPORTABLE RESULT: forecaster does NOT beat identity at k={k_max} "
        f"(NMSE={final['nmse']:.4f} >= 1.0). Logged, not tuned away."
    )
    lines += ["", f"**{verdict}**"]
    (run_dir / "forecaster_report.md").write_text("\n".join(lines))

    # Plot: forecaster vs identity MSE per horizon (baseline mandatory).
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(rows))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], [r["mse_forecaster"] for r in rows],
           width, label="forecaster")
    ax.bar([x + width / 2 for x in xs], [r["mse_identity"] for r in rows],
           width, label="identity baseline")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"k={r['horizon']}\nNMSE={r['nmse']:.3f}" for r in rows])
    ax.set_ylabel("MSE of intero forecast")
    ax.set_title("Forecaster vs identity predictor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "forecaster_nmse.png", dpi=120)

    for r in rows:
        print(f"k={r['horizon']}: MSE_f={r['mse_forecaster']:.6f} "
              f"MSE_id={r['mse_identity']:.6f} NMSE={r['nmse']:.4f}")
    print(verdict)
    print(f"report: {run_dir / 'forecaster_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
