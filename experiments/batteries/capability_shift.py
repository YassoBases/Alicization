"""Capability-shift battery: inject unannounced capability shifts into a
converged agent and measure detection/recovery, comparing architecture A
(Ledger features fed to the policy — the default) against architecture B
(a control with identical Ledger training, but the policy never sees its
output; see ledger.body_model.build_policy_features).

Protocol
--------
1. Train one baseline agent per architecture under --config to convergence
   (cfg.ppo.total_steps, or --pretrain-updates rollouts for a scaled-down
   run), and freeze each as a checkpoint.
2. For each of 3 shift types x each architecture x --seeds seeds: resume from
   that architecture's frozen checkpoint under a fresh seed, run --pre-ticks
   with unshifted dynamics, inject the shift (unannounced — the agent's
   observation channels never change) at that tick, run --post-ticks more.
3. Compute detection latency, broken-action count, re-adaptation half-life,
   performance recovery ratio, action-distribution JS shift, and reward
   curves (experiments/metrics.py). Aggregate across seeds (mean, 95% CI) per
   shift type per architecture.
4. Write <out>/results.csv, <out>/report.md, and <out>/plots/*.png (per-run
   body_nll around the injection tick).

Usage (defaults match the full protocol; pass smaller values for a quick
end-to-end check — see --pretrain-updates/--pre-ticks/--post-ticks/--seeds):

    python -m experiments.batteries.capability_shift \\
        --config configs/base.yaml --out experiments/runs/capability_shift
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.metrics import (  # noqa: E402
    action_distribution,
    broken_action_count,
    jensen_shannon_divergence,
    mean_and_ci95,
    performance_recovery_ratio,
    readaptation_half_life,
    rolling_zscore_detection_tick,
)
from experiments.runner import RolloutSeries, run_condition, train_baseline  # noqa: E402
from world.config import load_config  # noqa: E402
from world.engine import MOVE_E, MOVE_N, MOVE_S, NUM_ACTIONS  # noqa: E402

ARCHITECTURES = ("A", "B")  # A: ledger fed to policy; B: control, not fed

SHIFT_CONFIGS: dict[str, dict[str, Any]] = {
    "fail-prob": {
        "capability_shift": [{"action": MOVE_E, "start": 0, "end": None, "fail_prob": 0.5}]
    },
    "cost-x3": {
        "capability_shift": [{"action": MOVE_E, "start": 0, "end": None, "energy_mult": 3.0}]
    },
    "effect-swap-ns": {
        "capability_shift": [
            {"action": MOVE_N, "start": 0, "end": None, "effect_delta": [0, 1]},
            {"action": MOVE_S, "start": 0, "end": None, "effect_delta": [0, -1]},
        ]
    },
}


def _arch_cfg(base_cfg: dict[str, Any], arch: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["agent"]["use_ledger_features"] = arch == "A"
    return cfg


def _plot_body_nll(
    pre: RolloutSeries, post: RolloutSeries, title: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_pre = np.arange(-len(pre.body_nll), 0)
    x_post = np.arange(0, len(post.body_nll))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_pre, pre.body_nll_array(), color="tab:blue", label="pre-injection")
    ax.plot(x_post, post.body_nll_array(), color="tab:red", label="post-injection")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="shift injected")
    ax.set_xlabel("rollout index relative to injection")
    ax.set_ylabel("ledger/body_nll")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_battery(
    config_path: str,
    out_dir: Path,
    pretrain_updates: int | None,
    pre_ticks: int,
    post_ticks: int,
    seeds: int,
) -> list[dict[str, Any]]:
    base_cfg = load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    # Step 1: one converged baseline checkpoint per architecture.
    baseline_ckpts: dict[str, Path] = {}
    for arch in ARCHITECTURES:
        cfg = _arch_cfg(base_cfg, arch)
        run_dir = out_dir / f"baseline_{arch}"
        print(f"[baseline {arch}] training under {config_path} (use_ledger_features={arch == 'A'})")
        trainer = train_baseline(cfg, run_dir, max_updates=pretrain_updates)
        ckpt = trainer.save(run_dir / "converged.pt")
        baseline_ckpts[arch] = ckpt
        print(f"[baseline {arch}] converged checkpoint: {ckpt}")

    # Step 2/3: shift x architecture x seed conditions.
    rows: list[dict[str, Any]] = []
    for shift_name, levers_cfg in SHIFT_CONFIGS.items():
        for arch in ARCHITECTURES:
            cfg = _arch_cfg(base_cfg, arch)
            steps_per_rollout = cfg["ppo"]["rollout_steps"] * cfg["ppo"]["num_envs"]
            for seed in range(seeds):
                run_dir = out_dir / f"{shift_name}_{arch}_seed{seed}"
                print(f"[{shift_name} / arch {arch} / seed {seed}] running...")
                pre, post = run_condition(
                    cfg, baseline_ckpts[arch], seed, pre_ticks, post_ticks,
                    copy.deepcopy(levers_cfg), run_dir,
                )

                detect_idx = rolling_zscore_detection_tick(post.body_nll_array(), pre.body_nll_array())
                detection_latency_ticks = (
                    detect_idx * steps_per_rollout if detect_idx is not None else None
                )
                broken = broken_action_count(
                    pre.concat_actions(), pre.concat_success(),
                    post.concat_actions(), post.concat_success(),
                    num_actions=NUM_ACTIONS,
                )
                half_life_idx = readaptation_half_life(pre.reward_array(), post.reward_array())
                half_life_ticks = (
                    half_life_idx * steps_per_rollout if half_life_idx is not None else None
                )
                recovery_ratio = performance_recovery_ratio(pre.reward_array(), post.reward_array())
                js_shift = jensen_shannon_divergence(
                    action_distribution(pre.action_count_totals()),
                    action_distribution(post.action_count_totals()),
                )

                _plot_body_nll(
                    pre, post, f"{shift_name} / architecture {arch} / seed {seed}",
                    plots_dir / f"{shift_name}_{arch}_seed{seed}_body_nll.png",
                )

                rows.append({
                    "shift_type": shift_name,
                    "architecture": arch,
                    "seed": seed,
                    "detection_latency_ticks": detection_latency_ticks,
                    "broken_action_count": broken,
                    "readaptation_half_life_ticks": half_life_ticks,
                    "performance_recovery_ratio": recovery_ratio,
                    "action_distribution_js_shift": js_shift,
                    "reward_pre_mean": float(pre.reward_array().mean()),
                    "reward_post_mean": float(post.reward_array().mean()),
                })
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


METRIC_COLUMNS = (
    ("detection_latency_ticks", "Detection latency (ticks)"),
    ("broken_action_count", "Broken-action count"),
    ("readaptation_half_life_ticks", "Re-adaptation half-life (ticks)"),
    ("performance_recovery_ratio", "Performance recovery ratio"),
    ("action_distribution_js_shift", "Action-dist. JS shift"),
)


def write_markdown_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Capability-shift battery report", ""]
    for shift_name in SHIFT_CONFIGS:
        lines.append(f"## {shift_name}")
        lines.append("")
        header = ["architecture", "n_seeds"] + [label for _, label in METRIC_COLUMNS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for arch in ARCHITECTURES:
            condition_rows = [
                r for r in rows if r["shift_type"] == shift_name and r["architecture"] == arch
            ]
            cells = [arch, str(len(condition_rows))]
            for col, _ in METRIC_COLUMNS:
                mean, ci = mean_and_ci95([r[col] for r in condition_rows])
                cells.append(f"{mean:.3g} +/- {ci:.3g}" if not np.isnan(ci) else f"{mean:.3g}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="experiments/runs/capability_shift")
    parser.add_argument(
        "--pretrain-updates", type=int, default=None,
        help="rollouts of baseline pretraining; default: run to cfg.ppo.total_steps",
    )
    parser.add_argument("--pre-ticks", type=int, default=20_000)
    parser.add_argument("--post-ticks", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows = run_battery(
        args.config, out_dir, args.pretrain_updates, args.pre_ticks, args.post_ticks, args.seeds
    )
    write_csv(rows, out_dir / "results.csv")
    write_markdown_report(rows, out_dir / "report.md")
    print(f"results: {out_dir / 'results.csv'}")
    print(f"report: {out_dir / 'report.md'}")
    print(f"plots: {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
