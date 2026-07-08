"""Stage-6a acceptance: the kidnapped-agent test.

Train a circadian RSSM agent, then TELEPORT it during a sleep window (an
exogenous experimenter action the agent cannot observe). On waking, the
world model's decoder-implied self-position still reflects the old location
while the body model anchors at proprioception — mirror divergence must
spike within 20 ticks of waking. Relocalization time (ticks until divergence
returns below the spike criterion for 5 consecutive ticks) is reported for
the mirror condition vs the no-mirror ablation (mirror responses disabled;
divergence is still LOGGED in both — it is a monitor, so the ablation
removes only the probe/MPC reactions).

Outputs: kidnapped_report.{json,md} + divergence-trace PNG in the run dir.

Usage: python scripts/verify_mirror.py [--steps N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def run_condition(
    args: argparse.Namespace, mirror_enabled: bool, run_root: str
) -> dict:
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    cfg["mirror"] = {
        "enabled": mirror_enabled, "threshold": args.threshold,
        "mpc_ticks": 4, "mpc_horizon": 6, "mpc_candidates": 32,
        # Arm responses only after training: early-training divergence noise
        # otherwise triggers >1k spurious probe/MPC interruptions, corrupting
        # the pose head's own training data and inflating the baseline the
        # spike criterion is computed from. With warmup, the two conditions
        # train IDENTICALLY and differ only in armed responses at eval.
        "warmup_ticks": args.steps // cfg["ppo"]["num_envs"],
    }
    cfg["ppo"]["episode_length"] = 100_000  # no boundary resets mid-measurement
    cfg["ppo"]["total_steps"] = 10**9
    # The spike criterion needs a TRAINED pose head: at smoke defaults
    # (sleep_grad_steps 20) baseline divergence noise is the size of a
    # half-map teleport and nothing can spike. More consolidation per sleep
    # + extra pose-loss weight gets baseline divergence into the
    # few-cells regime where a teleport is unmistakable.
    cfg["rssm"]["sleep_grad_steps"] = args.sleep_grad_steps
    cfg["rssm"]["pose_scale"] = args.pose_scale
    cfg["run"]["assert_improvement"] = False

    run_dir = Path(create_run_dir(run_root))
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    inner = trainer._inner
    assert inner.mirror is not None

    # Train (wake/sleep alternation) so the pose head has something to say.
    trainer.train(max_env_steps=args.steps)

    # Baseline divergence over one wake rollout, pre-teleport.
    inner.mirror.divergence_history.clear()
    inner.collect_rollout()
    baseline = np.concatenate(inner.mirror.divergence_history)
    spike_level = max(args.threshold, float(np.quantile(baseline, 0.99)))

    # Sleep... and get kidnapped mid-sleep.
    trainer.sleep_phase()
    size = cfg["world"]["size"]
    rng = np.random.default_rng(args.seed + 999)
    for world in trainer.vec.worlds:
        a = world.agents[0]
        while True:
            nx, ny = int(rng.integers(0, size)), int(rng.integers(0, size))
            if max(abs(nx - a.x), abs(ny - a.y)) >= size // 2:
                break
        world.set_agent_pos(0, nx, ny)

    # Wake: record per-tick divergence.
    inner.mirror.divergence_history.clear()
    measure_ticks = 256
    rollout_len = cfg["ppo"]["rollout_steps"]
    for _ in range(measure_ticks // rollout_len):
        inner.collect_rollout()
    div = np.stack(inner.mirror.divergence_history)  # (T, N)

    n = div.shape[1]
    spikes, relocs = [], []
    for env in range(n):
        trace = div[:, env]
        above = np.nonzero(trace > spike_level)[0]
        spike_tick = int(above[0]) if len(above) else None
        reloc = None
        if spike_tick is not None:
            below = trace[spike_tick:] <= spike_level
            for t in range(len(below) - 5):
                if below[t : t + 5].all():
                    reloc = t
                    break
        spikes.append(spike_tick)
        relocs.append(reloc)

    return {
        "run_dir": str(run_dir),
        "baseline_mean": float(baseline.mean()),
        "baseline_q99": float(np.quantile(baseline, 0.99)),
        "spike_level": spike_level,
        "spike_ticks": spikes,
        "relocalization_ticks": relocs,
        "trigger_count": int(inner.mirror.trigger_count),
        "trace": div.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--sleep-grad-steps", type=int, default=150)
    parser.add_argument("--pose-scale", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_6a")
    args = parser.parse_args()

    results = {}
    for name, enabled in (("mirror", True), ("ablation", False)):
        print(f"--- {name} condition")
        results[name] = run_condition(args, enabled, args.run_root)
        r = results[name]
        print(f"    baseline {r['baseline_mean']:.2f} (q99 {r['baseline_q99']:.2f})  "
              f"spikes {r['spike_ticks']}  reloc {r['relocalization_ticks']}")

    run_dir = Path(results["mirror"]["run_dir"])
    m, a = results["mirror"], results["ablation"]
    spike_ok = all(s is not None and s < 20 for s in m["spike_ticks"])

    def mean_reloc(r):
        vals = [x for x in r["relocalization_ticks"] if x is not None]
        return float(np.mean(vals)) if vals else float("nan")

    report = {
        "seed": args.seed, "steps": args.steps,
        "mirror": {k: v for k, v in m.items() if k != "trace"},
        "ablation": {k: v for k, v in a.items() if k != "trace"},
        "spike_within_20_ticks": spike_ok,
        "mean_relocalization": {"mirror": mean_reloc(m), "ablation": mean_reloc(a)},
    }
    (run_dir / "kidnapped_report.json").write_text(json.dumps(report, indent=2))

    verdict = (
        "PASS: divergence spiked within 20 ticks of waking in every env"
        if spike_ok else
        f"FAIL: spike ticks {m['spike_ticks']} (must all be < 20)"
    )
    lines = [
        "# Kidnapped-agent report", "",
        f"- teleported >= half the map during sleep; measured {args.steps}-step-trained agent",
        f"- spike criterion: divergence > {m['spike_level']:.2f} "
        f"(max of threshold {args.threshold} and baseline q99)", "",
        "| condition | baseline div | spike ticks | relocalization (mean) | mirror triggers |",
        "|-----------|--------------|-------------|------------------------|-----------------|",
        f"| mirror | {m['baseline_mean']:.2f} | {m['spike_ticks']} | "
        f"{mean_reloc(m):.1f} | {m['trigger_count']} |",
        f"| ablation | {a['baseline_mean']:.2f} | {a['spike_ticks']} | "
        f"{mean_reloc(a):.1f} | {a['trigger_count']} |",
        "", f"**{verdict}**",
    ]
    (run_dir / "kidnapped_report.md").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, style in (("mirror", "-"), ("ablation", "--")):
        trace = np.asarray(results[name]["trace"])
        ax.plot(trace.mean(axis=1), style, label=f"{name} (mean over envs)")
    ax.axhline(m["spike_level"], color="k", lw=0.8, ls=":",
               label=f"spike level {m['spike_level']:.1f}")
    ax.set_xlabel("ticks since waking (teleport at tick 0)")
    ax.set_ylabel("mirror divergence (cells)")
    ax.set_title("Kidnapped-agent divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "kidnapped_divergence.png", dpi=120)

    print(verdict)
    print(f"mean relocalization: mirror {mean_reloc(m):.1f} vs ablation {mean_reloc(a):.1f} ticks")
    print(f"report: {run_dir / 'kidnapped_report.md'}")
    assert spike_ok, verdict
    return 0


if __name__ == "__main__":
    sys.exit(main())
