"""Experiment runner: train/resume PPOTrainer instances and run scripted
lever-injection conditions, collecting per-rollout metrics for battery
analysis (see experiments/batteries/*.py and experiments/metrics.py).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.ppo import PPOTrainer
from world.engine import NUM_ACTIONS


@dataclass
class RolloutSeries:
    """Per-rollout metrics accumulated across a run window."""

    reward: list[float] = field(default_factory=list)
    body_nll: list[float] = field(default_factory=list)
    action_counts: list[np.ndarray] = field(default_factory=list)  # each (NUM_ACTIONS,)
    action: list[np.ndarray] = field(default_factory=list)  # raw per-tick actions
    success: list[np.ndarray] = field(default_factory=list)  # raw per-tick success

    def reward_array(self) -> np.ndarray:
        return np.asarray(self.reward)

    def body_nll_array(self) -> np.ndarray:
        return np.asarray(self.body_nll)

    def action_count_totals(self) -> np.ndarray:
        return np.sum(self.action_counts, axis=0) if self.action_counts else np.zeros(NUM_ACTIONS)

    def concat_actions(self) -> np.ndarray:
        return np.concatenate(self.action) if self.action else np.zeros(0, dtype=int)

    def concat_success(self) -> np.ndarray:
        return np.concatenate(self.success) if self.success else np.zeros(0, dtype=bool)


def train_baseline(
    cfg: dict[str, Any], run_dir: str | Path, max_updates: int | None = None
) -> PPOTrainer:
    """Train a fresh PPOTrainer under ``cfg`` (to ``cfg.ppo.total_steps``, or
    ``max_updates`` rollouts for a scaled-down "convergence") and return it."""
    trainer = PPOTrainer(cfg, run_dir=run_dir)
    trainer.train(max_updates=max_updates)
    return trainer


def reseed_for_condition(trainer: PPOTrainer, seed: int) -> None:
    """After resuming a frozen baseline checkpoint — which restores RNG state
    for bit-identical resume (training/checkpoints.py) — give this
    condition's replication genuinely independent randomness: reseed the
    global torch/numpy streams and each vectorized env's own RNG generator.
    Without this, all "seeds" of a condition would replay identically."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    for i, world in enumerate(trainer.vec.worlds):
        world.rng = np.random.default_rng(seed * 1000 + i)


def reset_learning_rates(trainer: PPOTrainer) -> None:
    """Undo any LR annealing from baseline pretraining. If the baseline was
    trained to convergence with ``anneal_lr: true``, its optimizers may have
    decayed to ~0 by the time it's frozen — which would make post-injection
    "re-adaptation" metrics meaningless (no LR left to adapt with). Battery
    conditions need genuine, undiminished learning capacity going forward."""
    for group in trainer.opt.param_groups:
        group["lr"] = trainer.pcfg["lr"]
    for group in trainer.body_opt.param_groups:
        group["lr"] = trainer.cfg["ledger"]["lr"]
    for group in trainer.attr_opt.param_groups:
        group["lr"] = trainer.cfg["ledger"]["attribution"]["lr"]


def collect_rollouts(trainer: PPOTrainer, num_rollouts: int) -> RolloutSeries:
    """Run ``num_rollouts`` full PPO + Ledger update cycles (online training
    continues exactly as in ``PPOTrainer.train``, just without its
    checkpoint/SIGINT/lr-anneal orchestration) and record per-rollout metrics."""
    series = RolloutSeries()
    for _ in range(num_rollouts):
        buf = trainer.collect_rollout()
        trainer.update(buf)
        body_metrics = trainer.update_body_model(buf)
        trainer.update_attribution_model(buf)

        series.reward.append(buf["reward"].sum(dim=0).mean().item())
        series.body_nll.append(body_metrics["ledger/body_nll"])
        action = buf["action"].cpu().numpy().reshape(-1)
        success = buf["real_success"].cpu().numpy().reshape(-1)
        series.action_counts.append(np.bincount(action, minlength=NUM_ACTIONS).astype(float))
        series.action.append(action)
        series.success.append(success)
    return series


def run_condition(
    baseline_cfg: dict[str, Any],
    baseline_ckpt: str | Path,
    seed: int,
    pre_ticks: int,
    post_ticks: int,
    levers_cfg: dict[str, Any],
    run_dir: str | Path,
) -> tuple[RolloutSeries, RolloutSeries]:
    """Resume ``baseline_ckpt`` under a fresh seed, run ``pre_ticks`` with the
    baseline's own (unshifted) dynamics, inject ``levers_cfg`` (unannounced —
    the agent's observation channels never change, only future environment
    dynamics do), then run ``post_ticks`` more. Returns (pre_series, post_series).
    """
    cfg = copy.deepcopy(baseline_cfg)
    cfg["seed"] = seed
    trainer = PPOTrainer(cfg, run_dir=run_dir)
    trainer.load(baseline_ckpt, allow_config_mismatch=True)
    reseed_for_condition(trainer, seed)
    reset_learning_rates(trainer)

    steps_per_rollout = cfg["ppo"]["rollout_steps"] * cfg["ppo"]["num_envs"]
    n_pre = max(1, pre_ticks // steps_per_rollout)
    n_post = max(1, post_ticks // steps_per_rollout)

    pre_series = collect_rollouts(trainer, n_pre)
    trainer.vec.inject_levers(levers_cfg)
    post_series = collect_rollouts(trainer, n_post)
    return pre_series, post_series


# ---------------------------------------------------------------- tickets
# python -m experiments.runner --ticket <proposal-id> --run-dir runs/<id>
#
# The HUMAN side of the proposal loop: executes the experiment named in an
# approved ticket, then writes realized_benefit back into the proposal
# record (status=evaluated — which also unblinds its source in the review
# CLI). This module is deliberately OUTSIDE proposals/ and review/: those
# packages must never execute anything.


def evaluate_ticket(
    ticket_id: str,
    run_dir: str | Path,
    eval_ticks: int | None = None,
    config_path: str | None = None,
    eval_run_root: str = "runs",
) -> dict[str, Any]:
    """Run the ticket's evaluation experiment and close the loop."""
    import json

    from proposals.schema import load_proposal, save_proposal
    from world.config import load_config

    run_dir = Path(run_dir)
    record_path = run_dir / "proposals" / f"{ticket_id}.json"
    proposal = load_proposal(record_path)
    if proposal.status not in ("approved", "partially_approved", "modified"):
        raise ValueError(
            f"ticket {ticket_id} is {proposal.status}; approve it first"
        )

    criteria = proposal.success_criteria
    metric = criteria["metric"]
    ticks = int(eval_ticks or criteria["eval_window_ticks"])

    cfg_file = run_dir / "config.json"
    if config_path is not None:
        cfg = load_config(config_path)
    elif cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    else:
        cfg = load_config("configs/smoke.yaml")
    cfg["ppo"]["total_steps"] = ticks
    cfg["run"]["assert_improvement"] = False

    from training.loggers import create_run_dir

    eval_dir = create_run_dir(eval_run_root)
    needs_sleep_metrics = metric.startswith(("sleep/", "competence/"))
    if needs_sleep_metrics:
        from training.sleep import CircadianTrainer

        cfg["agent"]["core"] = "rssm"
        trainer: Any = CircadianTrainer(cfg, run_dir=eval_dir)
        trainer.train()
    else:
        trainer = PPOTrainer(cfg, run_dir=eval_dir)
        trainer.train()

    from viz.dashboard import load_tb_scalars

    scalars = load_tb_scalars(eval_dir)
    if metric not in scalars:
        observed = float("nan")
    else:
        observed = float(np.mean(scalars[metric][1]))
    direction = proposal.expected_benefit.get("direction", "up")
    threshold = float(criteria["threshold"])
    met = (observed >= threshold) if direction == "up" else (observed <= threshold)
    if np.isnan(observed):
        met = False

    proposal.realized_benefit = {
        "metric": metric, "observed": observed, "threshold": threshold,
        "direction": direction, "met_success_criteria": bool(met),
        "eval_run": Path(eval_dir).name, "eval_ticks": ticks,
    }
    proposal.status = "evaluated"
    proposal.linked_experiment_id = Path(eval_dir).name
    save_proposal(proposal, run_dir)
    return proposal.realized_benefit


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute an approved proposal ticket (human-run)."
    )
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-ticks", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = evaluate_ticket(args.ticket, args.run_dir, args.eval_ticks, args.config)
    print(f"realized_benefit: {result}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
