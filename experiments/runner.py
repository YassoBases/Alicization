"""Experiment runner: train/resume PPOTrainer instances and run scripted
lever-injection conditions, collecting per-rollout metrics for battery
analysis (see experiments/batteries/*.py and experiments/metrics.py).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.metrics import realized_benefit_ab, success_criteria_hit
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


# ------------------------------------------------------ evaluation ladder
# stage-C4. Two tiers, one shared A/B core:
#   tier 0 (--ladder runs/<id>): automatic seeded smoke-A/B over every
#           PENDING config-knob proposal, marked evaluation=smoke_ab — a
#           cheap screen the human triggers in batch.
#   tier 1 (--ticket <id>):      the fuller A/B for a human-APPROVED
#           proposal (longer eval window), marked evaluation=ab.
# Both also score two fixed INDEPENDENT metrics (world-model loss, reward)
# beside the proposal's own criterion, both apply the degenerate-control
# guard (realized_benefit_ab -> NaN when the control has no variance), and
# both flag criteria trivially entailed by the knob.

# Independent metrics scored on every evaluation, regardless of the
# proposal's own criterion: PPO vs circadian name wm-loss differently, so
# each is a candidate list resolved to whichever tag the run actually logged.
INDEPENDENT_METRICS: dict[str, tuple[str, ...]] = {
    "wm_loss": ("loss/wm", "sleep/wm_total"),
    "reward": ("reward/rollout",),
}

# Criteria trivially entailed by the knob itself (the proposal_quality
# ANALYSIS caveat, results/20260708-1808/ANALYSIS.md): lowering
# rssm.free_nats mechanically lowers the KL it is measured against, so a
# KL-metric success criterion is near-tautological — "KL went down after we
# lowered the KL floor" is not evidence the world model improved. Grows as
# more knob/metric tautologies are found.
def tautology_flag(proposal: Any) -> str | None:
    change = proposal.proposed_change or {}
    metric = proposal.success_criteria.get("metric", "")
    if change.get("config_path") == "rssm.free_nats" and metric in ("rssm/kl", "sleep/kl"):
        return ("success criterion is the KL the free_nats knob directly "
                "clamps — near-tautological (proposal_quality ANALYSIS caveat); "
                "read the independent wm_loss/reward columns instead")
    return None


def _apply_change(cfg: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of cfg with change['config_path'] (dotted) set."""
    out = copy.deepcopy(cfg)
    node = out
    keys = change["config_path"].split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = change["new_value"]
    return out


def _needs_sleep(proposal: Any) -> bool:
    """Circadian trainer needed when the metric or the knob is sleep/RSSM-side."""
    metric = proposal.success_criteria["metric"]
    path = (proposal.proposed_change or {}).get("config_path", "")
    return metric.startswith(("sleep/", "competence/")) or path.startswith(
        ("rssm.", "ledger."))


def _load_run_cfg(run_dir: Path, config_path: str | None) -> dict[str, Any]:
    from world.config import load_config

    cfg_file = run_dir / "config.json"
    if config_path is not None:
        cfg = load_config(config_path)
    elif cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    else:
        cfg = load_config("configs/smoke.yaml")
    cfg.setdefault("run", {})["assert_improvement"] = False
    return cfg


def _eval_run(cfg: dict[str, Any], ticks: int, run_root: str,
              needs_sleep: bool) -> Path:
    """Train one short eval run and return its dir. Monkeypatched in tests."""
    from training.loggers import create_run_dir

    cfg = copy.deepcopy(cfg)
    cfg["ppo"]["total_steps"] = ticks
    cfg.setdefault("run", {})["assert_improvement"] = False
    eval_dir = Path(create_run_dir(run_root))
    if needs_sleep:
        from training.sleep import CircadianTrainer

        cfg["agent"]["core"] = "rssm"
        CircadianTrainer(cfg, run_dir=eval_dir).train()
    else:
        PPOTrainer(cfg, run_dir=eval_dir).train()
    return eval_dir


def _series(run_dir: Path, metric: str) -> np.ndarray:
    """The recorded scalar series for a metric (empty if absent).
    Monkeypatched in tests."""
    from viz.dashboard import load_tb_scalars

    scalars = load_tb_scalars(run_dir)
    return (np.asarray(scalars[metric][1], dtype=float)
            if metric in scalars else np.zeros(0))


def _resolve(run_dir: Path, candidates: tuple[str, ...]) -> np.ndarray:
    for tag in candidates:
        s = _series(run_dir, tag)
        if s.size:
            return s
    return np.zeros(0)


def _independent_ab(control_dir: Path, treated_dir: Path) -> dict[str, float]:
    """A/B benefit on the two fixed independent metrics (control-std units;
    NaN under the degenerate-control guard)."""
    out: dict[str, float] = {}
    for name, cands in INDEPENDENT_METRICS.items():
        control = _resolve(control_dir, cands)
        treated = _resolve(treated_dir, cands)
        out[name] = (float(realized_benefit_ab(treated, control))
                     if control.size and treated.size else float("nan"))
    return out


def evaluate_ab(proposal: Any, cfg: dict[str, Any], eval_ticks: int,
                evaluation_label: str, eval_run_root: str) -> dict[str, Any]:
    """Shared A/B core for both tiers: seeded control vs the knob applied,
    benefit in control-std units on the proposal's own metric plus the two
    independent metrics, with the tautology flag. Degenerate control -> NaN."""
    metric = proposal.success_criteria["metric"]
    direction = proposal.expected_benefit.get("direction", "up")
    needs_sleep = _needs_sleep(proposal)
    control_dir = _eval_run(cfg, eval_ticks, eval_run_root, needs_sleep)
    treated_dir = _eval_run(_apply_change(cfg, proposal.proposed_change),
                            eval_ticks, eval_run_root, needs_sleep)
    control, treated = _series(control_dir, metric), _series(treated_dir, metric)
    benefit = (realized_benefit_ab(treated, control)
               if control.size and treated.size else float("nan"))
    if direction == "down" and not np.isnan(benefit):
        benefit = -benefit
    hit = (success_criteria_hit(treated, float(proposal.success_criteria["threshold"]),
                                direction, window=len(treated))
           if treated.size else False)
    return {
        "metric": metric, "benefit_normalized": float(benefit),
        "met_success_criteria": bool(hit), "evaluation": evaluation_label,
        "direction": direction,
        "independent_metrics": _independent_ab(control_dir, treated_dir),
        "tautological_criterion": tautology_flag(proposal),
        "control_run": control_dir.name, "treated_run": treated_dir.name,
        "eval_ticks": eval_ticks,
    }


def run_ladder(run_dir: str | Path, config_path: str | None = None,
               eval_ticks: int | None = None,
               eval_run_root: str = "runs") -> list[Any]:
    """Tier-0: auto seeded smoke-A/B over every PENDING config-knob proposal.
    Marks each evaluation=smoke_ab and status=evaluated (a cheap automatic
    screen — experiment/architecture proposals and already-decided records
    are left for the human-gated tier-1 ticket path)."""
    from proposals.schema import load_all, save_proposal

    run_dir = Path(run_dir)
    cfg = _load_run_cfg(run_dir, config_path)
    ticks = int(eval_ticks
                or cfg.get("evaluation_ladder", {}).get("tier0_eval_ticks", 2048))
    evaluated: list[Any] = []
    for proposal in load_all(run_dir):
        if proposal.status != "pending":
            continue
        if proposal.intervention_class != "config" or not proposal.proposed_change:
            continue
        rb = evaluate_ab(proposal, cfg, ticks, "smoke_ab", eval_run_root)
        proposal.realized_benefit = rb
        proposal.status = "evaluated"
        proposal.linked_experiment_id = rb["treated_run"]
        save_proposal(proposal, run_dir)
        evaluated.append(proposal)
    return evaluated


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
    """Tier-1: run an APPROVED ticket's evaluation and close the loop. A
    knob proposal gets the full A/B (evaluation=ab); a non-knob experiment
    proposal gets the single-run threshold check (evaluation=threshold),
    both with the two independent metrics recorded alongside."""
    from proposals.schema import load_proposal, save_proposal

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
    cfg = _load_run_cfg(run_dir, config_path)

    if proposal.proposed_change:
        # Full A/B (control vs knob applied), independent metrics + tautology.
        rb = evaluate_ab(proposal, cfg, ticks, "ab", eval_run_root)
        proposal.linked_experiment_id = rb["treated_run"]
    else:
        # Non-knob experiment: one treated run, threshold on the own metric,
        # independent metric MEANS (no control to normalize against).
        eval_dir = _eval_run(cfg, ticks, eval_run_root, _needs_sleep(proposal))
        observed_s = _series(eval_dir, metric)
        observed = float(np.mean(observed_s)) if observed_s.size else float("nan")
        direction = proposal.expected_benefit.get("direction", "up")
        threshold = float(criteria["threshold"])
        met = (False if np.isnan(observed)
               else (observed >= threshold) if direction == "up"
               else observed <= threshold)
        indep = {name: (float(np.mean(_resolve(eval_dir, cands)))
                        if _resolve(eval_dir, cands).size else float("nan"))
                 for name, cands in INDEPENDENT_METRICS.items()}
        rb = {
            "metric": metric, "observed": observed, "threshold": threshold,
            "direction": direction, "met_success_criteria": bool(met),
            "evaluation": "threshold", "independent_metrics": indep,
            "tautological_criterion": tautology_flag(proposal),
            "eval_run": eval_dir.name, "eval_ticks": ticks,
        }
        proposal.linked_experiment_id = eval_dir.name

    proposal.realized_benefit = rb
    proposal.status = "evaluated"
    save_proposal(proposal, run_dir)
    return proposal.realized_benefit


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Proposal evaluation ladder (human-run). Use --ladder "
                    "runs/<id> for the tier-0 batch smoke-A/B over config "
                    "knobs, or --ticket <id> for a tier-1 approved proposal."
    )
    parser.add_argument("--ladder", metavar="RUN_DIR", default=None,
                        help="tier-0: batch smoke-A/B all pending config knobs")
    parser.add_argument("--ticket", default=None,
                        help="tier-1: evaluate one approved proposal")
    parser.add_argument("--run-dir", default=None,
                        help="run dir for --ticket (--ladder takes it directly)")
    parser.add_argument("--eval-ticks", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.ladder:
        evaluated = run_ladder(args.ladder, config_path=args.config,
                               eval_ticks=args.eval_ticks)
        print(f"tier-0 ladder: {len(evaluated)} config-knob proposal(s) "
              f"smoke-A/B evaluated")
        for p in evaluated:
            rb = p.realized_benefit
            taut = " [TAUTOLOGICAL]" if rb.get("tautological_criterion") else ""
            print(f"  {p.id} {p.target}: benefit {rb['benefit_normalized']:+.3f} "
                  f"wm_loss {rb['independent_metrics']['wm_loss']:+.3f} "
                  f"reward {rb['independent_metrics']['reward']:+.3f}{taut}")
        return 0
    if not args.ticket or not args.run_dir:
        parser.error("provide --ladder RUN_DIR, or --ticket <id> --run-dir <dir>")
    result = evaluate_ticket(args.ticket, args.run_dir, args.eval_ticks, args.config)
    print(f"realized_benefit: {result}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
