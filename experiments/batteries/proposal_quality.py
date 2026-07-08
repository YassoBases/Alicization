"""Flagship comparative battery: ledger- vs logs-sourced proposal quality.

HYPOTHESIS (stated in every report header): ledger-sourced proposals
outperform logs-only-sourced proposals on realized benefit and calibration.
The analysis is written either way — a null or negative result goes in the
same table with the same prominence.

Protocol (per seed):
  1. LIFE RUN — circadian agent (rssm, arbiter, episodic memory, a seasonal
     lever for genuine mid-life degradation evidence), trained in segments.
     After each segment both generator variants run on the evidence so far,
     and the SCRIPTED REVIEWER processes the queue on that fixed schedule.
     The reviewer is blind BY CONSTRUCTION: its code path never reads the
     source field (it approves every pending proposal — a constant policy,
     so reviewer behavior cannot differ between sources; usefulness ratings
     are n/a under a scripted reviewer and reported as such).
  2. EVALUATION ROUND — every approved proposal is evaluated per Section 17:
     A/B (preferred) when the proposal carries a machine-readable
     proposed_change: a seeded control run vs the same run with the knob
     applied, benefit in control-std units; otherwise pre/post with drift
     correction on the life run's own metric series, and the record is
     marked evaluation=pre_post.
  3. ANALYSIS per source: realized-benefit mean with 95% CI, success-
     criteria hit rate, confidence ECE (10-bin, with counts — small-N noise
     expected below ~50 evaluated), acceptance rate (explicitly weak; never
     headlined alone), per-type breakdowns, and the SECONDARY lifetime
     analysis: does realized benefit improve with created_tick as the
     competence tracker accumulates history?

Output: experiments/results/<date>/proposal_quality/ with proposals.csv,
per_source_summary.csv, summary.md, calibration figures, lifetime figure.
ANALYSIS.md is written by the experimenter from the results, not
auto-generated.

Usage:
    python -m experiments.batteries.proposal_quality --seeds 5
    python -m experiments.batteries.proposal_quality --seeds 3 --scale quick
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.metrics import (  # noqa: E402
    acceptance_rate,
    ece_10bin,
    hit_rate,
    mean_and_ci95,
    realized_benefit_ab,
    success_criteria_hit,
    time_to_first_useful_ticks,
)
from proposals.evidence import evidence_from_run  # noqa: E402
from proposals.generator import GeneratorSuite  # noqa: E402
from proposals.schema import Proposal, load_all, save_proposal  # noqa: E402
from review.queue import ReviewQueue  # noqa: E402
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from viz.dashboard import load_tb_scalars  # noqa: E402
from viz.plots import calibration_diagram, reward_curve  # noqa: E402
from world.config import load_config  # noqa: E402

SCALES = {
    # review_every is the spec's "fixed schedule (e.g. every 500k ticks)"
    # scaled to laptop budgets; the results stamp their scale.
    "full": {"life_ticks": 500_000, "review_every": 100_000, "eval_ticks": 20_000},
    # eval_ticks must span several sleep cycles: TB scalars land once per
    # cycle, and realized_benefit_ab needs a control SERIES (>= 2 points,
    # nonzero variance) — 2048 steps produced exactly one point per tag.
    "quick": {"life_ticks": 24_576, "review_every": 8_192, "eval_ticks": 10_240},
}


def _life_cfg(config_path: str, seed: int, sc: dict[str, int]) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg["seed"] = seed
    cfg["agent"]["core"] = "rssm"
    cfg["agent"]["controller"] = "arbiter"
    cfg["memory"]["enabled"] = True
    cfg["ppo"]["episode_length"] = 100_000
    cfg["world"]["levers"] = {"seasonal_shift": {"interval": sc["review_every"] // 2}}
    cfg["world"]["food"] = {"num_patches": 48, "regrow_interval_range": [30, 80]}
    cfg["competence"] = {"ema_decay": 0.98, "min_samples": 30,
                         "degrade_ratio": 1.2, "progress_window": 8}
    cfg["run"]["assert_improvement"] = False
    return cfg


def _apply_change(cfg: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    node = out
    keys = change["config_path"].split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = change["new_value"]
    return out


def _metric_series(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    scalars = load_tb_scalars(run_dir)
    if metric not in scalars:
        return np.zeros(0), np.zeros(0)
    steps, values = scalars[metric]
    return np.asarray(steps), np.asarray(values, dtype=float)


def _eval_run(cfg: dict[str, Any], ticks: int, run_root: str) -> Path:
    cfg = copy.deepcopy(cfg)
    cfg["ppo"]["total_steps"] = ticks
    run_dir = Path(create_run_dir(run_root))
    CircadianTrainer(cfg, run_dir=run_dir).train()
    return run_dir


def run_life(seed: int, config_path: str, sc: dict[str, int],
             run_root: str) -> Path:
    """One life run with scheduled generation + blind scripted review."""
    cfg = _life_cfg(config_path, seed, sc)
    run_dir = Path(create_run_dir(run_root))
    print(f"[seed {seed}] life run: {run_dir}")
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    suite = GeneratorSuite(run_dir)
    queue = ReviewQueue(run_dir)

    segment_end = 0
    while segment_end < sc["life_ticks"]:
        segment_end = min(segment_end + sc["review_every"], sc["life_ticks"])
        trainer.train(max_env_steps=segment_end)
        ev_ledger = evidence_from_run(run_dir, "ledger")
        ev_logs = evidence_from_run(run_dir, "logs_only")
        fired = suite.run(ev_ledger, ev_logs)
        # Scripted blind review on the fixed schedule: approve everything
        # pending. This code path never reads .source — blind by construction.
        for p in queue.proposals(status="pending"):
            queue.decide(p.id, "approve", note="scripted battery reviewer")
        print(f"[seed {seed}] @ {segment_end}: {len(fired)} fired, "
              f"{len(queue.proposals(status='approved'))} approved total")
    return run_dir


def evaluate_all(run_dir: Path, sc: dict[str, int], eval_root: str) -> list[Proposal]:
    """Section-17 evaluation for every approved proposal of one life run."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    queue = ReviewQueue(run_dir)
    evaluated: list[Proposal] = []
    for p in queue.proposals(status="approved"):
        metric = p.success_criteria["metric"]
        direction = p.expected_benefit.get("direction", "up")
        # The eval runs are already capped at sc.eval_ticks, so the hit is
        # judged over the full recorded series (scalar points, not raw ticks
        # — the proposal's eval_window_ticks governs the run length instead).
        if p.proposed_change:
            control_dir = _eval_run(cfg, sc["eval_ticks"], eval_root)
            treated_dir = _eval_run(_apply_change(cfg, p.proposed_change),
                                    sc["eval_ticks"], eval_root)
            _, control = _metric_series(control_dir, metric)
            _, treated = _metric_series(treated_dir, metric)
            benefit = (realized_benefit_ab(treated, control)
                       if len(control) and len(treated) else float("nan"))
            if direction == "down":
                benefit = -benefit
            hit = success_criteria_hit(
                treated, float(p.success_criteria["threshold"]), direction,
                window=len(treated))
            form = "ab"
            eval_ref = f"{control_dir.name}|{treated_dir.name}"
        else:
            # No machine-readable change means this automated battery never
            # EXECUTED anything for the proposal — a pre/post around the
            # creation tick would measure the life run's natural drift, not
            # a causal effect. Section 17's pre/post fallback presumes the
            # change was actually applied (a human process); here we mark
            # the record not_executed and EXCLUDE it from benefit stats.
            benefit, hit, form, eval_ref = float("nan"), False, "not_executed", ""
        p.realized_benefit = {
            "metric": metric, "benefit_normalized": benefit,
            "met_success_criteria": bool(hit), "evaluation": form,
            "direction": direction, "eval_ref": eval_ref,
        }
        p.status = "evaluated"
        save_proposal(p, run_dir)
        evaluated.append(p)
    return evaluated


# ------------------------------------------------------------------ analysis


def summarize(all_proposals: list[Proposal], out_dir: Path, scale: str,
              seeds: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in all_proposals:
        rb = p.realized_benefit or {}
        rows.append({
            "id": p.id, "run": p.run_id, "seed_run": p.run_id,
            "source": p.source, "type": p.type, "target": p.target,
            "created_tick": p.created_tick, "confidence": p.confidence,
            "status": p.status,
            "benefit": rb.get("benefit_normalized"),
            "hit": rb.get("met_success_criteria"),
            "evaluation": rb.get("evaluation"),
        })
    with open(out_dir / "proposals.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Proposal-quality comparative battery",
        "",
        "**HYPOTHESIS: ledger-sourced proposals outperform logs-only-sourced "
        "proposals on realized benefit and calibration.** Written either way "
        "— a null or negative result belongs in this table with the same "
        "prominence.",
        "",
        f"- scale: **{scale}**, seeds: {seeds}, scripted blind reviewer "
        f"(approve-all on a fixed schedule; usefulness ratings n/a by "
        f"construction)",
        "",
        "| source | n evaluated | realized benefit (mean +/- CI95) | hit rate "
        "| ECE (n bins) | acceptance rate* | time-to-first-useful (ticks) |",
        "|--------|-------------|-----------------------------------|---------"
        "|--------------|------------------|------------------------------|",
    ]
    summary_rows = []
    for source in ("ledger", "logs_only"):
        sub = [r for r in rows if r["source"] == source]
        ev = [r for r in sub if r["benefit"] is not None
              and np.isfinite(r["benefit"])]
        benefits = [r["benefit"] for r in ev]
        hits = [bool(r["hit"]) for r in ev]
        mean_b, ci_b = mean_and_ci95(benefits)
        hr = hit_rate(hits)
        if ev:
            ece, bins = ece_10bin(np.array([r["confidence"] for r in ev]),
                                  np.array([float(r["hit"]) for r in ev]))
        else:
            ece, bins = float("nan"), []
        acc = acceptance_rate([r["status"] if r["status"] != "evaluated"
                               else "approved" for r in sub])
        ttfu = time_to_first_useful_ticks(
            [r["created_tick"] for r in ev], benefits)
        lines.append(
            f"| {source} | {len(ev)} | {mean_b:+.3f} +/- {ci_b:.3f} | "
            f"{hr:.2f} | {ece:.3f} ({len(bins)}) | {acc:.2f} | "
            f"{'inf' if np.isinf(ttfu) else int(ttfu)} |")
        summary_rows.append({
            "source": source, "n_evaluated": len(ev), "benefit_mean": mean_b,
            "benefit_ci95": ci_b, "hit_rate": hr, "ece": ece,
            "acceptance_rate": acc, "time_to_first_useful": ttfu,
        })
        if ev:
            calibration_diagram(
                [b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                out_dir / f"calibration_{source}.png",
                bin_counts=[b["count"] for b in bins], ece=ece,
                title=f"Proposal confidence calibration ({source})")
    lines += ["", "\\* acceptance rate measures reviewer behavior as much as "
                  "proposal quality — never headline it alone (Section 17). "
                  "Under this battery's approve-all scripted reviewer it is "
                  "1.0 by construction.", ""]

    # Per-type breakdown.
    lines += ["## Per-type breakdown", "",
              "| source | type | n | benefit mean | hit rate |",
              "|--------|------|---|--------------|----------|"]
    for source in ("ledger", "logs_only"):
        for ptype in sorted({r["type"] for r in rows}):
            sub = [r for r in rows if r["source"] == source
                   and r["type"] == ptype and r["benefit"] is not None
                   and np.isfinite(r["benefit"])]
            if not sub:
                continue
            mb, _ = mean_and_ci95([r["benefit"] for r in sub])
            lines.append(f"| {source} | {ptype} | {len(sub)} | {mb:+.3f} | "
                         f"{hit_rate([bool(r['hit']) for r in sub]):.2f} |")

    # Secondary analysis: quality vs lifetime.
    lines += ["", "## Proposal quality vs agent lifetime", ""]
    for source in ("ledger", "logs_only"):
        ev = [r for r in rows if r["source"] == source
              and r["benefit"] is not None and np.isfinite(r["benefit"])]
        if len(ev) >= 3:
            ticks = np.array([r["created_tick"] for r in ev], dtype=float)
            bens = np.array([r["benefit"] for r in ev], dtype=float)
            corr = float(np.corrcoef(ticks, bens)[0, 1])
            lines.append(f"- {source}: Pearson r(created_tick, benefit) = "
                         f"{corr:+.3f} over {len(ev)} proposals")
        else:
            lines.append(f"- {source}: too few evaluated proposals for a "
                         f"lifetime trend (n={len(ev)})")

    with open(out_dir / "per_source_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scale", choices=list(SCALES), default="full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sc = SCALES[args.scale]
    date = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out or f"experiments/results/{date}/proposal_quality")
    run_root = str(out_dir / "runs")
    eval_root = str(out_dir / "eval_runs")

    all_proposals: list[Proposal] = []
    for seed in range(args.seeds):
        life_dir = run_life(seed, args.config, sc, run_root)
        evaluated = evaluate_all(life_dir, sc, eval_root)
        print(f"[seed {seed}] evaluated {len(evaluated)} proposals")
        all_proposals.extend(load_all(life_dir))
        # Life-run reward curve for the report.
        _, reward = _metric_series(life_dir, "reward/rollout")
        if len(reward):
            reward_curve(reward, out_dir / f"life_reward_seed{seed}.png",
                         title=f"life run seed {seed}")

    summarize(all_proposals, out_dir, args.scale, args.seeds)
    print(f"summary: {out_dir / 'summary.md'}")
    print("Write ANALYSIS.md from the results — not auto-generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
