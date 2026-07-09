"""Stage-C4 / Gate-C acceptance: the tier-0 evaluation ladder produces
>= 10 smoke-A/B-evaluated config-knob proposals in one run, each with the
two independent metrics scored and the tautology flag set where it applies.

A single smoke run's generators only emit a handful of DISTINCT knobs
(dedup is by (type, target)), so this seeds a run dir with >= 10 config
knobs spanning distinct real config paths — a legitimate exercise of the
ladder machinery — then runs `experiments.runner.run_ladder` over a tiny
config so the 2x(#knobs) eval trainings finish in CI-ish time.

Usage: python scripts/verify_ladder.py [--eval-ticks 512]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.runner import run_ladder  # noqa: E402
from proposals.schema import Proposal, load_all, save_proposal  # noqa: E402
from world.config import load_config  # noqa: E402

# Distinct real config paths _apply_change can override; each becomes one
# tier-0 knob proposal. free_nats is judged on sleep/kl -> the tautology
# case; the rest are judged on reward/rollout (a metric both trainers log
# and no single knob mechanically entails).
KNOBS = [
    ("rssm.free_nats", 0.5, "sleep/kl"),          # tautological criterion
    ("ppo.lr", 1e-4, "reward/rollout"),
    ("ppo.entropy_coef", 0.02, "reward/rollout"),
    ("ppo.clip", 0.1, "reward/rollout"),
    ("ppo.gamma", 0.98, "reward/rollout"),
    ("ppo.gae_lambda", 0.9, "reward/rollout"),
    ("ppo.value_coef", 0.4, "reward/rollout"),
    ("ppo.max_grad_norm", 1.0, "reward/rollout"),
    ("rssm.kl_balance", 0.9, "reward/rollout"),
    ("checkpoints.interval", 25_000, "reward/rollout"),
    ("ppo.minibatch_transitions", 8, "reward/rollout"),
]


def _tiny_cfg() -> dict:
    cfg = load_config("configs/smoke.yaml")
    cfg["device"] = "cpu"
    cfg["agent"] = {"hidden_size": 16, "gru_layers": 1,
                    "encoder_channels": [4, 8], "core": "rssm"}
    cfg["ppo"].update(rollout_steps=16, seq_len=8, num_envs=2, episode_length=64,
                      minibatch_transitions=16, epochs=1, anneal_lr=False)
    cfg["rssm"].update(sleep_grad_steps=2, sleep_every=64, batch_seqs=4, seq_len=8)
    cfg["run"]["assert_improvement"] = False
    return cfg


def _knob_proposal(run_dir: Path, path: str, value, metric: str) -> Proposal:
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id=run_dir.name,
        source="ledger", intervention_class="config",
        rationale=f"smoke-screen the knob {path} -> {value}",
        expected_benefit={"metric": metric, "direction": "up",
                          "magnitude_estimate": 0.0},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0.0, "gpu_hours": 0.1}, risks=[],
        success_criteria={"metric": metric, "threshold": -1e9,
                          "eval_window_ticks": 512},
        target=path, proposed_change={"config_path": path, "new_value": value},
    )
    save_proposal(p, run_dir)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-ticks", type=int, default=512)
    parser.add_argument("--run-root", default="runs_ladder")
    args = parser.parse_args()

    run_dir = Path(args.run_root) / "src"
    (run_dir / "proposals").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(_tiny_cfg()), encoding="utf-8")

    for path, value, metric in KNOBS:
        _knob_proposal(run_dir, path, value, metric)
    # One non-knob experiment proposal must be left UNTOUCHED by tier-0.
    exp = Proposal.new(
        type="evaluation", created_tick=1, run_id=run_dir.name, source="ledger",
        intervention_class="experiment", rationale="run a battery",
        expected_benefit={"metric": "reward/rollout", "direction": "up",
                          "magnitude_estimate": 0.0},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0.5, "gpu_hours": 2.0}, risks=[],
        success_criteria={"metric": "reward/rollout", "threshold": 0.0,
                          "eval_window_ticks": 512}, target="battery.full")
    save_proposal(exp, run_dir)

    print(f"seeded {len(KNOBS)} config knobs + 1 experiment in {run_dir}")
    evaluated = run_ladder(run_dir, eval_ticks=args.eval_ticks,
                           eval_run_root=str(Path(args.run_root) / "eval"))

    print(f"\ntier-0 evaluated {len(evaluated)} config-knob proposals:")
    tautological = 0
    for p in evaluated:
        rb = p.realized_benefit
        assert rb["evaluation"] == "smoke_ab", rb["evaluation"]
        assert set(rb["independent_metrics"]) == {"wm_loss", "reward"}
        taut = rb.get("tautological_criterion")
        tautological += bool(taut)
        print(f"  {p.target:28s} benefit={rb['benefit_normalized']:+8.3f}  "
              f"wm_loss={rb['independent_metrics']['wm_loss']:+7.3f}  "
              f"reward={rb['independent_metrics']['reward']:+7.3f}"
              f"{'  [TAUTOLOGICAL]' if taut else ''}")

    # Gate C: >= 10 tier-0 evaluated; all smoke_ab; the experiment proposal
    # was NOT touched; the free_nats/KL tautology was flagged.
    assert len(evaluated) >= 10, f"only {len(evaluated)} tier-0 evaluations"
    assert all(p.intervention_class == "config" for p in evaluated)
    final = {p.id: p for p in load_all(run_dir)}
    assert final[exp.id].status == "pending", "tier-0 touched an experiment proposal"
    assert final[exp.id].realized_benefit is None
    assert tautological >= 1, "free_nats/KL tautology was not flagged"
    # Degenerate guard sanity: no astronomical benefits.
    for p in evaluated:
        b = p.realized_benefit["benefit_normalized"]
        assert np.isnan(b) or abs(b) < 1e6, f"astronomical benefit {b}"

    print(f"\n{len(evaluated)} tier-0 smoke-A/B evaluations; "
          f"{tautological} tautological criterion flagged; "
          f"experiment proposal left pending")
    print("verify_ladder OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
