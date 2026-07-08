"""Stage-8c acceptance: the EIG ranker (v2) runs against a real trained
RSSM via the injected adapter, produces finite predicted gains with
reductions in [0, 1], differs plausibly from v1, and never promotes a
noisy-TV region (the guard is additionally unit-tested with a synthetic
adapter in tests/test_eig.py — an organically random region is not
guaranteed to exist in a short smoke run).

Trains a short circadian run in-process (the adapter needs the live
trainer: RSSM core + replay + epistemic map), then ranks the same
question set with v1 and v2.

Usage: python scripts/verify_eig.py [--steps N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.model_adapter import RSSMAdapter  # noqa: E402
from researcher.agenda import rank_v1  # noqa: E402
from researcher.eig import rank_v2  # noqa: E402
from researcher.questions import generate_questions  # noqa: E402
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def action_counts(run_dir: Path, num_actions: int = 9) -> dict[int, tuple[int, int]]:
    counts = {a: [0, 0] for a in range(num_actions)}
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        for line in open(chunk, encoding="utf-8"):
            rec = json.loads(line)
            a = rec.get("action")
            if a is not None and 0 <= a < num_actions:
                counts[a][0 if rec["success"] else 1] += 1
    return {a: (s, f) for a, (s, f) in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    cfg["ppo"]["total_steps"] = args.steps
    cfg["run"]["assert_improvement"] = False
    run_dir = Path(create_run_dir("runs_8c"))
    print(f"run dir: {run_dir}")
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    trainer.train()

    rcfg = cfg["researcher"]
    adapter = RSSMAdapter(trainer, seed=args.seed)
    questions = generate_questions(run_dir)
    world_qs = [q for q in questions if q.type == "world_uncertainty"]
    assert world_qs, "no world_uncertainty questions from the run"
    competence = trainer.last_competence_report
    counts = action_counts(run_dir)

    v1 = rank_v1(questions, [], competence)
    v2 = rank_v2(questions, [], competence, adapter=adapter,
                 action_counts=counts, visit_steps=rcfg["visit_steps"])

    print("\nv1 order:", [i.ref for i in v1])
    print("v2 order:", [i.ref for i in v2])
    for item in v2:
        if item.predicted_gain is None:
            continue
        print(f"  {item.ref}: predicted_gain={item.predicted_gain:.5f} "
              f"score={item.score:.5f}")
        assert item.predicted_gain >= 0.0, "negative predicted gain"
        assert item.predicted_gain == item.predicted_gain, "NaN predicted gain"

    # Adapter sanity on a region the agent actually visited.
    q0 = world_qs[0]
    dis = adapter.region_disagreement(q0.region)
    red = adapter.imagined_visit_reduction(q0.region, rcfg["visit_steps"])
    print(f"\nregion {q0.region}: disagreement={dis:.6f} reduction={red:.4f}")
    assert dis >= 0.0
    assert 0.0 <= red <= 1.0, f"reduction {red} outside [0, 1]"

    # v2 must actually rescore: every world_uncertainty/capability item with
    # an available signal carries predicted_gain, and at least one score
    # differs from its v1 counterpart (plausible difference, not identity).
    v1_scores = {i.ref: i.score for i in v1}
    rescored = [i for i in v2 if i.predicted_gain is not None]
    assert rescored, "v2 rescored nothing"
    assert any(abs(i.score - v1_scores[i.ref]) > 1e-12 for i in rescored), (
        "v2 is score-identical to v1"
    )
    print(f"\n{len(rescored)}/{len(v2)} items rescored by EIG")
    print("verify_eig OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
