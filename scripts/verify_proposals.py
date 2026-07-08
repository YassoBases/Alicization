"""Stage-7b acceptance: generators fire on a real (stage-6-style) run.

Trains a smoke circadian run with the full diagnostic surface (rssm core,
arbiter controller, episodic memory + reliability, seasonal-shift lever,
competence reports), then replays it through the GeneratorSuite in BOTH
source variants. Requires: at least 3 distinct proposal types fire, and
every fired proposal's supporting observations reference records that
actually exist (tb: tags present in the run's scalars; competence: regions
present in the cited report).

Usage: python scripts/verify_proposals.py [--steps N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proposals.evidence import evidence_from_run  # noqa: E402
from proposals.generator import GeneratorSuite  # noqa: E402
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_7b")
    parser.add_argument("--run-dir", default=None,
                        help="replay an existing run instead of training")
    args = parser.parse_args()

    if args.run_dir:
        return replay(Path(args.run_dir))

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    cfg["agent"]["controller"] = "arbiter"
    cfg["memory"]["enabled"] = True
    cfg["ppo"]["total_steps"] = args.steps
    cfg["ppo"]["episode_length"] = 100_000
    cfg["world"]["levers"] = {"seasonal_shift": {"interval": 2000}}
    cfg["world"]["food"] = {"num_patches": 48, "regrow_interval_range": [30, 80]}
    cfg["competence"] = {"ema_decay": 0.98, "min_samples": 30,
                         "degrade_ratio": 1.2, "progress_window": 8}
    cfg["run"]["assert_improvement"] = False

    run_dir = Path(create_run_dir(args.run_root))
    print(f"run dir: {run_dir}")
    CircadianTrainer(cfg, run_dir=run_dir).train()
    return replay(run_dir)


def replay(run_dir: Path) -> int:
    from proposals.schema import load_all

    ev_ledger = evidence_from_run(run_dir, "ledger")
    ev_logs = evidence_from_run(run_dir, "logs_only")
    suite = GeneratorSuite(run_dir)
    suite.run(ev_ledger, ev_logs)
    # Everything in the dir counts (a replay after a partial earlier pass
    # dedup-suppresses records that already fired — they are still fired
    # proposals of this run).
    fired = load_all(run_dir)

    print(f"\n{len(fired)} proposals fired:")
    for p in fired:
        print(f"  {p.id}  {p.type:<20} source={p.source:<10} target={p.target}")
        print(f"    refs: {p.supporting_observations}")

    # Evidence refs must point at records that exist.
    for p in fired:
        for ref in p.supporting_observations:
            if ref.startswith("tb:"):
                tag = ref[3:].split("@")[0]
                assert tag in ev_ledger.scalars, f"{p.id}: dangling ref {ref}"
            elif ref.startswith("competence:"):
                assert ev_ledger.competence is not None, f"{p.id}: {ref} but no report"
                cited = ref.split("region-")[-1]
                regions = {str(tuple(r.region)) for r in ev_ledger.competence.regions}
                assert cited in regions, f"{p.id}: dangling region ref {ref}"

    types_fired = {p.type for p in fired}
    print(f"\ndistinct types fired: {sorted(types_fired)}")
    assert len(types_fired) >= 3, f"only {len(types_fired)} types fired (need >= 3)"
    sources = {p.source for p in fired}
    print(f"sources represented: {sorted(sources)}")
    print("verify_proposals OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
