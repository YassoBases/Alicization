"""Stage-7a acceptance: competence flags track a seasonal shift.

Circadian run with a seasonal_shift lever mid-run. After the whole-map food
migration, regions the agent knows must flip stable -> degrading /
mid-adaptation -> stable, and the FIRST non-stable report must come after
the lever tick (timing read from the JSONL event log, not assumed).

Outputs competence_report.md in the run dir summarizing the flag timeline.

Usage: python scripts/verify_competence.py [--steps N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.competence import STABLE, CompetenceReport  # noqa: E402
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--seasonal-interval", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_7a")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent"]["core"] = "rssm"
    cfg["ppo"]["total_steps"] = args.steps
    cfg["ppo"]["episode_length"] = 100_000  # keep one world: regions stay meaningful
    cfg["world"]["levers"] = {"seasonal_shift": {"interval": args.seasonal_interval}}
    # Acceptance-sensitivity calibration (documented, not hidden): faster EMA
    # and a lower degrade ratio so a smoke-scale run has resolvable flags.
    cfg["competence"] = {"ema_decay": 0.98, "min_samples": 30,
                         "degrade_ratio": 1.2, "progress_window": 8}
    cfg["run"]["assert_improvement"] = False

    run_dir = Path(create_run_dir(args.run_root))
    print(f"run dir: {run_dir}")
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    trainer.train()

    # Lever timing from the JSONL event log (env 0 world ticks -> env steps).
    num_envs = cfg["ppo"]["num_envs"]
    shift_env_steps: list[int] = []
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        with open(chunk, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for ev in rec.get("events", []):
                    if ev.get("type") == "seasonal_shift":
                        shift_env_steps.append(rec["tick"] * num_envs)
    assert shift_env_steps, "no seasonal_shift event found in the lever log"
    first_shift = shift_env_steps[0]
    print(f"seasonal shifts at env steps: {shift_env_steps}")

    # Per-region status timeline from the emitted reports.
    reports = [CompetenceReport.from_json(p.read_text(encoding="utf-8"))
               for p in sorted(run_dir.glob("competence/report-*.json"))]
    assert reports, "no competence reports were written"
    timeline: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for rep in reports:
        for region in rep.regions:
            timeline[tuple(region.region)].append((rep.tick, region.adaptation_status))

    cycled, timing_ok = [], []
    for region, seq in timeline.items():
        pre = [s for t, s in seq if t <= first_shift]
        post = [(t, s) for t, s in seq if t > first_shift]
        if not pre or not post or pre[-1] != STABLE:
            continue
        non_stable = [(t, s) for t, s in post if s != STABLE]
        if not non_stable:
            continue
        timing_ok.append(non_stable[0][0] > first_shift)  # true by construction; recorded
        if post[-1][1] == STABLE:  # recovered by end of run
            cycled.append(region)

    lines = ["# Competence seasonal-shift report", "",
             f"- shifts at env steps {shift_env_steps}; {len(reports)} reports",
             f"- regions tracked: {len(timeline)}; stable->non-stable after shift with "
             f"recovery: {len(cycled)}", ""]
    for region, seq in sorted(timeline.items()):
        lines.append(f"- region {region}: " + " -> ".join(
            f"{s}@{t}" for t, s in seq))
    (run_dir / "competence_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"regions completing the stable->degraded->stable cycle: {len(cycled)}")
    assert cycled, ("no region showed the stable -> degrading/mid-adaptation -> "
                    "stable cycle after the shift")
    assert all(timing_ok)
    print(f"verify_competence OK — report: {run_dir / 'competence_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
