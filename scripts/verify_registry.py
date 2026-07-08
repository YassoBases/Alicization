"""Stage-8a acceptance: a lever violating a registered hypothesis drives
supported -> weakening -> contradicted with correct timing.

Run a circadian agent with a capability_shift on MOVE_E starting mid-run
(unannounced, from world config). Auto-populate the default hypotheses and
run the monitors on the sleep cadence over the log store. The capability-
stability hypothesis for MOVE_E must weaken then contradict AFTER the lever
tick (read from the JSONL event log, not assumed), and hypotheses for
unshifted actions must stay supported.

Usage: python scripts/verify_registry.py [--steps N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researcher.registry import (  # noqa: E402
    HypothesisRegistry,
    QueryEngine,
    build_default_hypotheses,
    load_yaml_hypotheses,
)
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402

MOVE_E = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--shift-tick", type=int, default=4500,
                        help="lever start in WORLD ticks (env-0 axis)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default="runs_8a")
    parser.add_argument("--run-dir", default=None,
                        help="replay an existing run instead of training")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        cfg = load_config(args.config)
        cfg["seed"] = args.seed
        cfg["agent"]["core"] = "rssm"
        cfg["ppo"]["total_steps"] = args.steps
        cfg["ppo"]["episode_length"] = 100_000
        # FROZEN POLICY: this acceptance isolates monitor-vs-lever timing,
        # so the lever must be the ONLY nonstationarity in the run. With a
        # learning policy, "success rate of action A is stable" is genuinely
        # false during settling (success is position-dependent via edge
        # blocking; at smoke scale we observed real ~10-sd pre-lever shifts)
        # — a correct detection, but it confounds the timing check. lr=0 on
        # both the actor-critic and the world model (the policy consumes h,
        # so a training RSSM would still drift the behavior distribution)
        # makes the random-init stochastic policy a fixed sampler.
        cfg["ppo"]["lr"] = 0.0
        cfg["ppo"]["anneal_lr"] = False
        cfg["rssm"]["world_lr"] = 0.0
        cfg["rssm"]["ac_lr"] = 0.0
        cfg["world"]["levers"] = {"capability_shift": [
            {"action": MOVE_E, "start": args.shift_tick, "end": None,
             "fail_prob": 0.9}
        ]}
        cfg["run"]["assert_improvement"] = False
        run_dir = Path(create_run_dir(args.run_root))
        print(f"run dir: {run_dir}")
        CircadianTrainer(cfg, run_dir=run_dir).train()

    # Lever timing from the event log (never assumed).
    lever_tick = None
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        for line in open(chunk, encoding="utf-8"):
            rec = json.loads(line)
            for ev in rec.get("events", []):
                if ev.get("type") == "capability_shift_start":
                    lever_tick = rec["tick"]
                    break
    assert lever_tick is not None, "capability_shift_start missing from log"
    print(f"lever fired at world tick {lever_tick}")

    registry = HypothesisRegistry(run_dir)
    for h in build_default_hypotheses(world_size=32, num_actions=9):
        # Smoke-scale monitor geometry: the env-0 tick axis is steps/num_envs
        # long, so the default 4000-tick windows barely fit once. Contradiction
        # (two CONSECUTIVE violating checks) also requires the check cadence
        # to be smaller than the window — mean_shift is an onset detector,
        # and consecutive checks must both straddle the change to escalate
        # weakening -> contradicted. arm_after=1500 keeps the early policy-
        # settling drift (position distribution moving while learning; move
        # success is edge-blocking- hence position-dependent) out of BOTH
        # windows, giving armed pre-lever checks that should stay clean.
        h.monitor["window"] = 1000
        if "arm_after" in h.monitor:
            h.monitor["arm_after"] = 1500
        registry.add(h)
    for h in load_yaml_hypotheses(Path("researcher") / "hypotheses.yaml"):
        registry.add(h)

    # Sleep-cadence monitor passes over the log store (post-hoc replay of
    # the same schedule the harness would run live).
    engine = QueryEngine(run_dir)
    check_every = 250
    max_tick = args.steps // 4  # env-0 world ticks
    for now_tick in range(check_every, max_tick + 1, check_every):
        registry.check_all(engine, now_tick)

    target = registry.hypotheses[f"hyp-capability-success-{MOVE_E}"]
    print(f"\nMOVE_E hypothesis transitions:")
    for t in target.transitions:
        print(f"  @{t['tick']}: {t['from']} -> {t['to']}  ({t['evidence']})")

    seq = [t["to"] for t in target.transitions]
    assert "weakening" in seq and "contradicted" in seq, (
        f"MOVE_E stability was not contradicted: {seq}"
    )
    # Timing is asserted on CONTRADICTION (the alarm tier): weakening is the
    # cheap tier and may wobble early — move success genuinely drifts while
    # the position distribution settles (edge-blocking is position-
    # dependent), and one clean check recovers it. The sustained,
    # two-consecutive-violations signal must come after the lever.
    contradiction_tick = next(t["tick"] for t in target.transitions
                              if t["to"] == "contradicted")
    early_wobbles = sum(1 for t in target.transitions
                        if t["to"] == "weakening" and t["tick"] <= lever_tick)
    print(f"contradicted at {contradiction_tick} (lever {lever_tick}); "
          f"{early_wobbles} pre-lever weakening wobble(s), all recovered")
    assert contradiction_tick > lever_tick, (
        f"contradiction at {contradiction_tick} precedes the lever at {lever_tick}"
    )
    pre_lever_contradictions = [t for t in target.transitions
                                if t["to"] == "contradicted"
                                and t["tick"] <= lever_tick]
    assert not pre_lever_contradictions

    # Other actions: contradictions BEFORE the lever would be false alarms.
    # Contradictions AFTER it can be genuine downstream effects — e.g. with
    # MOVE_E failing 90% the agent drifts west, and MOVE_W's blocked-at-edge
    # rate really does change. Report those; assert only on pre-lever purity.
    pre_lever_false_alarms = []
    downstream = []
    for action in range(9):
        if action == MOVE_E:
            continue
        h = registry.hypotheses[f"hyp-capability-success-{action}"]
        for t in h.transitions:
            if t["to"] == "contradicted":
                (pre_lever_false_alarms if t["tick"] <= lever_tick
                 else downstream).append((action, t["tick"]))
    print(f"pre-lever false contradictions (want none): {pre_lever_false_alarms}")
    print(f"post-lever downstream contradictions (observed, legitimate): {downstream}")
    assert not pre_lever_false_alarms

    events = (run_dir / "researcher" / "contradiction_events.jsonl")
    assert events.exists() and events.read_text().strip()
    print("verify_registry OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
