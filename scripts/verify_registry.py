"""Stage-8a / stage-B acceptance: lever-driven contradiction with correct
timing, and the CUSUM upgrade closing mean_shift's blind spot.

Default mode (stage-8a, kept green): train a frozen-policy circadian agent
with a capability_shift on MOVE_E at a tick mean_shift CAN see; the
capability hypothesis must go supported -> weakening -> contradicted AFTER
the lever, with zero pre-lever contradictions anywhere.

--blind-spot mode (stage-B, B4): the lever is placed where the OLD
detector provably cannot escalate — 1600, between CUSUM's baseline freeze
(arm_after 500 + window 1000 = 1500) and mean_shift's first armed check
(arm_after + 2*window = 2500). mean_shift gets exactly ONE check with a
clean prev-window straddling the onset (2500); by the next check (2750)
the prev-window is contaminated, the statistic collapses, and the single
weakening "recovers" — never contradicted. CUSUM's frozen baseline never
slides, so S keeps accumulating and must contradict within --max-checks
checks of the lever, with zero pre-lever contradictions on any action,
across --seeds seeds. Both detectors replay the SAME fixture run.

Usage:
    python scripts/verify_registry.py                  # stage-8a mode
    python scripts/verify_registry.py --blind-spot --seeds 3
"""

from __future__ import annotations

import argparse
import json
import shutil
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
CHECK_EVERY = 250
WINDOW = 1000  # smoke-scale window; see geometry comments in replay()


def train_fixture(args: argparse.Namespace, seed: int, shift_tick: int) -> Path:
    cfg = load_config(args.config)
    cfg["seed"] = seed
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
        {"action": MOVE_E, "start": shift_tick, "end": None, "fail_prob": 0.9}
    ]}
    cfg["run"]["assert_improvement"] = False
    run_dir = Path(create_run_dir(args.run_root))
    print(f"run dir: {run_dir} (seed {seed}, lever {shift_tick})")
    CircadianTrainer(cfg, run_dir=run_dir).train()
    return run_dir


def find_lever_tick(run_dir: Path) -> int:
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        for line in open(chunk, encoding="utf-8"):
            rec = json.loads(line)
            for ev in rec.get("events", []):
                if ev.get("type") == "capability_shift_start":
                    return int(rec["tick"])
    raise AssertionError("capability_shift_start missing from log")


def replay(run_dir: Path, detector: str, arm_after: int,
           max_tick: int) -> HypothesisRegistry:
    """One post-hoc monitor pass with a fresh registry state.

    Smoke-scale monitor geometry: the env-0 tick axis is steps/num_envs
    long, so the default 4000-tick windows barely fit once — window=1000.
    For mean_shift, contradiction (two CONSECUTIVE violating checks) also
    requires the check cadence to be smaller than the window: it is an
    onset detector, and consecutive checks must both straddle the change.
    arm_after keeps early settling out of every consumed window (moot on
    the frozen-policy fixture, kept for geometry realism).
    """
    if (run_dir / "researcher").exists():
        shutil.rmtree(run_dir / "researcher")  # fresh state per detector
    registry = HypothesisRegistry(run_dir)
    for h in build_default_hypotheses(world_size=32, num_actions=9,
                                      capability_test=detector):
        h.monitor["window"] = WINDOW
        if "arm_after" in h.monitor:
            h.monitor["arm_after"] = arm_after
        registry.add(h)
    for h in load_yaml_hypotheses(Path("researcher") / "hypotheses.yaml"):
        registry.add(h)
    engine = QueryEngine(run_dir)
    for now_tick in range(CHECK_EVERY, max_tick + 1, CHECK_EVERY):
        registry.check_all(engine, now_tick)
    return registry


def capability_contradictions(registry: HypothesisRegistry,
                              lever_tick: int) -> tuple[list, list, list]:
    """(MOVE_E post-lever, pre-lever false alarms any action, downstream)."""
    target_post, pre_false, downstream = [], [], []
    for action in range(9):
        h = registry.hypotheses[f"hyp-capability-success-{action}"]
        for t in h.transitions:
            if t["to"] != "contradicted":
                continue
            if t["tick"] <= lever_tick:
                pre_false.append((action, t["tick"]))
            elif action == MOVE_E:
                target_post.append(t["tick"])
            else:
                downstream.append((action, t["tick"]))
    return target_post, pre_false, downstream


def run_stage_8a_mode(args: argparse.Namespace) -> int:
    run_dir = (Path(args.run_dir) if args.run_dir
               else train_fixture(args, args.seed, args.shift_tick))
    lever_tick = find_lever_tick(run_dir)
    print(f"lever fired at world tick {lever_tick}")
    max_tick = args.steps // 4  # env-0 world ticks
    registry = replay(run_dir, args.detector, arm_after=1500,
                      max_tick=max_tick)

    target = registry.hypotheses[f"hyp-capability-success-{MOVE_E}"]
    print("\nMOVE_E hypothesis transitions:")
    for t in target.transitions:
        print(f"  @{t['tick']}: {t['from']} -> {t['to']}  ({t['evidence']})")

    seq = [t["to"] for t in target.transitions]
    assert "weakening" in seq and "contradicted" in seq, (
        f"MOVE_E stability was not contradicted: {seq}"
    )
    # Timing is asserted on CONTRADICTION (the alarm tier): weakening is the
    # cheap tier and may wobble early; one clean check recovers it. The
    # sustained signal must come after the lever.
    contradiction_tick = next(t["tick"] for t in target.transitions
                              if t["to"] == "contradicted")
    early_wobbles = sum(1 for t in target.transitions
                        if t["to"] == "weakening" and t["tick"] <= lever_tick)
    print(f"contradicted at {contradiction_tick} (lever {lever_tick}); "
          f"{early_wobbles} pre-lever weakening wobble(s), all recovered")
    assert contradiction_tick > lever_tick, (
        f"contradiction at {contradiction_tick} precedes the lever at {lever_tick}"
    )
    _, pre_false, downstream = capability_contradictions(registry, lever_tick)
    # Contradictions AFTER the lever on other actions can be genuine
    # downstream effects — with MOVE_E failing 90% the agent drifts west and
    # MOVE_W's blocked-at-edge rate really does change. Report those; assert
    # only on pre-lever purity.
    print(f"pre-lever false contradictions (want none): {pre_false}")
    print(f"post-lever downstream contradictions (observed, legitimate): {downstream}")
    assert not pre_false

    events = (run_dir / "researcher" / "contradiction_events.jsonl")
    assert events.exists() and events.read_text().strip()
    print("verify_registry OK")
    return 0


def run_blind_spot_mode(args: argparse.Namespace) -> int:
    """B4: CUSUM must catch a lever the onset detector provably cannot."""
    arm_after = 500   # frozen policy: no settling; freeze early on purpose
    shift = args.blind_spot_tick
    freeze_tick = arm_after + WINDOW            # CUSUM baseline complete
    first_ms_check = arm_after + 2 * WINDOW     # mean_shift's first armed check
    assert freeze_tick <= shift < first_ms_check - WINDOW + CHECK_EVERY, (
        f"lever {shift} is not in the blind spot "
        f"[{freeze_tick}, {first_ms_check - WINDOW + CHECK_EVERY})"
    )
    max_tick = args.steps // 4
    budget_ticks = args.max_checks * CHECK_EVERY
    all_ok = True
    for seed in range(args.seeds):
        run_dir = (Path(args.run_dir) if args.run_dir
                   else train_fixture(args, seed, shift))
        lever_tick = find_lever_tick(run_dir)

        ms = replay(run_dir, "mean_shift", arm_after, max_tick)
        ms_post, ms_pre, _ = capability_contradictions(ms, lever_tick)
        ms_target = ms.hypotheses[f"hyp-capability-success-{MOVE_E}"]
        ms_weakenings = [t["tick"] for t in ms_target.transitions
                         if t["to"] == "weakening"]

        cu = replay(run_dir, "cusum_frozen_baseline", arm_after, max_tick)
        cu_post, cu_pre, cu_down = capability_contradictions(cu, lever_tick)

        detect = cu_post[0] - lever_tick if cu_post else None
        print(f"\nseed {seed} (lever {lever_tick}):")
        print(f"  mean_shift : contradictions {ms_post} (want none — blind "
              f"spot), weakenings @{ms_weakenings} (transient, recovered)")
        print(f"  cusum      : contradicted at {cu_post[:1] or 'NEVER'} "
              f"(+{detect} ticks, budget {budget_ticks}); "
              f"pre-lever false alarms {cu_pre + ms_pre} (want none); "
              f"downstream {cu_down}")
        seed_ok = (not ms_post and not cu_pre and not ms_pre
                   and detect is not None and detect <= budget_ticks)
        all_ok &= seed_ok
        if args.run_dir:
            break  # single-fixture replay
    assert all_ok, "blind-spot acceptance failed (see per-seed lines above)"
    print("\nverify_registry --blind-spot OK")
    return 0


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
    parser.add_argument("--detector", default="cusum_frozen_baseline",
                        choices=["cusum_frozen_baseline", "mean_shift"],
                        help="stage-8a mode detector (config default: cusum)")
    parser.add_argument("--blind-spot", action="store_true",
                        help="stage-B mode: lever in the onset detector's "
                             "blind spot; CUSUM must catch it, mean_shift "
                             "must (provably) not escalate")
    parser.add_argument("--blind-spot-tick", type=int, default=1600)
    parser.add_argument("--seeds", type=int, default=3,
                        help="blind-spot mode: fixtures trained per seed")
    parser.add_argument("--max-checks", type=int, default=6,
                        help="blind-spot mode: detection budget in checks")
    args = parser.parse_args()

    if args.blind_spot:
        return run_blind_spot_mode(args)
    return run_stage_8a_mode(args)


if __name__ == "__main__":
    sys.exit(main())
