"""Competence-tracker tests: known-answer synthetic trace through the
stable -> degrading -> mid-adaptation -> stable cycle, report/JSON round
trip, replay coverage, no-torch rule, and trainer integration."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ledger.competence import (
    DEGRADING,
    MID_ADAPTATION,
    STABLE,
    CompetenceReport,
    CompetenceTracker,
)

ROOT = Path(__file__).resolve().parent.parent
POS = (4, 4)  # region (0, 0)


def make_tracker(**kw) -> CompetenceTracker:
    defaults = dict(world_size=32, region_size=8, ema_decay=0.9,
                    progress_window=5, trail_window=12, degrade_ratio=1.5,
                    min_samples=10)
    defaults.update(kw)
    return CompetenceTracker(**defaults)


def feed(tracker: CompetenceTracker, wm: float, n: int) -> None:
    for _ in range(n):
        tracker.update_tick(POS, wm_loss=wm, body_brier=0.1, reward=0.5)
    tracker.snapshot_progress()


def status(tracker: CompetenceTracker) -> str:
    report = tracker.report(tick=0, run_id="test")
    (region,) = [r for r in report.regions if r.region == (0, 0)]
    return region.adaptation_status


def test_adaptation_cycle_on_synthetic_trace() -> None:
    """Loss 1.0 (stable baseline) -> step to 3.0 held flat (degrading) ->
    decaying toward 1.0 (mid-adaptation) -> settled (stable)."""
    t = make_tracker()
    for _ in range(8):
        feed(t, 1.0, 20)  # establish trailing best ~1.0
    assert status(t) == STABLE

    for _ in range(6):
        feed(t, 3.0, 20)  # jumped and NOT improving
    assert status(t) == DEGRADING

    for wm in (2.4, 2.0, 1.7, 1.55):  # falling but still > 1.5x best
        feed(t, wm, 20)
    assert status(t) == MID_ADAPTATION

    for _ in range(14):
        feed(t, 1.05, 20)  # recovered; step-era snapshots age out of the
    assert status(t) == STABLE  # trailing window (trail_window=12)


def test_min_samples_gate_and_per_region_isolation() -> None:
    t = make_tracker()
    t.update_tick((4, 4), 1.0, 0.1, 0.5)     # region (0,0): 1 sample only
    for _ in range(30):
        t.update_tick((20, 20), 1.0, 0.1, 0.5)  # region (2,2)
    report = t.report(0, "test")
    regions = {r.region for r in report.regions}
    assert (2, 2) in regions and (0, 0) not in regions


def test_report_json_roundtrip_and_replay_coverage() -> None:
    t = make_tracker()
    feed(t, 1.0, 30)
    # Replay positions: 3/4 in region (0,0), 1/4 in region (2,2); normalized.
    pos = np.array([[0.1, 0.1]] * 3 + [[0.7, 0.7]]).astype(float)
    report = t.report(tick=123, run_id="run-x", replay_positions=pos, world_size=32)
    (r00,) = [r for r in report.regions if r.region == (0, 0)]
    assert r00.replay_coverage == pytest.approx(0.75)
    assert r00.n_samples == 30

    restored = CompetenceReport.from_json(report.to_json())
    assert restored.tick == 123 and restored.run_id == "run-x"
    assert restored.regions[0].region == (0, 0)
    assert restored.regions[0].adaptation_status == r00.adaptation_status


def test_state_roundtrip() -> None:
    t = make_tracker()
    feed(t, 2.0, 40)
    t2 = make_tracker()
    t2.load_state_dict(t.state_dict())
    a = t.report(0, "a").regions[0]
    b = t2.report(0, "a").regions[0]
    assert a.wm_loss_ema == b.wm_loss_ema and a.n_samples == b.n_samples


def test_competence_module_never_imports_torch() -> None:
    """GRADIENT ISOLATION at the import level: this module computes from
    detached logs only — numpy in, floats out, no autograd possible."""
    tree = ast.parse((ROOT / "ledger" / "competence.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        assert not any("torch" in n for n in names), f"torch import found: {names}"


def test_trainer_feeds_tracker_and_checkpoints_it(tmp_path: Path) -> None:
    from training.ppo import PPOTrainer
    from world.config import load_config

    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = 5
    cfg["device"] = "cpu"
    cfg["agent"] = {"core": "rssm", "hidden_size": 16, "gru_layers": 1,
                    "encoder_channels": [4, 8]}
    cfg["rssm"].update(deter=16, stoch=4, embed=24, ensemble_k=3)
    cfg["competence"] = {"min_samples": 10}
    cfg["ppo"].update(rollout_steps=16, seq_len=8, num_envs=2, episode_length=64,
                      minibatch_transitions=16, epochs=1, total_steps=10**9,
                      anneal_lr=False)
    cfg["run"]["assert_improvement"] = False

    t = PPOTrainer(cfg)
    t.train(max_updates=3)
    report = t.competence.report(t.global_step, "fixture")
    assert report.regions, "tracker never received samples"
    assert all(np.isfinite(r.wm_loss_ema) for r in report.regions)

    ckpt = t.save(tmp_path / "ck.pt")
    t2 = PPOTrainer(cfg)
    t2.load(ckpt)
    r2 = t2.competence.report(t2.global_step, "fixture")
    assert len(r2.regions) == len(report.regions)
