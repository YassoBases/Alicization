"""Reliability-model tests: summary comparison, region volatility isolation
from lever config, logistic learning, ECE, retrieval weighting, inspect plan,
and gradient isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agent.drives import NUM_PLANS, PLANS, plan_action
from ledger.reliability import (
    RegionVolatility,
    ReliabilityModel,
    compare_summaries,
)

ROOT = Path(__file__).resolve().parent.parent

REL_CFG = {
    "enabled": True, "lr": 1e-2, "age_tau": 1000.0, "radius": 2,
    "min_age": 10, "verify_cooldown": 5, "queue_capacity": 500,
    "region_size": 8, "volatility_ema": 0.2,
}


# --------------------------------------------------------- summary comparison


def _window(fill_food: bool = False) -> dict[str, np.ndarray]:
    return {
        "food": np.full((5, 5), fill_food, dtype=bool),
        "water": np.zeros((5, 5), dtype=bool),
    }


def test_compare_summaries_food_recall_label() -> None:
    """Label = fraction of REMEMBERED food cells still present (food recall),
    not whole-window agreement — one moved patch must move the label a lot."""
    stored = _window(False)
    stored["food"][1, 1] = True
    stored["food"][3, 3] = True
    # Both still there -> 1.0.
    assert compare_summaries(stored, {k: v.copy() for k, v in stored.items()}, (0, 0)) == 1.0
    # One of two gone -> 0.5 (whole-window agreement would say ~0.96).
    half = {k: v.copy() for k, v in stored.items()}
    half["food"][3, 3] = False
    assert compare_summaries(stored, half, (0, 0)) == pytest.approx(0.5)
    # All gone -> 0.0; extra NEW food elsewhere does not inflate recall.
    gone = _window(False)
    gone["food"][0, 4] = True
    assert compare_summaries(stored, gone, (0, 0)) == 0.0


def test_compare_summaries_offset_alignment() -> None:
    """A food cell at stored (3,2) seen from one cell east must align with
    observed (2,2) — same world cell, shifted window."""
    stored = _window(False)
    stored["food"][2, 3] = True
    observed = _window(False)
    observed["food"][2, 2] = True  # same world cell, viewed from x+1
    assert compare_summaries(stored, observed, (1, 0)) == 1.0
    # Without the shift the remembered cell appears empty.
    assert compare_summaries(stored, observed, (0, 0)) == 0.0


def test_compare_summaries_nothing_to_verify_returns_none() -> None:
    assert compare_summaries(_window(), _window(), (5, 0)) is None  # no overlap
    assert compare_summaries(_window(), _window(True), (0, 0)) is None  # no stored food
    stored = _window(False)
    stored["food"][0, 0] = True  # remembered food falls OUTSIDE the overlap
    # offset (+2,+2): overlap covers stored[2:, 2:], so cell (0,0) is excluded.
    assert compare_summaries(stored, _window(), (2, 2)) is None


# ---------------------------------------------------------- region volatility


def test_region_volatility_tracks_mismatch_by_region() -> None:
    vol = RegionVolatility(world_size=32, region_size=8, ema=0.3)
    for _ in range(20):
        vol.update((4, 4), mismatch=1.0)    # volatile region (0,0)
        vol.update((28, 28), mismatch=0.0)  # stable region (3,3)
    assert vol.get((4, 4)) > 0.9
    assert vol.get((28, 28)) == 0.0
    assert vol.get((5, 6)) == vol.get((4, 4))  # same 8x8 region


def test_nothing_in_ledger_imports_levers() -> None:
    """Volatility must be learned from verification history only — no module
    under ledger/ (or agent/, memory/) may IMPORT world.levers (docstrings may
    mention it when documenting ground-truth provenance)."""
    import ast

    for pkg in ("ledger", "agent", "memory"):
        for path in (ROOT / pkg).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not any("levers" in n for n in names), (
                    f"{path} imports {names} — agent-side code must never "
                    f"read lever config"
                )


# ------------------------------------------------------------------ learning


def test_logistic_model_learns_volatility_dependent_decay() -> None:
    """Synthetic verifications: volatile-region labels decay with age, stable
    stay high. The fitted curves must separate — decayed reliability at high
    age in the volatile region, sustained in the stable one."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ReliabilityModel(REL_CFG, world_size=64)
    for _ in range(3000):
        age = float(rng.uniform(0, 5000))
        if rng.random() < 0.5:  # volatile region: staleness grows with age
            vol, label = 0.8, float(rng.random() > min(0.9, age / 3000.0))
            pos = (4, 4)
        else:                    # stable region
            vol, label = 0.05, float(rng.random() > 0.05)
            pos = (60, 60)
        feats = np.array([age / model.age_tau, 1.0, 0.2, vol], dtype=np.float32)
        model.record(feats, label, pos)
        model.train_step(batch_size=64)

    _, curve_volatile = model.decay_curve(volatility=0.8, max_age=5000)
    _, curve_stable = model.decay_curve(volatility=0.05, max_age=5000)
    assert curve_stable[-1] > curve_volatile[-1] + 0.2, (
        f"curves did not separate: stable {curve_stable[-1]:.3f} "
        f"volatile {curve_volatile[-1]:.3f}"
    )
    # Volatile region must show age decay.
    assert curve_volatile[-1] < curve_volatile[0] - 0.1

    ece, rows = model.calibration_ece()
    assert np.isfinite(ece) and 0.0 <= ece <= 1.0
    assert sum(r["count"] for r in rows) == len(model.queue_y)


def test_reliability_grad_isolation() -> None:
    """The BCE step touches only the logistic layer (inputs are plain floats,
    so there is no path to anything else — assert it stays that way)."""
    model = ReliabilityModel(REL_CFG, world_size=64)
    for i in range(20):
        model.record(np.array([0.1 * i, 1.0, 0.0, 0.2], dtype=np.float32),
                     float(i % 2), (i, i))
    loss = model.train_step()
    assert loss is not None and np.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())


# ------------------------------------------------------------- inspect plan


def test_inspect_plan_navigates_to_target_and_lingers() -> None:
    rng = np.random.default_rng(0)
    grid = np.zeros((8, 11, 11), dtype=np.float32)
    inspect = PLANS.index("inspect")
    # Far target: move toward it (east).
    a = plan_action(inspect, grid, 3, 5, rng, pos=(10, 10), target_pos=(20, 10))
    assert a == 2  # MOVE_E
    # Arrived (within r=1): linger (REST) so verification can fire.
    a = plan_action(inspect, grid, 3, 5, rng, pos=(20, 11), target_pos=(20, 10))
    assert a == 5  # REST
    # No target: NOOP (the arbiter masks inspect out in this case anyway).
    assert plan_action(inspect, grid, 3, 5, rng, pos=(1, 1)) == 8


def test_forage_uses_memory_target_when_no_food_visible() -> None:
    rng = np.random.default_rng(0)
    grid = np.zeros((8, 11, 11), dtype=np.float32)  # no food anywhere
    forage = PLANS.index("forage_nearest")
    a = plan_action(forage, grid, 3, 5, rng, pos=(10, 10), target_pos=(10, 30))
    assert a == 1  # MOVE_S toward remembered food
    assert NUM_PLANS == 5
