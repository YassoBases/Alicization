"""Lever tests: capability_shift isolation (chi-square), ghost ground truth,
volatility/seasonal relocation, and the agent/ledger import ban."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from world import engine
from world.config import load_config
from world.engine import NUM_ACTIONS, World

CHI2_CRIT_DF1 = 3.841  # alpha = 0.05

ROOT = Path(__file__).resolve().parent.parent


def base_cfg(seed: int = 0) -> dict:
    cfg = load_config(ROOT / "configs" / "base.yaml")
    cfg["seed"] = seed
    return cfg


def cfg_with_levers(levers: dict, seed: int = 0) -> dict:
    cfg = base_cfg(seed)
    cfg["world"] = copy.deepcopy(cfg["world"])
    cfg["world"]["levers"] = levers
    return cfg


def chi2_2x2(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Pearson chi-square for a 2x2 (variant x success/fail) table."""
    table = np.array([a, b], dtype=np.float64)
    total = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / total
    mask = expected > 0
    return float((((table - expected) ** 2)[mask] / expected[mask]).sum())


def test_capability_shift_changes_only_target_action_transitions() -> None:
    """Chi-square on >=20k paired transitions sampled from identical states."""
    target = engine.MOVE_N
    shift = {
        "capability_shift": [
            {"action": target, "start": 0, "end": None, "fail_prob": 0.5}
        ]
    }
    w_ref = World(base_cfg(seed=11))
    w_plain = World(base_cfg(seed=11))
    w_shift = World(cfg_with_levers(shift, seed=11))

    n_states = 1200  # 1200 states x 9 actions x 2 variants = 21600 transitions
    walk = np.random.default_rng(99)
    counts = {a: {"plain": [0, 0], "shift": [0, 0]} for a in range(NUM_ACTIONS)}
    outcomes_equal = {a: True for a in range(NUM_ACTIONS)}

    for _ in range(n_states):
        for _ in range(3):  # vary the reference state between samples
            w_ref.step([int(walk.integers(0, NUM_ACTIONS))])
        w_ref.drain_events()
        # A lever-free world consumes no RNG while stepping, so advance the
        # reference stream by hand: otherwise every snapshot carries the same
        # RNG state and the shift world's fail-roll is the same value forever.
        w_ref.rng.random()
        blob = w_ref.snapshot()
        for action in range(NUM_ACTIONS):
            results = {}
            for name, w in (("plain", w_plain), ("shift", w_shift)):
                w.restore(blob)
                _, infos = w.step([action])
                realized = infos[0]["realized"]
                counts[action][name][0 if realized["success"] else 1] += 1
                results[name] = (
                    realized["success"],
                    realized["dpos"],
                    round(realized["denergy"], 12),
                    (w.agents[0].x, w.agents[0].y),
                )
            if results["plain"] != results["shift"]:
                outcomes_equal[action] = False

    total = sum(sum(c["plain"]) + sum(c["shift"]) for c in counts.values())
    assert total >= 20_000

    for action in range(NUM_ACTIONS):
        chi2 = chi2_2x2(tuple(counts[action]["plain"]), tuple(counts[action]["shift"]))
        if action == target:
            assert chi2 > CHI2_CRIT_DF1, f"target action stats unchanged (chi2={chi2})"
        else:
            assert chi2 < CHI2_CRIT_DF1, f"action {action} leaked (chi2={chi2})"
            # Stronger than the chi-square: identical states + RNG must give
            # bitwise-identical transitions for non-targeted actions.
            assert outcomes_equal[action], f"action {action} outcomes diverged"


def test_capability_shift_is_reversible() -> None:
    shift = {
        "capability_shift": [
            {"action": engine.MOVE_E, "start": 50, "end": 100, "fail_prob": 1.0}
        ]
    }
    w = World(cfg_with_levers(shift))
    w.set_agent_pos(0, 32, 32)
    for _ in range(150):
        tick = w.tick
        action = engine.MOVE_E if w.tick % 2 == 0 else engine.MOVE_W
        _, infos = w.step([action])
        success = infos[0]["realized"]["success"]
        if action == engine.MOVE_E:
            assert success is (not 50 <= tick < 100), f"wrong success at tick {tick}"
        else:
            assert success is True


def test_capability_shift_effect_delta_swap() -> None:
    """Two shift entries with swapped effect_delta implement an NS effect-swap:
    pressing MOVE_N actually moves south (and vice versa) while active, and
    reverts to the normal delta once the window ends (reversible, like
    fail_prob/energy_mult)."""
    shift = {
        "capability_shift": [
            {"action": engine.MOVE_N, "start": 10, "end": 20, "effect_delta": [0, 1]},
            {"action": engine.MOVE_S, "start": 10, "end": 20, "effect_delta": [0, -1]},
        ]
    }
    w = World(cfg_with_levers(shift))
    for _ in range(30):
        tick = w.tick
        w.set_agent_pos(0, 32, 32)
        _, infos = w.step([engine.MOVE_N])
        dpos = infos[0]["realized"]["dpos"]
        expected = (0, 1) if 10 <= tick < 20 else (0, -1)
        assert dpos == expected, f"tick {tick}: dpos {dpos} != expected {expected}"


def test_ghost_labels_in_event_log_not_in_observations() -> None:
    levers = {"ghost_events": {"rate": 0.5, "kinds": ["push", "consume_food"]}}
    w = World(cfg_with_levers(levers))
    events = []
    for _ in range(500):
        obs, _ = w.step([engine.NOOP])
        # Observations carry no cause information: fixed keys, plain float arrays.
        assert set(obs[0].keys()) == {"grid", "intero"}
        assert obs[0]["grid"].dtype == np.float32
        assert obs[0]["intero"].shape == (6,)
        events.extend(w.drain_events())

    assert all(e["cause"] in ("self", "world") for e in events)
    world_events = [e for e in events if e["cause"] == "world"]
    assert any(e["type"] == "agent_moved" for e in world_events), "no ghost pushes"
    # Agent only ever chose NOOP, so any move event must be world-caused.
    assert all(e["cause"] == "world" for e in events if e["type"] == "agent_moved")


def test_region_volatility_relocates_within_rect() -> None:
    rect = [0, 0, 31, 31]
    levers = {"region_volatility": {"regions": [{"rect": rect, "interval": 100}]}}
    w = World(cfg_with_levers(levers))
    events = []
    for _ in range(250):
        w.step([engine.NOOP])
        events.extend(w.drain_events())
    moves = [e for e in events if e["type"] == "food_relocated"]
    assert moves, "no relocation events in a volatile region"
    for e in moves:
        assert e["cause"] == "world"
        x, y = e["pos"]
        assert rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def test_seasonal_shift_migrates_distribution() -> None:
    levers = {"seasonal_shift": {"interval": 100}}
    w = World(cfg_with_levers(levers))
    before = {(p["x"], p["y"]) for p in w.patches}
    events = []
    for _ in range(150):
        w.step([engine.NOOP])
        events.extend(w.drain_events())
    assert any(e["type"] == "seasonal_shift" for e in events)
    after = {(p["x"], p["y"]) for p in w.patches}
    assert before != after


def test_exogenous_reset_marker_logged() -> None:
    levers = {"exogenous_reset": {"ticks": [25]}}
    w = World(cfg_with_levers(levers))
    events = []
    for _ in range(50):
        w.step([engine.NOOP])
        events.extend(w.drain_events())
    resets = [e for e in events if e["type"] == "exogenous_reset"]
    assert len(resets) == 1 and resets[0]["tick"] == 25


def test_agent_and_ledger_never_import_levers() -> None:
    """Grep-able hard rule: nothing in agent/ or ledger/ imports levers."""
    import ast

    for pkg in ("agent", "ledger"):
        for path in (ROOT / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                assert not any("levers" in n for n in names), (
                    f"{path} imports levers: {names}"
                )
