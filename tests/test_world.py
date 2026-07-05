"""Determinism, snapshot/restore, and action-legality tests for world.engine."""

from __future__ import annotations

import numpy as np
import pytest

from world import engine
from world.config import load_config
from world.engine import NUM_ACTIONS, World

CONFIG = "configs/base.yaml"


def make_world(seed: int = 0) -> World:
    cfg = load_config(CONFIG)
    cfg["seed"] = seed
    return World(cfg)


def action_sequence(n: int, seed: int = 123) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(a) for a in rng.integers(0, NUM_ACTIONS, size=n)]


def test_same_seed_same_actions_identical_hash() -> None:
    actions = action_sequence(10_000)
    w1, w2 = make_world(seed=7), make_world(seed=7)
    assert w1.state_hash() == w2.state_hash()
    for a in actions:
        w1.step([a])
        w2.step([a])
    assert w1.state_hash() == w2.state_hash()


def test_different_seed_differs() -> None:
    assert make_world(seed=0).state_hash() != make_world(seed=1).state_hash()


def test_snapshot_restore_replay_identical_hash() -> None:
    w = make_world(seed=3)
    for a in action_sequence(5_000, seed=1):
        w.step([a])
    blob = w.snapshot()
    tail = action_sequence(1_000, seed=2)
    for a in tail:
        w.step([a])
    hash_first = w.state_hash()

    w.restore(blob)
    for a in tail:
        w.step([a])
    assert w.state_hash() == hash_first


def test_observation_spec() -> None:
    w = make_world()
    obs, infos = w.step([engine.NOOP])
    assert len(obs) == len(infos) == 1
    grid, intero = obs[0]["grid"], obs[0]["intero"]
    assert grid.shape == (w.num_channels, w.window, w.window)
    assert grid.dtype == np.float32
    assert intero.shape == (6,)
    assert intero[2] == 0.0  # memory_pressure placeholder
    assert intero[5] == 1.0  # bias
    assert set(obs[0].keys()) == {"grid", "intero"}


class TestLegalityMatrix:
    """Action-table legality per spec; illegal actions are no-ops, success=False."""

    def success_of(self, w: World, action: int) -> bool:
        _, infos = w.step([action])
        return infos[0]["realized"]["success"]

    def test_moves_legal_in_interior(self) -> None:
        for action in (engine.MOVE_N, engine.MOVE_S, engine.MOVE_E, engine.MOVE_W):
            w = make_world()
            w.set_agent_pos(0, 32, 32)
            assert self.success_of(w, action) is True

    @pytest.mark.parametrize(
        "pos,action,legal",
        [
            ((0, 0), engine.MOVE_N, False),
            ((0, 0), engine.MOVE_W, False),
            ((0, 0), engine.MOVE_S, True),
            ((0, 0), engine.MOVE_E, True),
            ((63, 63), engine.MOVE_S, False),
            ((63, 63), engine.MOVE_E, False),
            ((63, 63), engine.MOVE_N, True),
            ((63, 63), engine.MOVE_W, True),
        ],
    )
    def test_moves_at_corners(self, pos: tuple[int, int], action: int, legal: bool) -> None:
        w = make_world()
        w.set_agent_pos(0, *pos)
        assert self.success_of(w, action) is legal
        if not legal:  # illegal move is a full no-op on position
            assert (w.agents[0].x, w.agents[0].y) == pos

    def test_eat_requires_food(self) -> None:
        w = make_world()
        w.set_agent_pos(0, 10, 10)
        w.set_food(10, 10, False)
        assert self.success_of(w, engine.EAT) is False
        w.set_food(10, 10, True)
        assert self.success_of(w, engine.EAT) is True
        assert not w.food[10, 10]  # consumed
        assert self.success_of(w, engine.EAT) is False  # gone now

    def test_mark_place_erase(self) -> None:
        w = make_world()
        w.set_agent_pos(0, 5, 5)
        w.set_mark(5, 5, False)
        assert self.success_of(w, engine.PLACE_MARK) is True
        assert self.success_of(w, engine.PLACE_MARK) is False  # already marked
        assert self.success_of(w, engine.ERASE_MARK) is True
        assert self.success_of(w, engine.ERASE_MARK) is False  # nothing to erase

    def test_rest_and_noop_always_legal(self) -> None:
        w = make_world()
        assert self.success_of(w, engine.REST) is True
        assert self.success_of(w, engine.NOOP) is True

    def test_out_of_range_action_is_illegal_noop(self) -> None:
        w = make_world()
        pos_before = (w.agents[0].x, w.agents[0].y)
        assert self.success_of(w, 99) is False
        assert (w.agents[0].x, w.agents[0].y) == pos_before


def test_realized_info_reports_dpos_and_denergy() -> None:
    w = make_world()
    w.set_agent_pos(0, 32, 32)
    e0 = w.agents[0].energy
    _, infos = w.step([engine.MOVE_E])
    realized = infos[0]["realized"]
    assert realized["dpos"] == (1, 0)
    assert realized["success"] is True
    assert realized["denergy"] == pytest.approx(w.agents[0].energy - e0)
    assert realized["denergy"] < 0


def test_food_regrows_on_schedule() -> None:
    w = make_world()
    patch = w.patches[0]
    x, y, interval = patch["x"], patch["y"], patch["interval"]
    w.set_agent_pos(0, x, y)
    w.step([engine.EAT])
    assert not w.food[y, x]
    for _ in range(interval + 1):
        w.step([engine.NOOP])
    assert w.food[y, x]
