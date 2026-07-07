"""Homeostatic drives + macro-plan arbiter.

The arbiter chooses among four macro-plans by scoring each plan's FORECASTED
drive error against setpoints (ledger/forecaster.py: predicted intero at the
scoring horizon, given h.detach() and the plan id), then epsilon-greedy over
scores. Plan executors are scripted primitive-action policies that read ONLY
the agent's own egocentric observation (grid window + intero) and its own
internal epistemic map estimate — never world internals or lever config
(nothing in this package may import world.levers; enforced by test).

No objective here may reference run duration, reset timing, or the training
process itself (CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Fixed action table ids (world/engine.py; kept literal here so agent/ does
# not need to import the engine — the table is part of the world contract).
_MOVE_N, _MOVE_S, _MOVE_E, _MOVE_W = 0, 1, 2, 3
_EAT, _REST, _NOOP = 4, 5, 8

PLANS: tuple[str, ...] = (
    "forage_nearest", "explore_high_epistemic", "rest", "goto_shelter"
)
NUM_PLANS = len(PLANS)

# intero vector layout: [energy, fatigue, memory_pressure, sin, cos, 1]
_ENERGY, _FATIGUE = 0, 1


def homeostatic_reward(intero: np.ndarray, cfg: dict[str, Any]) -> float:
    """Drive error as negative reward: weighted squared distance to setpoints.

    Reward from the intero vector [energy, fatigue, memory_pressure, sin, cos, 1].
    """
    return -drive_error(intero[_ENERGY], intero[_FATIGUE], cfg)


def drive_error(energy: float, fatigue: float, cfg: dict[str, Any]) -> float:
    """Weighted squared deviation of (energy, fatigue) from their setpoints.

    ``cfg`` is the ``ledger.arbiter`` config section (setpoints + weights).
    """
    set_e = cfg.get("setpoint_energy", 1.0)
    set_f = cfg.get("setpoint_fatigue", 0.0)
    w_e = cfg.get("weight_energy", 1.0)
    w_f = cfg.get("weight_fatigue", 0.25)
    return w_e * (energy - set_e) ** 2 + w_f * (fatigue - set_f) ** 2


# ------------------------------------------------------------ plan executors


def _move_toward(dx: int, dy: int, rng: np.random.Generator) -> int:
    """Greedy move toward a relative offset; ties broken randomly."""
    options = []
    if dx > 0:
        options.append(_MOVE_E)
    elif dx < 0:
        options.append(_MOVE_W)
    if dy > 0:
        options.append(_MOVE_S)
    elif dy < 0:
        options.append(_MOVE_N)
    if not options:
        return _NOOP
    return int(options[rng.integers(len(options))])


def _nearest_in_channel(grid: np.ndarray, channel: int) -> tuple[int, int] | None:
    """Relative (dx, dy) of the nearest set cell in an egocentric channel."""
    ys, xs = np.nonzero(grid[channel])
    if len(xs) == 0:
        return None
    center = grid.shape[1] // 2
    d = np.abs(xs - center) + np.abs(ys - center)
    i = int(np.argmin(d))
    return int(xs[i] - center), int(ys[i] - center)


def plan_action(
    plan: int,
    grid: np.ndarray,
    ch_food: int,
    ch_shelter: int,
    rng: np.random.Generator,
    epistemic_map: np.ndarray | None = None,
    pos: tuple[int, int] | None = None,
) -> int:
    """One primitive action for ``plan`` given the egocentric grid window.

    ``epistemic_map``/``pos`` serve explore_high_epistemic: the map is the
    agent's OWN running estimate of ensemble disagreement by position (an
    internal estimate, not privileged world state); ``pos`` is its current
    coordinate. Without them, explore degrades to a uniform random move.
    """
    name = PLANS[plan]
    if name == "rest":
        return _REST

    if name == "forage_nearest":
        rel = _nearest_in_channel(grid, ch_food)
        if rel is None:
            return int((_MOVE_N, _MOVE_S, _MOVE_E, _MOVE_W)[rng.integers(4)])
        if rel == (0, 0):
            return _EAT
        return _move_toward(rel[0], rel[1], rng)

    if name == "goto_shelter":
        rel = _nearest_in_channel(grid, ch_shelter)
        if rel is None:
            return int((_MOVE_N, _MOVE_S, _MOVE_E, _MOVE_W)[rng.integers(4)])
        if rel == (0, 0):
            return _REST
        return _move_toward(rel[0], rel[1], rng)

    # explore_high_epistemic: head toward the most uncertain nearby region of
    # the agent's own epistemic map; random walk when no map is available.
    if epistemic_map is None or pos is None:
        return int((_MOVE_N, _MOVE_S, _MOVE_E, _MOVE_W)[rng.integers(4)])
    x, y = pos
    n = epistemic_map.shape[0]
    scores = {}
    for action, (dx, dy) in ((_MOVE_N, (0, -1)), (_MOVE_S, (0, 1)),
                             (_MOVE_E, (1, 0)), (_MOVE_W, (-1, 0))):
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            scores[action] = epistemic_map[ny, nx]
    if not scores:
        return _NOOP
    best = max(scores.values())
    top = [a for a, s in scores.items() if s == best]
    return int(top[rng.integers(len(top))])


# ----------------------------------------------------------------- arbiter


class Arbiter:
    """Epsilon-greedy macro-plan selection over forecasted drive errors.

    For each plan, ask the forecaster for the intero forecast at the scoring
    horizon given h.detach(); score = -drive_error(forecast). The forecaster
    output is already gradient-free (its input is detached and selection
    happens under no_grad in the trainer), and plan choice affects the world
    only through sampled actions — no autograd path back to anything.
    """

    def __init__(self, cfg: dict[str, Any], forecaster: Any, seed: int = 0) -> None:
        """``cfg`` is the ``ledger.arbiter`` config section."""
        self.cfg = cfg
        self.forecaster = forecaster
        self.epsilon: float = cfg.get("epsilon", 0.1)
        self.horizon: int = cfg.get("score_horizon", 10)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def select_plans(self, h_detached: torch.Tensor) -> np.ndarray:
        """(B, core_dim) detached -> (B,) plan ids."""
        b = h_detached.shape[0]
        scores = np.zeros((b, NUM_PLANS))
        for plan in range(NUM_PLANS):
            onehot = torch.zeros(b, NUM_PLANS, device=h_detached.device)
            onehot[:, plan] = 1.0
            out = self.forecaster(h_detached, onehot)
            mean, _ = out[self.horizon]
            for i in range(b):
                scores[i, plan] = -drive_error(
                    float(mean[i, _ENERGY]), float(mean[i, _FATIGUE]), self.cfg
                )
        plans = scores.argmax(axis=1)
        explore = self.rng.random(b) < self.epsilon
        plans[explore] = self.rng.integers(0, NUM_PLANS, size=int(explore.sum()))
        return plans
