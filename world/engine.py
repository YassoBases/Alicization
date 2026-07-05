"""Deterministic gridworld engine.

64x64 tile grid. Tile channels: terrain (3 types with movement cost), food,
water, shelter, mark, agent. Fixed-timestep ``step(actions)``; the API is
vectorized over a list of agents (currently one). A single numpy Generator owned
by the world provides all stochasticity. ``snapshot``/``restore`` cover every
piece of mutable state and ``state_hash`` digests it for determinism tests.

Time: ``tick % day_length`` is time-of-day; the last ``1 - night_start_frac``
fraction of the day is night, during which agents off shelter tiles pay an
extra energy drain. Food regrows on exogenous per-patch schedules defined only
in world config.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import pickle
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from world.levers import LeverEngine

# Fixed action table.
MOVE_N, MOVE_S, MOVE_E, MOVE_W = 0, 1, 2, 3
EAT, REST, PLACE_MARK, ERASE_MARK, NOOP = 4, 5, 6, 7, 8
NUM_ACTIONS = 9
ACTION_NAMES = (
    "move_n", "move_s", "move_e", "move_w",
    "eat", "rest", "place_mark", "erase_mark", "noop",
)
# (dx, dy) per move action; y grows southward.
_MOVE_DELTA = {MOVE_N: (0, -1), MOVE_S: (0, 1), MOVE_E: (1, 0), MOVE_W: (-1, 0)}


@dataclass
class AgentState:
    """Mutable per-agent state."""

    x: int
    y: int
    energy: float
    fatigue: float


class World:
    """The sandboxed gridworld."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Build the world from a resolved config dict (top-level: seed, world)."""
        self.cfg = cfg
        wcfg = cfg["world"]
        self.size: int = wcfg["grid_size"]
        self.day_length: int = wcfg["day_length"]
        self.night_start: int = int(wcfg["night_start_frac"] * self.day_length)
        self.window: int = wcfg["window"]
        assert self.window % 2 == 1, "observation window must be odd"
        self._pad = self.window // 2

        tcfg = wcfg["terrain"]
        self.num_terrain_types: int = tcfg["num_types"]
        self.terrain_move_cost = np.asarray(tcfg["move_cost"], dtype=np.float64)
        self.num_channels: int = self.num_terrain_types + 5  # food,water,shelter,mark,agent
        self._ch_food = self.num_terrain_types
        self._ch_water = self.num_terrain_types + 1
        self._ch_shelter = self.num_terrain_types + 2
        self._ch_mark = self.num_terrain_types + 3
        self._ch_agent = self.num_terrain_types + 4

        self.ecfg = wcfg["energy"]
        self.fcfg = wcfg["fatigue"]

        self.rng = np.random.default_rng(cfg["seed"])
        n = self.size

        # --- immutable layout (regenerated identically from the seed) ---
        self.terrain = self.rng.choice(
            self.num_terrain_types, size=(n, n), p=tcfg["frequencies"]
        ).astype(np.int8)
        special = self.rng.choice(
            n * n,
            size=wcfg["water"]["num_tiles"]
            + wcfg["shelter"]["num_tiles"]
            + wcfg["food"]["num_patches"],
            replace=False,
        )
        n_water = wcfg["water"]["num_tiles"]
        n_shelter = wcfg["shelter"]["num_tiles"]
        self.water = np.zeros((n, n), dtype=bool)
        self.water.flat[special[:n_water]] = True
        self.shelter = np.zeros((n, n), dtype=bool)
        self.shelter.flat[special[n_water : n_water + n_shelter]] = True

        # --- mutable state ---
        self.tick: int = 0
        lo, hi = wcfg["food"]["regrow_interval_range"]
        patch_cells = special[n_water + n_shelter :]
        # Patch schedule state is exogenous: defined only here, from config.
        self.patches: list[dict[str, int]] = [
            {
                "x": int(c % n),
                "y": int(c // n),
                "interval": int(self.rng.integers(lo, hi + 1)),
            }
            for c in patch_cells
        ]
        self.food = np.zeros((n, n), dtype=bool)
        for p in self.patches:
            self.food[p["y"], p["x"]] = True
        self._regrow_heap: list[tuple[int, int]] = []  # (due_tick, patch_idx)
        self.mark = np.zeros((n, n), dtype=bool)

        start = self.rng.integers(0, n, size=2)
        self.agents: list[AgentState] = [
            AgentState(
                x=int(start[0]),
                y=int(start[1]),
                energy=float(self.ecfg["init"]),
                fatigue=float(self.fcfg["init"]),
            )
        ]
        self._events: list[dict[str, Any]] = []

        self.levers: LeverEngine | None = (
            LeverEngine(wcfg["levers"]) if wcfg.get("levers") else None
        )

        self._build_channel_buffer()

    # ------------------------------------------------------------------ setup

    def _build_channel_buffer(self) -> None:
        """(Re)build the padded per-channel float buffer used for observations."""
        n, p = self.size, self._pad
        buf = np.zeros((self.num_channels, n + 2 * p, n + 2 * p), dtype=np.float32)
        inner = (slice(None), slice(p, p + n), slice(p, p + n))
        for t in range(self.num_terrain_types):
            buf[t, p : p + n, p : p + n] = self.terrain == t
        buf[self._ch_water, p : p + n, p : p + n] = self.water
        buf[self._ch_shelter, p : p + n, p : p + n] = self.shelter
        buf[self._ch_food, p : p + n, p : p + n] = self.food
        buf[self._ch_mark, p : p + n, p : p + n] = self.mark
        for a in self.agents:
            buf[self._ch_agent, p + a.y, p + a.x] = 1.0
        del inner
        self._buf = buf

    # ------------------------------------------------------- state mutation

    def _set_cell(self, channel: int, x: int, y: int, value: bool) -> None:
        p = self._pad
        self._buf[channel, p + y, p + x] = float(value)

    def set_food(self, x: int, y: int, present: bool) -> None:
        """Test/experiment utility; keeps the observation buffer in sync."""
        self.food[y, x] = present
        self._set_cell(self._ch_food, x, y, present)

    def set_mark(self, x: int, y: int, present: bool) -> None:
        """Test/experiment utility; keeps the observation buffer in sync."""
        self.mark[y, x] = present
        self._set_cell(self._ch_mark, x, y, present)

    def set_agent_pos(self, agent_idx: int, x: int, y: int) -> None:
        """Test/experiment utility; keeps the observation buffer in sync."""
        a = self.agents[agent_idx]
        self._set_cell(self._ch_agent, a.x, a.y, False)
        a.x, a.y = x, y
        self._set_cell(self._ch_agent, x, y, True)

    def log_event(self, type_: str, cause: str, **fields: Any) -> None:
        """Append a ground-truth event record (cause is 'self' or 'world').

        Cause labels are for evaluation logs only; they never enter observations.
        """
        self._events.append({"tick": self.tick, "type": type_, "cause": cause, **fields})

    def drain_events(self) -> list[dict[str, Any]]:
        """Return and clear the buffered ground-truth event records."""
        out = self._events
        self._events = []
        return out

    def consume_food(self, x: int, y: int, cause: str, agent: int | None = None) -> bool:
        """Remove food at (x, y) if present and schedule its patch regrowth."""
        if not self.food[y, x]:
            return False
        self.set_food(x, y, False)
        for idx, patch in enumerate(self.patches):
            if patch["x"] == x and patch["y"] == y:
                heapq.heappush(self._regrow_heap, (self.tick + patch["interval"], idx))
                break
        self.log_event("food_consumed", cause, pos=[x, y], agent=agent)
        return True

    def move_agent(self, agent_idx: int, dx: int, dy: int, cause: str) -> bool:
        """Displace an agent by (dx, dy) if in bounds; logs with ground-truth cause."""
        a = self.agents[agent_idx]
        nx, ny = a.x + dx, a.y + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return False
        self.set_agent_pos(agent_idx, nx, ny)
        self.log_event("agent_moved", cause, pos=[nx, ny], agent=agent_idx, dpos=[dx, dy])
        return True

    # --------------------------------------------------------------- stepping

    def step(
        self, actions: Sequence[int]
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        """Advance one tick. Returns per-agent (observation, info) lists.

        Illegal actions are no-ops with success=False. Each info dict carries
        the realized transition: {'dpos': (dx, dy), 'denergy': float,
        'success': bool} plus 'pos', 'action', 'tick'.
        """
        if len(actions) != len(self.agents):
            raise ValueError(f"expected {len(self.agents)} actions, got {len(actions)}")

        before = [(a.x, a.y, a.energy) for a in self.agents]
        successes = [self._apply_action(i, int(act)) for i, act in enumerate(actions)]

        if self.levers is not None:
            self.levers.post_step(self)

        # Exogenous food regrowth.
        while self._regrow_heap and self._regrow_heap[0][0] <= self.tick:
            _, idx = heapq.heappop(self._regrow_heap)
            patch = self.patches[idx]
            if not self.food[patch["y"], patch["x"]]:
                self.set_food(patch["x"], patch["y"], True)
                self.log_event("food_regrown", "world", pos=[patch["x"], patch["y"]])

        # Metabolic drains.
        night = (self.tick % self.day_length) >= self.night_start
        emax = self.ecfg["max"]
        for a in self.agents:
            a.energy -= self.ecfg["base_drain"]
            if night and not self.shelter[a.y, a.x]:
                a.energy -= self.ecfg["night_drain"]
            a.energy = min(max(a.energy, 0.0), emax)

        self.tick += 1

        infos = [
            {
                "tick": self.tick,
                "action": int(act),
                "pos": (a.x, a.y),
                "realized": {
                    "dpos": (a.x - bx, a.y - by),
                    "denergy": a.energy - be,
                    "success": ok,
                },
            }
            for a, act, ok, (bx, by, be) in zip(self.agents, actions, successes, before)
        ]
        return self.observe(), infos

    def _apply_action(self, agent_idx: int, action: int) -> bool:
        """Apply one agent action; returns success. Illegal actions cost nothing."""
        if not 0 <= action < NUM_ACTIONS:
            return False
        a = self.agents[agent_idx]

        fail_prob, energy_mult = 0.0, 1.0
        if self.levers is not None:
            fail_prob, energy_mult = self.levers.capability(action, self.tick)

        if action in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[action]
            nx, ny = a.x + dx, a.y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return False  # illegal: no cost, no effect
            cost = self.ecfg["move_cost"] * self.terrain_move_cost[self.terrain[ny, nx]]
            a.energy = max(a.energy - cost * energy_mult, 0.0)
            a.fatigue = min(a.fatigue + self.fcfg["move_gain"], self.fcfg["max"])
            if fail_prob > 0.0 and self.rng.random() < fail_prob:
                return False  # attempted but failed: cost paid, no movement
            return self.move_agent(agent_idx, dx, dy, "self")

        if action == EAT:
            ok = self.consume_food(a.x, a.y, "self", agent=agent_idx)
            if ok:
                a.energy = min(a.energy + self.ecfg["eat_gain"], self.ecfg["max"])
            return ok

        if action == REST:
            a.fatigue = max(a.fatigue - self.fcfg["rest_recovery"], 0.0)
            return True

        if action == PLACE_MARK:
            if self.mark[a.y, a.x]:
                return False
            self.set_mark(a.x, a.y, True)
            self.log_event("mark_placed", "self", pos=[a.x, a.y], agent=agent_idx)
            return True

        if action == ERASE_MARK:
            if not self.mark[a.y, a.x]:
                return False
            self.set_mark(a.x, a.y, False)
            self.log_event("mark_erased", "self", pos=[a.x, a.y], agent=agent_idx)
            return True

        return True  # NOOP

    # ------------------------------------------------------------ observation

    def observe(self) -> list[dict[str, np.ndarray]]:
        """Per-agent observations: egocentric one-hot 'grid' + 'intero' vector."""
        tod = (self.tick % self.day_length) / self.day_length
        angle = 2.0 * math.pi * tod
        sin_t, cos_t = math.sin(angle), math.cos(angle)
        out = []
        for a in self.agents:
            grid = self._buf[:, a.y : a.y + self.window, a.x : a.x + self.window].copy()
            intero = np.array(
                [a.energy, a.fatigue, 0.0, sin_t, cos_t, 1.0], dtype=np.float32
            )
            out.append({"grid": grid, "intero": intero})
        return out

    # ------------------------------------------------------- snapshot/restore

    def _mutable_state(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "agents": [(a.x, a.y, a.energy, a.fatigue) for a in self.agents],
            "food": self.food.copy(),
            "mark": self.mark.copy(),
            "patches": [dict(p) for p in self.patches],
            "regrow_heap": list(self._regrow_heap),
            "events": [dict(e) for e in self._events],
            "rng_state": self.rng.bit_generator.state,
            "levers": self.levers.get_state() if self.levers is not None else None,
        }

    def snapshot(self) -> bytes:
        """Serialize all mutable state (grid, schedules, RNG state, tick)."""
        return pickle.dumps(self._mutable_state(), protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        """Restore state produced by :meth:`snapshot`."""
        s = pickle.loads(blob)
        self.tick = s["tick"]
        self.agents = [AgentState(x, y, e, f) for x, y, e, f in s["agents"]]
        self.food = s["food"].copy()
        self.mark = s["mark"].copy()
        self.patches = [dict(p) for p in s["patches"]]
        self._regrow_heap = list(s["regrow_heap"])
        self._events = [dict(e) for e in s["events"]]
        self.rng.bit_generator.state = s["rng_state"]
        if self.levers is not None and s["levers"] is not None:
            self.levers.set_state(s["levers"])
        self._build_channel_buffer()

    def state_hash(self) -> str:
        """Hex digest of the full mutable state, for determinism tests."""
        s = self._mutable_state()
        h = hashlib.sha256()
        h.update(struct.pack("<q", s["tick"]))
        for x, y, e, f in s["agents"]:
            h.update(struct.pack("<qqdd", x, y, e, f))
        h.update(s["food"].tobytes())
        h.update(s["mark"].tobytes())
        meta = {
            "patches": s["patches"],
            "regrow_heap": sorted(s["regrow_heap"]),
            "events": s["events"],
            "rng_state": s["rng_state"],
            "levers": s["levers"],
        }
        h.update(json.dumps(meta, sort_keys=True, default=repr).encode("utf-8"))
        return h.hexdigest()
