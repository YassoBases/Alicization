"""Deterministic gridworld engine (stub — implemented in stage-1a).

64x64 tile grid; channels: terrain (3 types with movement cost), food, water,
shelter, mark, agent. Fixed timestep, single RNG owned by the world,
snapshot/restore, and a state hash for determinism tests.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class World:
    """The sandboxed gridworld. API is vectorized over a list of agents."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Build the world from a resolved config dict (top-level: seed, world)."""
        raise NotImplementedError

    def step(
        self, actions: Sequence[int]
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        """Advance one tick. Returns per-agent (observation, info) lists."""
        raise NotImplementedError

    def observe(self) -> list[dict[str, np.ndarray]]:
        """Current per-agent observations without advancing time."""
        raise NotImplementedError

    def snapshot(self) -> bytes:
        """Serialize all mutable state (grid, schedules, RNG state, tick)."""
        raise NotImplementedError

    def restore(self, blob: bytes) -> None:
        """Restore state produced by :meth:`snapshot`."""
        raise NotImplementedError

    def state_hash(self) -> str:
        """Hex digest of the full mutable state, for determinism tests."""
        raise NotImplementedError
