"""Short soak: 50k ticks of a random agent — no exceptions, bounded memory."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from world.config import load_config
from world.engine import NUM_ACTIONS, World


@pytest.mark.slow
def test_random_soak_short_50k_ticks_no_exceptions_bounded_memory() -> None:
    cfg = load_config("configs/smoke.yaml")
    cfg["seed"] = 123
    # Levers on, to soak the full step path (ghosts + volatility).
    cfg["world"]["levers"] = {
        "ghost_events": {"rate": 0.01, "kinds": ["push", "consume_food"]},
        "region_volatility": {"regions": [{"rect": [0, 0, 15, 31], "interval": 500}]},
    }
    world = World(cfg)
    rng = np.random.default_rng(0)

    tracemalloc.start()
    checkpoints: list[int] = []
    for tick in range(50_000):
        world.step([int(rng.integers(NUM_ACTIONS))])
        events = world.drain_events()  # the trainer drains every tick too
        assert len(events) < 1000  # single tick can't produce unbounded events
        if tick % 10_000 == 9_999:
            checkpoints.append(tracemalloc.get_traced_memory()[0])
    tracemalloc.stop()

    # Bounded memory: heap at 50k ticks within 8 MB of heap at 10k ticks
    # (steady state — no per-tick accumulation leaks).
    assert checkpoints[-1] - checkpoints[0] < 8 * 1024 * 1024, checkpoints
    assert world.tick == 50_000
