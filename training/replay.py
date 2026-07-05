"""Replay buffer (stub): stores transitions for Ledger-head and sleep training."""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """Ring buffer of transitions (obs, action, realized info, hidden snapshots)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def add(self, transition: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
