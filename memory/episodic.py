"""Episodic memory (stub): bounded store of embeddings with age-based decay.

Occupancy feeds the intero ``memory_pressure`` slot (currently a placeholder 0).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class EpisodicMemory:
    """Fixed-capacity key/value store over core embeddings."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def write(self, key: np.ndarray, value: np.ndarray, tick: int) -> None:
        """Insert an entry, evicting per the configured policy when full."""
        raise NotImplementedError

    def read(self, query: np.ndarray, k: int) -> list[tuple[np.ndarray, int]]:
        """k-nearest entries as (value, age_in_ticks) pairs."""
        raise NotImplementedError

    def pressure(self) -> float:
        """Occupancy in [0, 1] for the intero memory_pressure slot."""
        raise NotImplementedError
