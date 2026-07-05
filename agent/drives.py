"""Homeostatic drives: map interoceptive variables to per-tick reward.

No objective here may reference run duration, reset timing, or the training
process itself (CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def homeostatic_reward(intero: np.ndarray, cfg: dict[str, Any]) -> float:
    """Reward from the intero vector [energy, fatigue, memory_pressure, sin, cos, 1]."""
    raise NotImplementedError
