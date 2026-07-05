"""Baseline task reward — the single place reward shaping lives.

Terms (all from the ``ppo.reward`` config section):
- +eat            per successful eat action
- -deficit_penalty per tick while energy < deficit_threshold
- -step_cost      every tick

Per CLAUDE.md hard rules, nothing here may reference run duration, reset
timing, or the training process.
"""

from __future__ import annotations

from typing import Any

from world.engine import EAT


def compute_reward(action: int, success: bool, energy: float, rcfg: dict[str, Any]) -> float:
    """Reward for one realized transition."""
    reward = -float(rcfg["step_cost"])
    if action == EAT and success:
        reward += float(rcfg["eat"])
    if energy < float(rcfg["deficit_threshold"]):
        reward -= float(rcfg["deficit_penalty"])
    return reward
