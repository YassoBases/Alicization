"""Offline consolidation phase (stub): replay-based Ledger refits between rollouts.

Scheduling is an exogenous experimental condition set in config; no loss here may
reference run duration or reset timing (CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any


def run_sleep_phase(trainer: Any, cfg: dict[str, Any]) -> dict[str, float]:
    """One consolidation pass over replay; returns scalar metrics for logging."""
    raise NotImplementedError
