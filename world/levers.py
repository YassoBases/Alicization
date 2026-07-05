"""Config-driven experiment levers (stub — implemented in stage-1b).

Levers are injected by world config and are invisible to the agent: nothing in
agent/ or ledger/ may import this module. Ground-truth cause labels
(cause={self|world}) are for evaluation logs only and must never enter any
observation or loss.

Levers: capability_shift, ghost_events, region_volatility, seasonal_shift,
exogenous_reset. All schedules and rates come from the ``world.levers`` config
section; timing never comes from code.
"""

from __future__ import annotations

from typing import Any


class LeverEngine:
    """Applies scheduled perturbations inside World.step and logs ground truth."""

    def __init__(self, cfg: dict[str, Any] | None) -> None:
        """``cfg`` is the ``world.levers`` config section (or None: all levers off)."""
        raise NotImplementedError

    def capability(self, action: int, tick: int) -> tuple[float, float]:
        """Effective (fail_prob, energy_mult) for ``action`` at ``tick``."""
        raise NotImplementedError

    def post_step(self, world: Any) -> None:
        """Apply ghost events / volatility / seasonal shifts / reset markers."""
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        """Mutable lever state for World.snapshot."""
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`get_state`."""
        raise NotImplementedError
