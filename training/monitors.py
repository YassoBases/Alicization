"""Training-side health monitors (no gradients, no agent-visible state).

ParticipationRatioMonitor: representation-collapse detector for the RSSM
deterministic state. PR = (sum lambda)^2 / sum lambda^2 over the eigenvalues
of the state covariance in a rolling window — ranges from 1 (all variance in
one direction) to dim (isotropic). A collapse shows up as PR dropping far
below its own running max.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ParticipationRatioMonitor:
    """Rolling-window participation ratio of a state vector, computed every
    ``every_ticks`` env ticks; logs a WARNING when PR < collapse_frac * its
    running max."""

    def __init__(
        self,
        every_ticks: int = 1000,
        window: int = 1000,
        collapse_frac: float = 0.25,
        min_samples: int = 64,
    ) -> None:
        self.every_ticks = every_ticks
        self.window = window
        self.collapse_frac = collapse_frac
        self.min_samples = min_samples
        self._buf: deque[np.ndarray] = deque(maxlen=window)
        self._last_computed = 0
        self.running_max = 0.0
        self.history: list[tuple[int, float]] = []  # (env_step, pr)

    def add(self, states: np.ndarray) -> None:
        """Add a (B, D) batch of state vectors to the rolling window."""
        for row in states:
            self._buf.append(row.astype(np.float64))

    def compute(self) -> float | None:
        """Participation ratio of the current window (None if too few samples)."""
        if len(self._buf) < self.min_samples:
            return None
        x = np.stack(self._buf)
        x = x - x.mean(axis=0, keepdims=True)
        # Eigenvalues of the covariance = squared singular values / (n - 1).
        svals = np.linalg.svd(x, compute_uv=False)
        eig = svals**2 / max(x.shape[0] - 1, 1)
        total = eig.sum()
        if total <= 0:
            return 0.0
        return float(total**2 / (eig**2).sum())

    def maybe_compute(self, env_steps: int) -> float | None:
        """Compute if ``every_ticks`` have elapsed since the last computation.

        Updates the running max and logs a WARNING on collapse (PR below
        ``collapse_frac`` of the running max).
        """
        if env_steps - self._last_computed < self.every_ticks:
            return None
        pr = self.compute()
        if pr is None:
            return None
        self._last_computed = env_steps
        self.history.append((env_steps, pr))
        if self.running_max > 0 and pr < self.collapse_frac * self.running_max:
            logger.warning(
                "participation ratio collapse: PR=%.2f < %.0f%% of running max %.2f "
                "(env step %d)",
                pr, 100 * self.collapse_frac, self.running_max, env_steps,
            )
        self.running_max = max(self.running_max, pr)
        return pr

    # ------------------------------------------------------------- state

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_computed": self._last_computed,
            "running_max": self.running_max,
            "history": list(self.history),
            "window_states": np.stack(self._buf) if self._buf else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._last_computed = state["last_computed"]
        self.running_max = state["running_max"]
        self.history = [tuple(x) for x in state["history"]]
        self._buf.clear()
        if state.get("window_states") is not None:
            for row in state["window_states"]:
                self._buf.append(row)
