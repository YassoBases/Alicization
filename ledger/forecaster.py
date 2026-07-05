"""Forecaster head (stub): predicts future interoceptive state and capability drift.

Every metric that evaluates a forecast must report an identity-predictor baseline
(CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Forecaster(nn.Module):
    """K-step-ahead forecasts of intero variables from the detached core state."""

    def __init__(self, cfg: dict[str, Any], core_dim: int, intero_dim: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, h_detached: torch.Tensor, horizon: int) -> torch.Tensor:
        """(B, core_dim) detached -> (B, horizon, intero_dim) forecast."""
        raise NotImplementedError


def identity_baseline(intero: torch.Tensor, horizon: int) -> torch.Tensor:
    """Baseline forecast: repeat the current intero vector for ``horizon`` steps."""
    raise NotImplementedError
