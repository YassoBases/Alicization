"""Body model: per-action capability estimates (success prob, expected costs)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class BodyModel(nn.Module):
    """Predicts per-action realized-transition stats from the detached core state."""

    def __init__(self, cfg: dict[str, Any], core_dim: int, num_actions: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, h_detached: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, core_dim) detached -> {'success_prob': (B, A), 'denergy': (B, A)}."""
        raise NotImplementedError
