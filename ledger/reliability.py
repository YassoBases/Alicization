"""Memory-reliability head (stub): estimates trustworthiness of recalled memories."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ReliabilityHead(nn.Module):
    """Scores episodic recalls with an expected-staleness / reliability estimate."""

    def __init__(self, cfg: dict[str, Any], core_dim: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, h_detached: torch.Tensor, recall_age: torch.Tensor) -> torch.Tensor:
        """(B, core_dim) detached + (B,) recall age -> (B,) reliability in [0, 1]."""
        raise NotImplementedError
