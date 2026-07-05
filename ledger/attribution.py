"""Attribution head: estimates P(cause=self) for observed state changes.

Trained only against the agent's own prediction errors — never against the
ground-truth cause labels in the world event log (those are evaluation-only).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class AttributionHead(nn.Module):
    """Self-vs-world attribution over recent transitions, from detached core state."""

    def __init__(self, cfg: dict[str, Any], core_dim: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, h_detached: torch.Tensor) -> torch.Tensor:
        """(B, core_dim) detached -> (B,) logit of P(cause=self) for the last change."""
        raise NotImplementedError
