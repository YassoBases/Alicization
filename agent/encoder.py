"""Observation encoder: egocentric grid window + interoceptive vector -> embedding."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Conv encoder for the 11x11xC grid window, concatenated with intero features."""

    def __init__(self, cfg: dict[str, Any], grid_channels: int, intero_dim: int) -> None:
        """``cfg`` is the ``model`` config section (encoder_channels, obs_embed_dim)."""
        super().__init__()
        raise NotImplementedError

    def forward(self, grid: torch.Tensor, intero: torch.Tensor) -> torch.Tensor:
        """(B, C, 11, 11) grid + (B, intero_dim) -> (B, obs_embed_dim) embedding."""
        raise NotImplementedError
