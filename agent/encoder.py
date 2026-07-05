"""Observation encoder: egocentric grid window + interoceptive vector -> embedding.

Architecture (channel widths from the ``agent`` config section's
``encoder_channels``): Conv(C->c1, 3x3) ReLU -> Conv(c1->c2, 3x3) ReLU ->
flatten -> Linear -> embed_dim, concat intero -> Linear -> embed_dim. Convs are
unpadded, so an 11x11 window shrinks to 7x7 before flattening.

``embed_dim`` is a constructor argument rather than a config key: by
convention the caller sets it equal to the recurrent core's ``hidden_size``
(see training/ppo.py's PPOModel), so the observation embedding and the GRU
input share one width knob instead of two redundant ones.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Conv encoder for the (C, W, W) grid window fused with intero features."""

    def __init__(
        self,
        cfg: dict[str, Any],
        grid_channels: int,
        intero_dim: int,
        embed_dim: int,
        window: int = 11,
    ) -> None:
        """``cfg`` is the ``agent`` config section (encoder_channels)."""
        super().__init__()
        c1, c2 = cfg["encoder_channels"]
        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, c1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = c2 * (window - 4) ** 2
        self.fc_grid = nn.Linear(conv_out, embed_dim)
        self.fc_fuse = nn.Linear(embed_dim + intero_dim, embed_dim)

    def forward(self, grid: torch.Tensor, intero: torch.Tensor) -> torch.Tensor:
        """(B, C, W, W) grid + (B, intero_dim) -> (B, embed_dim) embedding."""
        x = torch.relu(self.fc_grid(self.conv(grid)))
        return torch.relu(self.fc_fuse(torch.cat([x, intero], dim=-1)))
