"""Persistent recurrent core (RSSM variant) — stub, future alternative to GRUCore.

Same interface as agent.core_gru.GRUCore; adds a stochastic latent and a
world-prediction loss. Core trains only on world-prediction + task loss.
Config: the top-level ``rssm`` section (deter, stoch, embed, ensemble_k, ...).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class RSSMCore(nn.Module):
    """Recurrent state-space model core (deterministic + stochastic latent)."""

    def __init__(self, cfg: dict[str, Any], input_dim: int) -> None:
        super().__init__()
        raise NotImplementedError

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero state of shape (B, deter + stoch)."""
        raise NotImplementedError

    def forward(
        self, embed: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: (B, input_dim), (B, H) -> (output (B, H), next state (B, H))."""
        raise NotImplementedError
