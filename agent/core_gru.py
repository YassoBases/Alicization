"""Persistent recurrent core (GRU variant).

The core hidden state h is the agent's only persistent internal state. Ledger
heads consume h.detach() only (gradient isolation — see CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GRUCore(nn.Module):
    """Single-layer GRU core over observation embeddings."""

    def __init__(self, cfg: dict[str, Any], input_dim: int) -> None:
        """``cfg`` is the ``model`` config section (core_hidden)."""
        super().__init__()
        raise NotImplementedError

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state of shape (B, core_hidden)."""
        raise NotImplementedError

    def forward(
        self, embed: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: (B, input_dim), (B, H) -> (output (B, H), next hidden (B, H))."""
        raise NotImplementedError
