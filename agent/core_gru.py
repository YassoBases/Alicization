"""Persistent recurrent core (GRU variant).

The core hidden state h is the agent's only persistent internal state. Ledger
heads consume h.detach() only (gradient isolation — see CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GRUCore(nn.Module):
    """Single-layer GRU cell over observation embeddings."""

    def __init__(self, cfg: dict[str, Any], input_dim: int) -> None:
        """``cfg`` is the ``model`` config section (core_hidden)."""
        super().__init__()
        self.hidden_dim: int = cfg["core_hidden"]
        self.cell = nn.GRUCell(input_dim, self.hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state of shape (B, core_hidden)."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, embed: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: (B, input_dim), (B, H) -> (output (B, H), next hidden (B, H))."""
        h_next = self.cell(embed, h)
        return h_next, h_next
