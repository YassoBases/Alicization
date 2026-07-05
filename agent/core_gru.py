"""Persistent recurrent core (GRU variant).

The core hidden state h is the agent's only persistent internal state. Ledger
heads consume h.detach() only (gradient isolation — see CLAUDE.md Hard rules).

Supports ``gru_layers`` > 1 by stacking plain GRUCells. The externally visible
hidden state is a single flat ``(B, hidden_size * gru_layers)`` tensor (one
layer's state concatenated after another) so the rollout buffer, BPTT replay,
and checkpoint format never need to know about layer structure; ``output_dim``
(always ``hidden_size``, the top layer's width) is what feeds the policy/value
heads. With the default ``gru_layers=1`` this is bit-identical to a single
GRUCell.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GRUCore(nn.Module):
    """One or more stacked GRUCells over observation embeddings."""

    def __init__(self, cfg: dict[str, Any], input_dim: int) -> None:
        """``cfg`` is the ``agent`` config section (hidden_size, gru_layers)."""
        super().__init__()
        self.layer_size: int = cfg["hidden_size"]
        self.num_layers: int = cfg.get("gru_layers", 1)
        self.hidden_dim: int = self.layer_size * self.num_layers  # flat state size
        self.output_dim: int = self.layer_size
        self.cells = nn.ModuleList(
            nn.GRUCell(input_dim if i == 0 else self.layer_size, self.layer_size)
            for i in range(self.num_layers)
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state of shape (B, hidden_size * gru_layers)."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, embed: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: (B, input_dim), (B, hidden_dim) ->
        (top-layer output (B, output_dim), next flat hidden (B, hidden_dim))."""
        h_layers = h.reshape(h.shape[0], self.num_layers, self.layer_size)
        x = embed
        next_layers = []
        for i, cell in enumerate(self.cells):
            x = cell(x, h_layers[:, i])
            next_layers.append(x)
        h_next = torch.stack(next_layers, dim=1).reshape(h.shape[0], self.hidden_dim)
        return x, h_next
