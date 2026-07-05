"""Actor-critic policy over the fixed 9-action table.

Inputs: core output h (later, plus detached Ledger estimates as additional
features). Heads use the standard PPO orthogonal init (0.01 policy gain).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Categorical policy head + value head over core output features."""

    def __init__(self, cfg: dict[str, Any], input_dim: int, num_actions: int) -> None:
        super().__init__()
        del cfg  # reserved for future head config
        self.pi = nn.Linear(input_dim, num_actions)
        self.v = nn.Linear(input_dim, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """(B, input_dim) -> (action distribution, value estimate (B,))."""
        dist = torch.distributions.Categorical(logits=self.pi(features))
        value = self.v(features).squeeze(-1)
        return dist, value
