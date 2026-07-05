"""Actor-critic policy over the fixed 9-action table.

Inputs: core output h plus Ledger estimates (capability, reliability,
uncertainty) as additional features. The Ledger features arrive detached.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Categorical policy head + value head over core output and ledger features."""

    def __init__(self, cfg: dict[str, Any], input_dim: int, num_actions: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """(B, input_dim) -> (action distribution, value estimate (B,))."""
        raise NotImplementedError
