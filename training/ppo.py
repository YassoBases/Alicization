"""PPO trainer over the gridworld with a recurrent core.

Core trains only on world-prediction + task loss; Ledger heads train on
detached hidden states via separate optim groups (CLAUDE.md Hard rules).
"""

from __future__ import annotations

from typing import Any

import torch


class PPOTrainer:
    """Rollout collection + clipped-surrogate updates for the actor-critic."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """``cfg`` is the fully resolved config dict."""
        raise NotImplementedError

    def train(self, resume_from: str | None = None) -> None:
        """Run the training loop for cfg['ppo']['total_env_steps'] env steps."""
        raise NotImplementedError

    def collect_rollout(self) -> dict[str, torch.Tensor]:
        """Collect one rollout of length cfg['ppo']['rollout_length'] per env."""
        raise NotImplementedError

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        """One PPO update over a rollout; returns scalar metrics for logging."""
        raise NotImplementedError
