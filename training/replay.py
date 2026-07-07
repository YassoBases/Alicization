"""Prioritized sequence replay for world-model training.

Ring buffer of transitions (obs, action, reward, done), one stream per env,
with sequence sampling for RSSM training. Prioritization is proportional
(p^alpha, alpha=0.6 by default) on per-transition priorities updated from
world-model loss; new transitions enter at the current max priority so they
are sampled at least once soon.

Grids are stored as uint8 (all observation channels are binary occupancy
maps) to keep the base-config capacity (500k transitions) in the hundreds of
MB rather than GB; they are converted back to float32 on sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SequenceReplay:
    """Per-env ring buffers with proportional prioritized sequence sampling."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        grid_shape: tuple[int, ...],
        intero_dim: int,
        alpha: float = 0.6,
        seed: int = 0,
    ) -> None:
        """``capacity`` is TOTAL transitions across all env streams."""
        self.per_env = max(1, capacity // num_envs)
        self.num_envs = num_envs
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        n, cap = num_envs, self.per_env
        self.grid = np.zeros((n, cap, *grid_shape), dtype=np.uint8)
        self.intero = np.zeros((n, cap, intero_dim), dtype=np.float32)
        self.action = np.zeros((n, cap), dtype=np.int64)
        self.reward = np.zeros((n, cap), dtype=np.float32)
        self.done = np.zeros((n, cap), dtype=bool)
        self.position = np.zeros((n, cap, 2), dtype=np.float32)  # normalized
        self.priority = np.zeros((n, cap), dtype=np.float64)
        self._ptr = 0  # synchronized across envs (vecenv steps them together)
        self._filled = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._filled * self.num_envs

    def add_batch(
        self,
        grid: np.ndarray,
        intero: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        position: np.ndarray | None = None,
    ) -> None:
        """Add one synchronized tick for every env: grid (N, C, W, W) float,
        intero (N, D), action (N,), reward (N,), done (N,), position (N, 2)
        normalized post-action coordinates (optional; zeros when absent)."""
        i = self._ptr
        self.grid[:, i] = (grid > 0.5).astype(np.uint8)
        self.intero[:, i] = intero
        self.action[:, i] = action
        self.reward[:, i] = reward
        self.done[:, i] = done.astype(bool)
        self.position[:, i] = position if position is not None else 0.0
        self.priority[:, i] = self._max_priority
        self._ptr = (self._ptr + 1) % self.per_env
        self._filled = min(self._filled + 1, self.per_env)

    # ---------------------------------------------------------------- sample

    def _valid_starts(self, seq_len: int) -> np.ndarray:
        """(num_valid, 2) array of (env, start) index pairs.

        A start is valid when the whole window [start, start+L) lies in
        filled territory, does not cross the ring write pointer, and contains
        no done before its final position (an RSSM sequence must not span an
        episode boundary mid-window; a done AT the final position is fine).
        """
        cap, filled, ptr = self.per_env, self._filled, self._ptr
        if filled < seq_len:
            return np.zeros((0, 2), dtype=np.int64)
        if filled < cap:
            starts = np.arange(0, filled - seq_len + 1)
        else:
            # Full ring: valid starts must not cross ptr (oldest data boundary).
            offsets = np.arange(0, cap - seq_len + 1)
            starts = (ptr + offsets) % cap
        pairs = []
        for env in range(self.num_envs):
            done_row = self.done[env]
            for s in starts:
                idx = (s + np.arange(seq_len - 1)) % cap
                if not done_row[idx].any():
                    pairs.append((env, s))
        return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)

    def sample(
        self, batch: int, seq_len: int, device: torch.device
    ) -> dict[str, Any] | None:
        """Sample ``batch`` sequences of ``seq_len``; None if too few valid
        starts. Probability of a start is proportional to its transition
        priority^alpha."""
        pairs = self._valid_starts(seq_len)
        if len(pairs) < batch:
            return None
        prios = self.priority[pairs[:, 0], pairs[:, 1]] ** self.alpha
        probs = prios / prios.sum()
        chosen = self.rng.choice(len(pairs), size=batch, replace=False, p=probs)
        sel = pairs[chosen]

        cap = self.per_env
        idx = (sel[:, 1][:, None] + np.arange(seq_len)[None, :]) % cap  # (B, T)
        env_idx = sel[:, 0][:, None]
        to_t = lambda arr, dtype: torch.as_tensor(  # noqa: E731
            arr, dtype=dtype, device=device
        ).transpose(0, 1)  # (B, T, ...) -> (T, B, ...)
        return {
            "grid": to_t(self.grid[env_idx, idx].astype(np.float32), torch.float32),
            "intero": to_t(self.intero[env_idx, idx], torch.float32),
            "action": to_t(self.action[env_idx, idx], torch.long),
            "reward": to_t(self.reward[env_idx, idx], torch.float32),
            "done": to_t(self.done[env_idx, idx].astype(np.float32), torch.float32),
            "position": to_t(self.position[env_idx, idx], torch.float32),
            "envs": sel[:, 0],
            "starts": sel[:, 1],
        }

    def update_priorities(
        self, envs: np.ndarray, starts: np.ndarray, seq_len: int, losses: np.ndarray
    ) -> None:
        """Assign each sampled sequence's world-model loss to its transitions."""
        cap = self.per_env
        for env, start, loss in zip(envs, starts, losses):
            p = float(abs(loss)) + 1e-6
            idx = (start + np.arange(seq_len)) % cap
            self.priority[env, idx] = p
            self._max_priority = max(self._max_priority, p)

    # ----------------------------------------------------------------- state

    def state_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid, "intero": self.intero, "action": self.action,
            "reward": self.reward, "done": self.done, "priority": self.priority,
            "position": self.position,
            "ptr": self._ptr, "filled": self._filled,
            "max_priority": self._max_priority,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.grid = state["grid"]
        self.intero = state["intero"]
        self.action = state["action"]
        self.reward = state["reward"]
        self.done = state["done"]
        self.priority = state["priority"]
        if "position" in state:
            self.position = state["position"]
        self._ptr = state["ptr"]
        self._filled = state["filled"]
        self._max_priority = state["max_priority"]
        self.rng.bit_generator.state = state["rng_state"]
