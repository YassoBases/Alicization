"""Synchronized batch of world instances for vectorized rollouts.

Episodes are trainer-side, config-driven time limits (``ppo.episode_length``):
the world itself is a continuing environment, so at each boundary the env slot
is rebuilt with a fresh deterministic seed and ``done=True`` is reported. The
observation returned for a done step is the first observation of the new
episode (standard reset-on-done convention).

``get_state``/``set_state`` capture exact world snapshots so training can
resume bit-identically.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from training.reward import compute_reward
from world.engine import World


class VecWorld:
    """N synchronized worlds stepped together; all arrays are numpy, batch-first."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.num_envs: int = cfg["ppo"]["num_envs"]
        self.episode_length: int = cfg["ppo"]["episode_length"]
        self.reward_cfg: dict[str, Any] = cfg["ppo"]["reward"]
        self._seed_counter = 0
        self.worlds: list[World] = [self._make_world() for _ in range(self.num_envs)]
        self.ep_steps = np.zeros(self.num_envs, dtype=np.int64)
        probe = self.worlds[0].observe()[0]
        self.grid_shape: tuple[int, ...] = probe["grid"].shape  # (C, W, W)
        self.intero_dim: int = probe["intero"].shape[0]

    def _make_world(self) -> World:
        cfg = copy.deepcopy(self.cfg)
        cfg["seed"] = int(self.cfg["seed"]) * 100_000 + self._seed_counter
        self._seed_counter += 1
        return World(cfg)

    def observe(self) -> dict[str, np.ndarray]:
        """Batched observations: {'grid': (N, C, W, W), 'intero': (N, D)}."""
        per_env = [w.observe()[0] for w in self.worlds]
        return {
            "grid": np.stack([o["grid"] for o in per_env]),
            "intero": np.stack([o["intero"] for o in per_env]),
        }

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step every env. Returns (obs, rewards (N,), dones (N,), infos)."""
        assert actions.shape == (self.num_envs,)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        infos: list[dict[str, Any]] = []
        for i, world in enumerate(self.worlds):
            action = int(actions[i])
            obs_i, info_i = world.step([action])
            events = world.drain_events()  # ground truth; evaluation-only, see infos["events"]
            rewards[i] = compute_reward(
                action=action,
                success=info_i[0]["realized"]["success"],
                energy=float(obs_i[0]["intero"][0]),
                rcfg=self.reward_cfg,
            )
            infos.append({**info_i[0], "events": events})
            self.ep_steps[i] += 1
            if self.ep_steps[i] >= self.episode_length:
                dones[i] = 1.0
                self.worlds[i] = self._make_world()
                self.ep_steps[i] = 0
        return self.observe(), rewards, dones, infos

    # ------------------------------------------------------- exact save/load

    def get_state(self) -> dict[str, Any]:
        """Everything needed to resume stepping bit-identically."""
        return {
            "seeds": [w.cfg["seed"] for w in self.worlds],
            "snapshots": [w.snapshot() for w in self.worlds],
            "ep_steps": self.ep_steps.copy(),
            "seed_counter": self._seed_counter,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._seed_counter = state["seed_counter"]
        self.worlds = []
        for seed, blob in zip(state["seeds"], state["snapshots"]):
            cfg = copy.deepcopy(self.cfg)
            cfg["seed"] = seed
            world = World(cfg)
            world.restore(blob)
            self.worlds.append(world)
        self.ep_steps = np.asarray(state["ep_steps"]).copy()
