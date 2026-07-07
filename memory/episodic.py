"""Episodic memory: fixed-capacity surprise-gated store with spatial retrieval.

Entries: (compressed latent [frozen random linear projection of the core
state to ``latent_dim``], position, tick, surprise_at_write, and a local
food/water window summary used by ledger/reliability.py's verification).

Writes are gated by KL-surprise (RSSM posterior-vs-prior KL at that tick)
against a running threshold controlled toward a target write rate
(~1/200 ticks): the threshold multiplicatively adapts on every tick from an
EMA of the realized write rate, so the gate self-corrects as the world model
sharpens and surprise shrinks.

Retrieval: top-k by ``cos_sim(latent) * w_sim + spatial_kernel(pos) *
w_spatial`` (Gaussian kernel), optionally multiplied by a per-entry
reliability score (stage-5b). The retrieved summary vector (mean of top-k
latents) is concatenated DETACHED into the policy input — memory can inform
the policy but no gradient flows through it.

``pressure()`` (fill fraction) feeds the intero ``memory_pressure`` slot.
Pruning (during sleep, or forced on write when full): drop the lowest
``importance = surprise * exp(-age / tau)``.

One instance per env: each vecenv slot is a different world, and an episode
boundary rebuilds the world, so the trainer clears that env's memory then.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SurpriseWriteGate:
    """Threshold controller targeting a fixed write rate.

    Pure and separately testable: ``update(surprise) -> bool`` decides one
    tick. The threshold moves each tick by ``exp(eta * (rate_ema - target))``
    scaled by 1/target, so over- (under-) writing raises (lowers) it
    regardless of the surprise scale — no assumptions about the KL's units.
    """

    def __init__(
        self,
        target_rate: float = 1.0 / 200.0,
        eta: float = 0.05,
        rate_ema_decay: float = 0.995,
        init_threshold: float = 1.0,
    ) -> None:
        self.target_rate = target_rate
        self.eta = eta
        self.rate_ema_decay = rate_ema_decay
        self.threshold = init_threshold
        self.rate_ema = target_rate  # start at target: no initial transient

    def update(self, surprise: float) -> bool:
        write = surprise > self.threshold
        self.rate_ema = (
            self.rate_ema_decay * self.rate_ema
            + (1.0 - self.rate_ema_decay) * float(write)
        )
        # Multiplicative, scale-free adjustment toward the target rate.
        self.threshold *= float(
            np.exp(self.eta * (self.rate_ema - self.target_rate) / self.target_rate)
        )
        return write

    def state_dict(self) -> dict[str, float]:
        return {"threshold": self.threshold, "rate_ema": self.rate_ema}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.threshold = state["threshold"]
        self.rate_ema = state["rate_ema"]


class EpisodicMemory:
    """Fixed-capacity latent store with surprise gating and spatial retrieval."""

    def __init__(self, cfg: dict[str, Any], core_dim: int, seed: int = 0) -> None:
        """``cfg`` is the ``memory`` config section."""
        self.capacity: int = cfg.get("capacity", 2000)
        self.latent_dim: int = cfg.get("latent_dim", 32)
        self.k: int = cfg.get("retrieve_k", 4)
        self.w_sim: float = cfg.get("w_sim", 1.0)
        self.w_spatial: float = cfg.get("w_spatial", 1.0)
        self.spatial_sigma: float = cfg.get("spatial_sigma", 8.0)
        self.importance_tau: float = cfg.get("importance_tau", 20000.0)
        self.gate = SurpriseWriteGate(
            target_rate=cfg.get("write_rate_target", 1.0 / 200.0),
            eta=cfg.get("gate_eta", 0.05),
            rate_ema_decay=cfg.get("gate_rate_ema_decay", 0.995),
            init_threshold=cfg.get("gate_init_threshold", 1.0),
        )
        # Frozen random projection (seeded): compression, not representation
        # learning — nothing backpropagates through memory.
        rng = np.random.default_rng(seed)
        self.projection = (
            rng.standard_normal((core_dim, self.latent_dim)) / np.sqrt(core_dim)
        ).astype(np.float32)

        self.latents = np.zeros((self.capacity, self.latent_dim), dtype=np.float32)
        self.positions = np.zeros((self.capacity, 2), dtype=np.int64)
        self.ticks = np.zeros(self.capacity, dtype=np.int64)
        self.surprises = np.zeros(self.capacity, dtype=np.float32)
        # Local observation summary at write time (food/water bitmaps of the
        # ego window) for stage-5b verification; list of small dicts.
        self.summaries: list[dict[str, np.ndarray] | None] = [None] * self.capacity
        self.revisit_counts = np.zeros(self.capacity, dtype=np.int64)
        self.size = 0

    # ---------------------------------------------------------------- writes

    def project(self, core_state: np.ndarray) -> np.ndarray:
        """(core_dim,) -> (latent_dim,) compressed latent."""
        return core_state.astype(np.float32) @ self.projection

    def maybe_write(
        self,
        core_state: np.ndarray,
        pos: tuple[int, int],
        tick: int,
        surprise: float,
        summary: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """Gate on surprise; store if the gate opens. Returns written?"""
        if not self.gate.update(surprise):
            return False
        self._insert(self.project(core_state), pos, tick, surprise, summary)
        return True

    def _insert(
        self,
        latent: np.ndarray,
        pos: tuple[int, int],
        tick: int,
        surprise: float,
        summary: dict[str, np.ndarray] | None,
    ) -> None:
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            i = int(np.argmin(self.importance(tick)))  # replace least important
        self.latents[i] = latent
        self.positions[i] = pos
        self.ticks[i] = tick
        self.surprises[i] = surprise
        self.summaries[i] = summary
        self.revisit_counts[i] = 0

    # ------------------------------------------------------------- retrieval

    def scores(
        self,
        query_latent: np.ndarray,
        pos: tuple[int, int],
        reliability_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Combined retrieval score for every stored entry (size,)."""
        lat = self.latents[: self.size]
        q = query_latent / (np.linalg.norm(query_latent) + 1e-8)
        norms = np.linalg.norm(lat, axis=1) + 1e-8
        cos = (lat @ q) / norms
        d2 = ((self.positions[: self.size] - np.asarray(pos)) ** 2).sum(axis=1)
        kernel = np.exp(-d2 / (2.0 * self.spatial_sigma**2))
        score = self.w_sim * cos + self.w_spatial * kernel
        if reliability_fn is not None:
            score = score * reliability_fn(np.arange(self.size))
        return score

    def retrieve(
        self,
        core_state: np.ndarray,
        pos: tuple[int, int],
        reliability_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k summary vector (latent_dim,) + the chosen entry indices.

        Empty memory -> zero vector (the policy learns to ignore it early on).
        """
        if self.size == 0:
            return np.zeros(self.latent_dim, dtype=np.float32), np.zeros(0, dtype=np.int64)
        score = self.scores(self.project(core_state), pos, reliability_fn)
        k = min(self.k, self.size)
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top])]
        return self.latents[top].mean(axis=0), top

    # -------------------------------------------------------------- pruning

    def importance(self, now_tick: int) -> np.ndarray:
        """surprise * recency-decay for every stored entry (size,)."""
        age = now_tick - self.ticks[: self.size]
        return self.surprises[: self.size] * np.exp(-age / self.importance_tau)

    def prune(self, now_tick: int, keep_fraction: float = 0.9) -> int:
        """Sleep-time pruning: when full, drop the lowest-importance entries
        down to ``keep_fraction`` of capacity. Returns entries dropped."""
        target = int(self.capacity * keep_fraction)
        if self.size <= target:
            return 0
        imp = self.importance(now_tick)
        keep = np.sort(np.argsort(-imp)[:target])
        self.latents[: len(keep)] = self.latents[keep]
        self.positions[: len(keep)] = self.positions[keep]
        self.ticks[: len(keep)] = self.ticks[keep]
        self.surprises[: len(keep)] = self.surprises[keep]
        self.summaries[: len(keep)] = [self.summaries[j] for j in keep]
        self.revisit_counts[: len(keep)] = self.revisit_counts[keep]
        dropped = self.size - len(keep)
        self.size = len(keep)
        return dropped

    def clear(self) -> None:
        """Episode boundary: this env's world was rebuilt; memories of the old
        world are no longer about anything."""
        self.size = 0

    def pressure(self) -> float:
        """Occupancy in [0, 1] for the intero memory_pressure slot."""
        return self.size / self.capacity

    # ----------------------------------------------------------------- state

    def state_dict(self) -> dict[str, Any]:
        return {
            "latents": self.latents.copy(), "positions": self.positions.copy(),
            "ticks": self.ticks.copy(), "surprises": self.surprises.copy(),
            "summaries": list(self.summaries),
            "revisit_counts": self.revisit_counts.copy(),
            "size": self.size, "gate": self.gate.state_dict(),
            "projection": self.projection.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.latents = state["latents"].copy()
        self.positions = state["positions"].copy()
        self.ticks = state["ticks"].copy()
        self.surprises = state["surprises"].copy()
        self.summaries = list(state["summaries"])
        self.revisit_counts = state["revisit_counts"].copy()
        self.size = state["size"]
        self.gate.load_state_dict(state["gate"])
        self.projection = state["projection"].copy()
