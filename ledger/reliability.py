"""Memory-reliability model: learned trustworthiness of episodic recalls.

Verification pipeline: when the agent revisits (within ``radius`` of) a
stored entry's position, the entry's stored local food/water summary is
compared against what is actually observed -> a match label in [0, 1]. Each
verification queues (features, label) and updates a per-8x8-region running
mismatch rate — the agent's OWN volatility estimate, learned exclusively
from its verification history, never read from lever config (nothing in
ledger/ may import world.levers; enforced by test).

Model: a deliberately simple logistic regression over
``[age_norm, surprise_at_write, revisit_count_norm, local_volatility]``.
Trained online (ledger.online_updates covers body + reliability heads) with
its own optimizer on the queued pairs; gradient isolation holds trivially
(every input is a plain observed scalar) and is still tested.

Uses: retrieval scores are multiplied by predicted reliability
(memory/episodic.py's ``reliability_fn`` hook), and the arbiter gains an
"inspect" drive — the entry with the highest ``importance * (1 -
predicted_reliability)`` becomes a revisit target. ``reliability.enabled:
false`` is the reliability-blind ablation: verification still runs (so the
comparison is fair) but predictions influence nothing.

Report: 10-bin expected calibration error (ECE) + per-region fitted
reliability-vs-age curves (scripts/verify_reliability.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

FEATURES = ("age_norm", "surprise_at_write", "revisit_count_norm", "local_volatility")


def compare_summaries(
    stored: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    offset: tuple[int, int],
) -> float | None:
    """Match label in [0, 1]: FOOD RECALL — the fraction of remembered food
    cells still holding food in the live observation, over the overlapping
    window region.

    ``offset`` = (revisit_pos - write_pos): the observed window is shifted
    relative to the stored one, so only the overlap is compared. Returns None
    when the overlap is empty or holds no remembered food (nothing to
    verify). Whole-window cell agreement was tried first and is useless as a
    label: one moved patch shifts it by ~1/(2*W*W), so every label saturates
    near 1.0 and the model can only learn a bias.
    """
    dx, dy = offset
    w = stored["food"].shape[0]
    if abs(dx) >= w or abs(dy) >= w:
        return None
    # Overlap slices: stored cell (x) aligns with observed cell (x - dx).
    x0s, x1s = max(0, dx), w + min(0, dx)
    y0s, y1s = max(0, dy), w + min(0, dy)
    x0o, x1o = max(0, -dx), w + min(0, -dx)
    y0o, y1o = max(0, -dy), w + min(0, -dy)
    s_food = stored["food"][y0s:y1s, x0s:x1s]
    o_food = observed["food"][y0o:y1o, x0o:x1o]
    n_stored = int(s_food.sum())
    if n_stored == 0:
        return None
    return float((s_food & o_food).sum()) / n_stored


class RegionVolatility:
    """Running mismatch rate per ``region_size``-square region — the agent's
    own volatility estimate from verification outcomes only."""

    def __init__(self, world_size: int, region_size: int = 8, ema: float = 0.05) -> None:
        self.region_size = region_size
        n = (world_size + region_size - 1) // region_size
        self.grid = np.zeros((n, n), dtype=np.float64)
        self.counts = np.zeros((n, n), dtype=np.int64)
        self.ema = ema

    def _idx(self, pos: tuple[int, int]) -> tuple[int, int]:
        return pos[1] // self.region_size, pos[0] // self.region_size

    def update(self, pos: tuple[int, int], mismatch: float) -> None:
        r, c = self._idx(pos)
        self.counts[r, c] += 1
        self.grid[r, c] += self.ema * (mismatch - self.grid[r, c])

    def get(self, pos: tuple[int, int]) -> float:
        r, c = self._idx(pos)
        return float(self.grid[r, c])

    def state_dict(self) -> dict[str, Any]:
        return {"grid": self.grid.copy(), "counts": self.counts.copy()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.grid = state["grid"].copy()
        self.counts = state["counts"].copy()


class ReliabilityModel(nn.Module):
    """Logistic regression over FEATURES -> P(memory still matches world)."""

    def __init__(self, cfg: dict[str, Any], world_size: int) -> None:
        """``cfg`` is the ``ledger.reliability`` config section."""
        super().__init__()
        self.enabled: bool = cfg.get("enabled", True)
        self.age_tau: float = cfg.get("age_tau", 5000.0)
        self.radius: int = cfg.get("radius", 2)
        self.min_age: int = cfg.get("min_age", 50)
        self.verify_cooldown: int = cfg.get("verify_cooldown", 25)
        self.queue_capacity: int = cfg.get("queue_capacity", 5000)
        self.linear = nn.Linear(len(FEATURES), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # start at 0.5 everywhere: no prior
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.get("lr", 1e-3))
        self.volatility = RegionVolatility(
            world_size, cfg.get("region_size", 8), cfg.get("volatility_ema", 0.05)
        )
        self.queue_x: list[np.ndarray] = []
        self.queue_y: list[float] = []
        self.n_verifications = 0

    # -------------------------------------------------------------- features

    def features(
        self,
        age: np.ndarray,
        surprise: np.ndarray,
        revisits: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """(size,) arrays -> (size, 4) feature matrix (vectorized)."""
        vol = np.array([
            self.volatility.get((int(x), int(y))) for x, y in positions
        ])
        return np.stack([
            age / self.age_tau,
            surprise,
            np.log1p(revisits) / 3.0,
            vol,
        ], axis=1).astype(np.float32)

    # ---------------------------------------------------------- verification

    def record(self, feats: np.ndarray, label: float, pos: tuple[int, int]) -> None:
        """Queue one verification outcome and update the region volatility."""
        self.queue_x.append(feats.astype(np.float32))
        self.queue_y.append(float(label))
        if len(self.queue_x) > self.queue_capacity:
            self.queue_x.pop(0)
            self.queue_y.pop(0)
        self.volatility.update(pos, 1.0 - label)
        self.n_verifications += 1

    # ------------------------------------------------------------- train/use

    @torch.no_grad()
    def predict(self, feats: np.ndarray) -> np.ndarray:
        """(B, 4) -> (B,) predicted reliability in [0, 1]."""
        x = torch.from_numpy(feats.astype(np.float32))
        return torch.sigmoid(self.linear(x)).squeeze(-1).numpy()

    def train_step(self, batch_size: int = 256) -> float | None:
        """One BCE gradient step on a random queue minibatch. None if empty."""
        if len(self.queue_x) < 8:
            return None
        idx = np.random.default_rng(self.n_verifications).choice(
            len(self.queue_x), size=min(batch_size, len(self.queue_x)), replace=False
        )
        x = torch.from_numpy(np.stack([self.queue_x[i] for i in idx]))
        y = torch.tensor([self.queue_y[i] for i in idx], dtype=torch.float32)
        logits = self.linear(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    # -------------------------------------------------------------- reports

    def calibration_ece(self, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
        """10-bin expected calibration error over the verification queue."""
        if not self.queue_x:
            return float("nan"), []
        preds = self.predict(np.stack(self.queue_x))
        labels = np.asarray(self.queue_y)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        rows = []
        for b in range(bins):
            hi_ok = preds < edges[b + 1] if b < bins - 1 else preds <= 1.0
            mask = (preds >= edges[b]) & hi_ok
            if not mask.any():
                continue
            conf, acc, frac = preds[mask].mean(), labels[mask].mean(), mask.mean()
            ece += frac * abs(conf - acc)
            rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                         "confidence": float(conf), "accuracy": float(acc),
                         "count": int(mask.sum())})
        return float(ece), rows

    def decay_curve(
        self, volatility: float, max_age: float = 10000.0, n: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicted reliability vs age at fixed volatility (fitted curve for
        the report; surprise/revisits held at typical values)."""
        ages = np.linspace(0.0, max_age, n)
        feats = np.stack([
            ages / self.age_tau,
            np.full(n, 1.0),          # typical surprise
            np.full(n, np.log1p(1) / 3.0),
            np.full(n, volatility),
        ], axis=1).astype(np.float32)
        return ages, self.predict(feats)

    # ----------------------------------------------------------------- state

    def reliability_state_dict(self) -> dict[str, Any]:
        return {
            "model": self.state_dict(), "opt": self.opt.state_dict(),
            "volatility": self.volatility.state_dict(),
            "queue_x": list(self.queue_x), "queue_y": list(self.queue_y),
            "n_verifications": self.n_verifications,
        }

    def load_reliability_state_dict(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.volatility.load_state_dict(state["volatility"])
        self.queue_x = list(state["queue_x"])
        self.queue_y = list(state["queue_y"])
        self.n_verifications = state["n_verifications"]


class ReliabilityHead(ReliabilityModel):
    """Back-compat alias for the original stub's class name."""
