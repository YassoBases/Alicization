"""Forecaster head: predicts future interoceptive state per macro-plan.

Input: ``h.detach()`` (core output — gradient isolation per CLAUDE.md Hard
rules, tested in tests/test_grad_isolation.py) concatenated with a one-hot
macro-plan id (the arbiter's options, agent/drives.py's PLANS). Output: mean
and logvar of the intero vector at each horizon in ``ledger.horizons``
([1, 10] now; 100 comes in Stage 6). Trained in sleep on stored
(h, plan, realized-future) tuples with Gaussian NLL, under its OWN optimizer.

Every metric that evaluates a forecast must report the identity-predictor
baseline (CLAUDE.md): ``identity_baseline`` repeats the current intero
vector, and ``nmse`` = MSE(forecast) / MSE(identity) — NMSE < 1.0 means the
forecaster beats "nothing will change". That baseline is mandatory on every
forecasting plot.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_LOGVAR_MIN, _LOGVAR_MAX = -8.0, 4.0  # NaN safety on the NLL


class Forecaster(nn.Module):
    """K-step-ahead intero forecasts from the detached core state + plan id."""

    def __init__(
        self, cfg: dict[str, Any], core_dim: int, intero_dim: int, num_plans: int
    ) -> None:
        """``cfg`` is the ``ledger`` config section (forecaster_hidden, horizons)."""
        super().__init__()
        self.horizons: tuple[int, ...] = tuple(cfg["horizons"])
        self.intero_dim = intero_dim
        self.num_plans = num_plans
        layers: list[nn.Module] = []
        prev = core_dim + num_plans
        for size in cfg["forecaster_hidden"]:
            layers += [nn.Linear(prev, size), nn.ELU()]
            prev = size
        self.trunk = nn.Sequential(*layers)
        # One (mean, logvar) head per horizon.
        self.heads = nn.ModuleDict({
            str(k): nn.Linear(prev, 2 * intero_dim) for k in self.horizons
        })

    def forward(
        self, h_detached: torch.Tensor, plan_onehot: torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """(B, core_dim) detached + (B, num_plans) ->
        {horizon: (mean (B, D), logvar (B, D))}."""
        feat = self.trunk(torch.cat([h_detached, plan_onehot], dim=-1))
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for k in self.horizons:
            mean, logvar = self.heads[str(k)](feat).chunk(2, dim=-1)
            out[k] = (mean, logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX))
        return out


def forecaster_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood (up to the constant term)."""
    return 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).sum(dim=-1).mean()


def identity_baseline(intero_now: torch.Tensor, horizon: int) -> torch.Tensor:
    """The mandatory baseline: predict intero at t+k as intero at t."""
    del horizon  # the identity predictor is horizon-independent by definition
    return intero_now


def nmse(
    forecast_mean: torch.Tensor, intero_now: torch.Tensor, target: torch.Tensor
) -> float:
    """MSE(forecast) / MSE(identity predictor). < 1.0 beats identity.

    Returns inf when the identity MSE is exactly zero (nothing changed —
    identity is unbeatable there and the ratio is degenerate).
    """
    mse_f = (forecast_mean - target).pow(2).mean().item()
    mse_i = (intero_now - target).pow(2).mean().item()
    return mse_f / mse_i if mse_i > 0 else float("inf")


class ForecastTupleStore:
    """FIFO of (h, plan, intero_now, {horizon: intero_future}) training tuples.

    Filled during wake by the trainer (which knows realized futures once k
    more ticks have elapsed); consumed during sleep by the forecaster's NLL
    updates and by NMSE evaluation. Plain tensors, no autograd anywhere.
    """

    def __init__(self, capacity: int, horizons: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.horizons = horizons
        self.h: list[torch.Tensor] = []
        self.plan: list[int] = []
        self.intero_now: list[torch.Tensor] = []
        self.intero_future: list[dict[int, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.h)

    def add(
        self,
        h: torch.Tensor,
        plan: int,
        intero_now: torch.Tensor,
        intero_future: dict[int, torch.Tensor],
    ) -> None:
        self.h.append(h.detach().cpu())
        self.plan.append(plan)
        self.intero_now.append(intero_now.detach().cpu())
        self.intero_future.append({k: v.detach().cpu() for k, v in intero_future.items()})
        if len(self.h) > self.capacity:
            self.h.pop(0); self.plan.pop(0)
            self.intero_now.pop(0); self.intero_future.pop(0)

    def batch(
        self, num_plans: int, device: torch.device
    ) -> dict[str, Any] | None:
        if not self.h:
            return None
        h = torch.stack(self.h).to(device)
        plan = torch.nn.functional.one_hot(
            torch.tensor(self.plan, dtype=torch.long), num_plans
        ).float().to(device)
        now = torch.stack(self.intero_now).to(device)
        future = {
            k: torch.stack([f[k] for f in self.intero_future]).to(device)
            for k in self.horizons
        }
        return {"h": h, "plan": plan, "intero_now": now, "future": future}
