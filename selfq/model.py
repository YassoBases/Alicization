"""The SelfQ conditional self-model (stage-E1).

One MLP over [h.detach(), intent embedding, learned horizon embedding] with
four output heads spanning the whole self-observable vector:
  - dpos_logits  (5-way class, the body model's move outcome)
  - success_logit
  - denergy      (Delta-energy)
  - intero       (mean, logvar) at horizon k  (the forecaster's target)

Intent is EITHER an action (num_actions one-hot, for k=1 body queries) or a
macro-plan (num_plans one-hot, for k>=1 forecaster queries), embedded through
separate linear maps into a shared space; the horizon is a learned embedding
over the supported set (always including 1). At k=1 an action query exercises
the body heads; at each configured horizon a plan query exercises the intero
head — the same trunk, unified.

Gradient isolation: the input h is detached by the caller (as for the heads
SelfQ replaces); SelfQ never sees a grad-connected core state, and its output
is detached before entering the policy (selfq/adapters.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Intent kinds.
INTENT_ACTION = "action"
INTENT_PLAN = "plan"

_LOGVAR_MIN, _LOGVAR_MAX = -8.0, 4.0  # NaN safety on the intero NLL (matches forecaster)
N_DPOS_CLASSES = 5                     # matches ledger.body_model.DPOS_CLASSES


@dataclass
class SelfPrediction:
    """One SelfQ query result. dpos/success/denergy are the k=1 body outcome;
    intero_(mean,logvar) is the horizon-k interoceptive forecast."""

    dpos_logits: torch.Tensor      # (B, 5)
    success_logit: torch.Tensor    # (B,)
    denergy: torch.Tensor          # (B,)
    intero_mean: torch.Tensor      # (B, D)
    intero_logvar: torch.Tensor    # (B, D)
    horizon: int = 1


class SelfQ(nn.Module):
    def __init__(self, cfg: dict[str, Any], core_dim: int, num_actions: int,
                 num_plans: int, intero_dim: int) -> None:
        """``cfg`` is the ``ledger`` config section (horizons, selfq_embed,
        selfq_hidden). The horizon set always includes 1 (the body query)."""
        super().__init__()
        self.num_actions = num_actions
        self.num_plans = num_plans
        self.intero_dim = intero_dim
        self.horizons: tuple[int, ...] = tuple(
            sorted(set(cfg["horizons"]) | {1}))
        self._hz_index = {k: i for i, k in enumerate(self.horizons)}

        embed = int(cfg.get("selfq_embed", 64))
        self.action_embed = nn.Linear(num_actions, embed)
        self.plan_embed = nn.Linear(num_plans, embed)
        self.horizon_embed = nn.Embedding(len(self.horizons), embed)

        layers: list[nn.Module] = []
        prev = core_dim + 2 * embed        # h + intent embed + horizon embed
        for size in cfg.get("selfq_hidden", [128, 128]):
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        self.trunk = nn.Sequential(*layers)
        self.dpos_head = nn.Linear(prev, N_DPOS_CLASSES)
        self.success_head = nn.Linear(prev, 1)
        self.denergy_head = nn.Linear(prev, 1)
        self.intero_head = nn.Linear(prev, 2 * intero_dim)

    def _intent_vec(self, intent_onehot: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == INTENT_ACTION:
            return self.action_embed(intent_onehot)
        if kind == INTENT_PLAN:
            return self.plan_embed(intent_onehot)
        raise ValueError(f"unknown intent kind {kind!r}")

    def query(self, h_detached: torch.Tensor, intent_onehot: torch.Tensor,
              kind: str, horizon: int) -> SelfPrediction:
        """(B, core_dim) detached + (B, intent_dim) one-hot + kind + horizon
        -> SelfPrediction. ``horizon`` must be in ``self.horizons``."""
        if horizon not in self._hz_index:
            raise ValueError(f"horizon {horizon} not in {self.horizons}")
        b = h_detached.shape[0]
        intent = self._intent_vec(intent_onehot, kind)
        hz_idx = torch.full((b,), self._hz_index[horizon], dtype=torch.long,
                            device=h_detached.device)
        hz = self.horizon_embed(hz_idx)
        feat = self.trunk(torch.cat([h_detached, intent, hz], dim=-1))
        mean, logvar = self.intero_head(feat).chunk(2, dim=-1)
        return SelfPrediction(
            dpos_logits=self.dpos_head(feat),
            success_logit=self.success_head(feat).squeeze(-1),
            denergy=self.denergy_head(feat).squeeze(-1),
            intero_mean=mean,
            intero_logvar=logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX),
            horizon=horizon)


def body_outputs(pred: SelfPrediction) -> dict[str, torch.Tensor]:
    """The subset ledger.body_model.compute_body_losses consumes (a k=1
    action query), in the BodyModel.predict_action output shape."""
    return {"dpos_logits": pred.dpos_logits,
            "success_logit": pred.success_logit,
            "denergy": pred.denergy}


def selfq_body_losses(pred: SelfPrediction, real_dpos_class: torch.Tensor,
                      real_success: torch.Tensor,
                      real_denergy: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same CE+BCE+MSE (+ Brier/MAE diagnostics) as compute_body_losses, on a
    SelfQ k=1 prediction — so parity is measured on identical metrics."""
    ce = F.cross_entropy(pred.dpos_logits, real_dpos_class)
    bce = F.binary_cross_entropy_with_logits(pred.success_logit, real_success)
    mse = F.mse_loss(pred.denergy, real_denergy)
    with torch.no_grad():
        success_prob = torch.sigmoid(pred.success_logit)
        success_brier = ((success_prob - real_success) ** 2).mean()
        denergy_mae = (pred.denergy - real_denergy).abs().mean()
    return {"total": ce + bce + mse, "body_nll": ce, "success_bce": bce,
            "denergy_mse": mse, "success_brier": success_brier,
            "denergy_mae": denergy_mae}


def selfq_forecaster_nll(pred: SelfPrediction, target: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL on the intero forecast (identical to forecaster_nll)."""
    logvar, mean = pred.intero_logvar, pred.intero_mean
    return 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).sum(dim=-1).mean()
