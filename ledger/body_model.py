"""Body model: per-action capability estimates (success prob, expected costs).

Trained online, one gradient step per rollout, on fresh (h, action, realized
outcome) transitions from vecenv info dicts (see training/ppo.py's
``update_body_model``). Its input is ``h.detach()`` — the core's GRU output,
never gradient-connected to the encoder/core — concatenated with a one-hot
action vector.

GRADIENT ISOLATION (CLAUDE.md Hard rules): the input is always detached before
this module sees it, and ``build_policy_features`` detaches this module's
OUTPUT before it is used as a policy input. So no gradient from either this
module's own loss or the policy loss ever reaches the encoder/GRU core, in
either direction. See tests/test_grad_isolation.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Realized dpos -> 5-way class index. Matches world.engine's fixed move deltas
# (MOVE_N/S/E/W); any non-move action, or a blocked/failed move, realizes
# dpos=(0, 0) ("stay").
DPOS_CLASSES: tuple[tuple[int, int], ...] = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))
_DPOS_VECTORS_INT = torch.tensor(DPOS_CLASSES, dtype=torch.long)  # (5, 2)


def dpos_to_class(dpos: torch.Tensor) -> torch.Tensor:
    """(B, 2) integer dpos -> (B,) long class index into DPOS_CLASSES.

    Realized dpos is normally exactly one of DPOS_CLASSES. It can rarely be a
    compound value outside that set — e.g. the agent's own successful move
    combines with an unrelated same-tick ghost push (world/levers.py), giving
    e.g. (1, -1). Mapping every non-exact-match to class 0 ("stay") would be a
    silent, systematic mislabel (the agent manifestly did move); nearest
    DPOS_CLASSES entry by Manhattan distance is a strictly better fallback,
    and it's a no-op for the overwhelming majority of exact-match ticks
    (distance 0 always wins the argmin).
    """
    vecs = _DPOS_VECTORS_INT.to(dpos.device)
    diffs = (dpos.unsqueeze(1) - vecs.unsqueeze(0)).abs().sum(dim=-1)  # (B, 5)
    return diffs.argmin(dim=-1)


class BodyModel(nn.Module):
    """MLP over [h.detach(), one_hot(action)] predicting the realized transition."""

    def __init__(self, cfg: dict[str, Any], core_dim: int, num_actions: int) -> None:
        """``cfg`` is the ``ledger`` config section (body_hidden: two hidden sizes)."""
        super().__init__()
        self.num_actions = num_actions
        layers: list[nn.Module] = []
        prev = core_dim + num_actions
        for size in cfg["body_hidden"]:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        self.trunk = nn.Sequential(*layers)
        self.dpos_head = nn.Linear(prev, len(DPOS_CLASSES))
        self.success_head = nn.Linear(prev, 1)
        self.denergy_head = nn.Linear(prev, 1)

    def predict_action(
        self, h_detached: torch.Tensor, action_onehot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """(B, core_dim), (B, num_actions) -> raw per-transition predictions:
        {'dpos_logits': (B, 5), 'success_logit': (B,), 'denergy': (B,)}."""
        feat = self.trunk(torch.cat([h_detached, action_onehot], dim=-1))
        return {
            "dpos_logits": self.dpos_head(feat),
            "success_logit": self.success_head(feat).squeeze(-1),
            "denergy": self.denergy_head(feat).squeeze(-1),
        }

    def forward(self, h_detached: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, core_dim) detached -> {'success_prob': (B, A), 'denergy': (B, A),
        'dpos_class': (B, A) long}, one prediction per possible action — used
        to feed the policy (success_prob/denergy) and, for the action actually
        taken, the attribution head (dpos_class; see ledger/attribution.py)."""
        b, a = h_detached.shape[0], self.num_actions
        h_rep = h_detached.unsqueeze(1).expand(b, a, -1).reshape(b * a, -1)
        eye = torch.eye(a, device=h_detached.device).unsqueeze(0).expand(b, a, a)
        out = self.predict_action(h_rep, eye.reshape(b * a, a))
        return {
            "success_prob": torch.sigmoid(out["success_logit"]).reshape(b, a),
            "denergy": out["denergy"].reshape(b, a),
            "dpos_class": out["dpos_logits"].argmax(dim=-1).reshape(b, a),
        }


def compute_body_losses(
    outputs: dict[str, torch.Tensor],
    real_dpos_class: torch.Tensor,
    real_success: torch.Tensor,
    real_denergy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """CE + BCE + MSE training losses, plus Brier/MAE diagnostic metrics.

    ``outputs`` is a ``BodyModel.predict_action`` result for the action
    actually taken in each transition.
    """
    ce = F.cross_entropy(outputs["dpos_logits"], real_dpos_class)
    bce = F.binary_cross_entropy_with_logits(outputs["success_logit"], real_success)
    mse = F.mse_loss(outputs["denergy"], real_denergy)
    with torch.no_grad():
        success_prob = torch.sigmoid(outputs["success_logit"])
        success_brier = ((success_prob - real_success) ** 2).mean()
        denergy_mae = (outputs["denergy"] - real_denergy).abs().mean()
    return {
        "total": ce + bce + mse,
        "body_nll": ce,
        "success_bce": bce,
        "denergy_mse": mse,
        "success_brier": success_brier,
        "denergy_mae": denergy_mae,
    }


def build_policy_features(
    core_out: torch.Tensor, body_model: BodyModel, use_ledger_features: bool = True
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """(B, core_dim), using ``body_model`` -> ((B, core_dim [+ 2*num_actions]), raw body_out).

    Concatenates the core output — left untouched, so the policy/value loss
    trains the encoder/core through it as usual — with the body model's
    per-action success/denergy predictions, explicitly detached: no policy
    gradient reaches the body model, and (belt-and-suspenders, since
    ``BodyModel.forward`` is also given a detached input) no gradient reaches
    the core through the body-model path either.

    The raw ``body_out`` dict (success_prob, denergy, dpos_class; all already
    detached) is also returned so callers that need the taken-action's
    prediction (e.g. the attribution head) don't need a second forward pass —
    the body model and attribution head always run and train regardless of
    ``use_ledger_features``.

    ``use_ledger_features=False`` (see ``agent.use_ledger_features`` config)
    is the capability-shift battery's architecture-B control: identical
    Ledger training, but the policy/value heads never see its output, only
    the raw core output. Used to isolate the causal effect of feeding Ledger
    estimates into the policy (experiments/batteries/capability_shift.py).
    """
    body_out = body_model(core_out.detach())
    if not use_ledger_features:
        return core_out, body_out
    features = torch.cat(
        [core_out, body_out["success_prob"].detach(), body_out["denergy"].detach()],
        dim=-1,
    )
    return features, body_out


class RollingMean:
    """Exponential moving average, for smoothing noisy per-rollout scalars in TB."""

    def __init__(self, decay: float = 0.98) -> None:
        self.decay = decay
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.decay * self.value + (1 - self.decay) * x
        return self.value
