"""Adapters presenting SelfQ under the exact BodyModel and Forecaster
interfaces (stage-E2), so the policy features, arbiter scoring, mirror, and
sleep updates are unchanged when ``ledger.impl: selfq``.

Both adapters wrap the SAME SelfQ instance and delegate parameters/state to
it (they are plain wrappers, NOT nn.Modules, so SelfQ is not double-
registered and one optimizer trains it from both the wake body update and
the sleep forecaster update). The trainer constructs SelfQ once and shares
its optimizer across the two adapters.
"""

from __future__ import annotations

from typing import Any

import torch

from ledger.body_model import DPOS_CLASSES
from selfq.model import INTENT_ACTION, INTENT_PLAN, SelfQ, body_outputs


class _SelfQWrapper:
    """Shared delegation of the nn.Module surface the trainer touches."""

    selfq: SelfQ

    def parameters(self):  # noqa: ANN201
        return self.selfq.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.selfq.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> Any:  # noqa: ANN401
        return self.selfq.load_state_dict(sd)

    def to(self, device: Any) -> "_SelfQWrapper":  # noqa: ANN401
        self.selfq.to(device)
        return self

    def train(self, mode: bool = True) -> "_SelfQWrapper":
        self.selfq.train(mode)
        return self

    def eval(self) -> "_SelfQWrapper":
        self.selfq.eval()
        return self


class BodyModelAdapter(_SelfQWrapper):
    """SelfQ under ledger.body_model.BodyModel's interface (predict_action +
    per-action forward)."""

    def __init__(self, selfq: SelfQ) -> None:
        self.selfq = selfq
        self.num_actions = selfq.num_actions

    def predict_action(self, h_detached: torch.Tensor,
                       action_onehot: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.selfq.query(h_detached, action_onehot, INTENT_ACTION, 1)
        return body_outputs(pred)

    def __call__(self, h_detached: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-action k=1 predictions for the policy features + attribution,
        exactly matching BodyModel.forward's output shapes."""
        b, a = h_detached.shape[0], self.num_actions
        h_rep = h_detached.unsqueeze(1).expand(b, a, -1).reshape(b * a, -1)
        eye = torch.eye(a, device=h_detached.device).unsqueeze(0).expand(b, a, a)
        out = self.predict_action(h_rep, eye.reshape(b * a, a))
        return {
            "success_prob": torch.sigmoid(out["success_logit"]).reshape(b, a),
            "denergy": out["denergy"].reshape(b, a),
            "dpos_class": out["dpos_logits"].argmax(dim=-1).reshape(b, a),
        }

    # BodyModel is an nn.Module; a couple of call sites treat it as callable
    # via forward — expose both.
    forward = __call__


class ForecasterAdapter(_SelfQWrapper):
    """SelfQ under ledger.forecaster.Forecaster's interface (per-plan,
    per-horizon (mean, logvar))."""

    def __init__(self, selfq: SelfQ, horizons: tuple[int, ...]) -> None:
        self.selfq = selfq
        self.horizons = tuple(horizons)
        self.intero_dim = selfq.intero_dim
        self.num_plans = selfq.num_plans

    def __call__(self, h_detached: torch.Tensor, plan_onehot: torch.Tensor
                 ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for k in self.horizons:
            pred = self.selfq.query(h_detached, plan_onehot, INTENT_PLAN, k)
            out[k] = (pred.intero_mean, pred.intero_logvar)
        return out

    forward = __call__


assert len(DPOS_CLASSES) == 5  # SelfQ.N_DPOS_CLASSES must match the body model
