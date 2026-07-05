"""Attribution head: was an observed transition self-caused, world-caused, or
both?

Per tick: the body model's prediction for the action actually taken gives a
"predicted self-caused delta" (dpos class, denergy). The residual is the gap
between that prediction and the observed transition. A tiny multinomial
logistic classifier maps three scalar features —
``[|residual_pos|, |residual_energy|, action == noop]`` — to one of three
classes: SELF, WORLD, BOTH.

Trained SELF-SUPERVISED from residual-magnitude thresholds (``pseudo_label``)
— NEVER from the ground-truth cause labels in the world's event log. Those
ground-truth labels exist ONLY for evaluation; nothing in this module (or
anything it calls) ever reads them. See ``training/attribution_eval.py``,
which is deliberately a separate, non-ledger module for that reason, mirroring
world/levers.py's "evaluation only, never enters any observation or loss"
contract.

GRADIENT ISOLATION (CLAUDE.md Hard rules): this classifier's input is three
plain scalar features derived from (a) the body model's prediction — itself
already detached from the core — and (b) the observed transition (numbers
read out of vecenv info dicts, never part of any autograd graph). So its own
loss cannot reach the core or the body model. See tests/test_grad_isolation.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ledger.body_model import DPOS_CLASSES

SELF, WORLD, BOTH = 0, 1, 2
NUM_CLASSES = 3
CLASS_NAMES = ("self", "world", "both")

_DPOS_VECTORS = torch.tensor(DPOS_CLASSES, dtype=torch.float32)  # (5, 2)


class AttributionHead(nn.Module):
    """Multinomial logistic regression: 3 scalar features -> {self, world, both}.

    Deliberately no hidden layers — a genuine "logistic" classifier per spec,
    not an MLP like BodyModel.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """``cfg`` is the ``ledger.attribution`` config section (tau_pos,
        tau_energy, lr); unused at construction time, kept for symmetry with
        BodyModel/PPOTrainer's per-head config-section convention."""
        super().__init__()
        del cfg
        self.linear = nn.Linear(3, NUM_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, 3) -> (B, 3) logits over {self, world, both}."""
        return self.linear(features)


def residual_features(
    predicted_dpos_class: torch.Tensor,
    observed_dpos_class: torch.Tensor,
    predicted_denergy: torch.Tensor,
    observed_denergy: torch.Tensor,
    action: torch.Tensor,
    noop_action: int,
) -> torch.Tensor:
    """-> (B, 3): [|residual_pos| (Manhattan, in DPOS_CLASSES vector space),
    |residual_energy|, action == noop (0.0/1.0)].

    All four prediction/observation inputs are plain (already-detached, or
    never-differentiable) tensors; this function never introduces a gradient
    path anywhere.
    """
    vecs = _DPOS_VECTORS.to(predicted_dpos_class.device)
    r_pos = (vecs[predicted_dpos_class] - vecs[observed_dpos_class]).abs().sum(dim=-1)
    r_energy = (predicted_denergy - observed_denergy).abs()
    is_noop = (action == noop_action).float()
    return torch.stack([r_pos, r_energy, is_noop], dim=-1)


def pseudo_label(features: torch.Tensor, tau_pos: float, tau_energy: float) -> torch.Tensor:
    """Self-supervised targets from residual-magnitude thresholds — NEVER from
    ground truth.

    NOOP is a structural exception, not a residual-based one: the fixed
    action table guarantees NOOP can have no self-caused effect, so it is
    always labeled WORLD regardless of residual size. This is what makes
    "no-op ticks are never attributed to self" hold by construction, and it
    is public knowledge about the action table — not privileged ground truth.
    """
    r_pos, r_energy, is_noop = features[:, 0], features[:, 1], features[:, 2]
    pos_surprise = r_pos > tau_pos
    energy_surprise = r_energy > tau_energy
    label = torch.full_like(r_pos, SELF, dtype=torch.long)
    label[pos_surprise ^ energy_surprise] = WORLD
    label[pos_surprise & energy_surprise] = BOTH
    label[is_noop.bool()] = WORLD
    return label


def compute_attribution_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
