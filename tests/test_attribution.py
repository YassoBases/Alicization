"""Unit tests for ledger/attribution.py: residual features, self-supervised
pseudo-labeling (including the NOOP structural exception), and the classifier."""

from __future__ import annotations

import torch

from ledger.attribution import (
    BOTH,
    NUM_CLASSES,
    SELF,
    WORLD,
    AttributionHead,
    pseudo_label,
    residual_features,
)
from world.engine import EAT, MOVE_E, NOOP

NOOP_ACTION = NOOP


def test_residual_features_zero_when_prediction_matches() -> None:
    pred_cls = torch.tensor([0, 3])
    obs_cls = torch.tensor([0, 3])
    pred_e = torch.tensor([0.1, -0.2])
    obs_e = torch.tensor([0.1, -0.2])
    action = torch.tensor([MOVE_E, EAT])
    feats = residual_features(pred_cls, obs_cls, pred_e, obs_e, action, NOOP_ACTION)
    assert torch.allclose(feats[:, 0], torch.zeros(2))
    assert torch.allclose(feats[:, 1], torch.zeros(2))
    assert torch.equal(feats[:, 2], torch.zeros(2))


def test_residual_features_manhattan_distance_and_noop_flag() -> None:
    # class 1 = (0, -1), class 3 = (1, 0) -> Manhattan distance |1-0|+|0-(-1)| = 2
    pred_cls = torch.tensor([1])
    obs_cls = torch.tensor([3])
    pred_e = torch.tensor([0.0])
    obs_e = torch.tensor([0.05])
    action = torch.tensor([NOOP_ACTION])
    feats = residual_features(pred_cls, obs_cls, pred_e, obs_e, action, NOOP_ACTION)
    assert feats[0, 0].item() == 2.0
    assert feats[0, 1].item() == pytest_approx(0.05)
    assert feats[0, 2].item() == 1.0


def pytest_approx(x: float, tol: float = 1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_pseudo_label_self_when_no_surprise() -> None:
    feats = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.001, 0.0]])
    labels = pseudo_label(feats, tau_pos=0.5, tau_energy=0.01)
    assert labels.tolist() == [SELF, SELF]


def test_pseudo_label_world_when_one_channel_surprises() -> None:
    feats = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    labels = pseudo_label(feats, tau_pos=0.5, tau_energy=0.01)
    assert labels.tolist() == [WORLD, WORLD]


def test_pseudo_label_both_when_both_channels_surprise() -> None:
    feats = torch.tensor([[2.0, 0.5, 0.0]])
    labels = pseudo_label(feats, tau_pos=0.5, tau_energy=0.01)
    assert labels.tolist() == [BOTH]


def test_pseudo_label_noop_always_world_regardless_of_residual() -> None:
    """The acceptance property: no-op ticks are never labeled self, even with
    a huge residual (structural override, not threshold-based)."""
    feats = torch.tensor(
        [
            [0.0, 0.0, 1.0],   # noop, zero residual -> still WORLD
            [5.0, 5.0, 1.0],   # noop, huge residual -> still WORLD (not BOTH)
        ]
    )
    labels = pseudo_label(feats, tau_pos=0.5, tau_energy=0.01)
    assert labels.tolist() == [WORLD, WORLD]


def test_attribution_head_shape() -> None:
    torch.manual_seed(0)
    head = AttributionHead({"tau_pos": 0.5, "tau_energy": 0.01, "lr": 1e-3})
    feats = torch.randn(7, 3)
    logits = head(feats)
    assert logits.shape == (7, NUM_CLASSES)
