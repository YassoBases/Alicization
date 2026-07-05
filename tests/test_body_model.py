"""Unit tests for ledger/body_model.py: dpos classing, forward shapes, losses,
metrics, and the rolling-mean smoother."""

from __future__ import annotations

import pytest
import torch

from ledger.body_model import (
    DPOS_CLASSES,
    BodyModel,
    RollingMean,
    build_policy_features,
    compute_body_losses,
    dpos_to_class,
)
from world.engine import MOVE_E, MOVE_N, MOVE_S, MOVE_W, NOOP, World
from world.config import load_config

NUM_ACTIONS = 9
ROOT_CFG = "configs/base.yaml"


def test_dpos_to_class_matches_engine_move_deltas() -> None:
    """DPOS_CLASSES must line up with world.engine's actual realized deltas."""
    cfg = load_config(ROOT_CFG)
    w = World(cfg)
    w.set_agent_pos(0, 32, 32)
    expected_class = {NOOP: 0, MOVE_N: 1, MOVE_S: 2, MOVE_E: 3, MOVE_W: 4}
    for action, cls in expected_class.items():
        w.set_agent_pos(0, 32, 32)
        _, infos = w.step([action])
        dpos = torch.tensor([infos[0]["realized"]["dpos"]], dtype=torch.long)
        assert int(dpos_to_class(dpos)[0]) == cls, f"action {action} -> wrong class"


def test_dpos_to_class_batch() -> None:
    dpos = torch.tensor([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], dtype=torch.long)
    assert dpos_to_class(dpos).tolist() == [0, 1, 2, 3, 4]
    assert len(DPOS_CLASSES) == 5


def test_body_model_forward_all_actions_shapes_and_range() -> None:
    torch.manual_seed(0)
    core_dim = 12
    model = BodyModel({"body_hidden": [16, 16]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    h = torch.randn(4, core_dim)
    out = model(h)
    assert out["success_prob"].shape == (4, NUM_ACTIONS)
    assert out["denergy"].shape == (4, NUM_ACTIONS)
    assert out["dpos_class"].shape == (4, NUM_ACTIONS)
    assert out["dpos_class"].dtype == torch.long
    assert torch.all((out["success_prob"] >= 0) & (out["success_prob"] <= 1))
    assert torch.all((out["dpos_class"] >= 0) & (out["dpos_class"] < len(DPOS_CLASSES)))


def test_body_model_predict_action_shapes() -> None:
    torch.manual_seed(0)
    core_dim = 12
    model = BodyModel({"body_hidden": [16, 16]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    h = torch.randn(5, core_dim)
    onehot = torch.nn.functional.one_hot(torch.randint(0, NUM_ACTIONS, (5,)), NUM_ACTIONS).float()
    out = model.predict_action(h, onehot)
    assert out["dpos_logits"].shape == (5, 5)
    assert out["success_logit"].shape == (5,)
    assert out["denergy"].shape == (5,)


def test_forward_all_actions_consistent_with_predict_action() -> None:
    """forward() for action k must equal predict_action() given action k's one-hot."""
    torch.manual_seed(2)
    core_dim = 8
    model = BodyModel({"body_hidden": [8, 8]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    h = torch.randn(3, core_dim)
    all_out = model(h)
    for action in range(NUM_ACTIONS):
        onehot = torch.nn.functional.one_hot(torch.full((3,), action), NUM_ACTIONS).float()
        single = model.predict_action(h, onehot)
        expected_prob = torch.sigmoid(single["success_logit"])
        assert torch.allclose(all_out["success_prob"][:, action], expected_prob, atol=1e-6)
        assert torch.allclose(all_out["denergy"][:, action], single["denergy"], atol=1e-6)
        expected_class = single["dpos_logits"].argmax(dim=-1)
        assert torch.equal(all_out["dpos_class"][:, action], expected_class)


def test_compute_body_losses_values() -> None:
    """Hand-check the Brier score and MAE metrics against known predictions."""
    outputs = {
        "dpos_logits": torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0, 0.0]]),
        "success_logit": torch.tensor([0.0, 0.0]),  # sigmoid -> 0.5
        "denergy": torch.tensor([1.0, -1.0]),
    }
    real_dpos_class = torch.tensor([0, 1])  # matches the confident logits above
    real_success = torch.tensor([1.0, 0.0])
    real_denergy = torch.tensor([0.5, 0.0])

    losses = compute_body_losses(outputs, real_dpos_class, real_success, real_denergy)

    assert losses["body_nll"].item() < 0.01  # near-perfect confident CE
    expected_brier = ((0.5 - 1.0) ** 2 + (0.5 - 0.0) ** 2) / 2
    assert losses["success_brier"].item() == pytest.approx(expected_brier)
    assert losses["denergy_mae"].item() == pytest.approx((0.5 + 1.0) / 2)
    assert losses["total"].item() == pytest.approx(
        (losses["body_nll"] + losses["success_bce"] + losses["denergy_mse"]).item()
    )


def test_rolling_mean_ema() -> None:
    ema = RollingMean(decay=0.5)
    assert ema.update(10.0) == 10.0  # first value seeds it exactly
    assert ema.update(0.0) == 5.0  # 0.5*10 + 0.5*0
    assert ema.update(0.0) == 2.5


def test_build_policy_features_shape() -> None:
    """Shape/plumbing only. ``requires_grad`` on a tensor produced by a module
    with trainable parameters is True regardless of whether its INPUT was
    detached (e.g. ``linear(x.detach())`` still requires grad, from the
    weights) — so it can't be used to check isolation here. The actual
    no-gradient-reaches-body-model / no-gradient-reaches-core guarantees are
    verified properly (via backward() + param.grad) in
    tests/test_grad_isolation.py.
    """
    torch.manual_seed(3)
    core_dim = 10
    body = BodyModel({"body_hidden": [8, 8]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    core_out = torch.randn(4, core_dim, requires_grad=True)
    features, body_out = build_policy_features(core_out, body)
    assert features.shape == (4, core_dim + 2 * NUM_ACTIONS)
    assert body_out["success_prob"].shape == (4, NUM_ACTIONS)
    assert body_out["denergy"].shape == (4, NUM_ACTIONS)
    assert body_out["dpos_class"].shape == (4, NUM_ACTIONS)
