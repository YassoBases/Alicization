"""Stage-E1: the SelfQ conditional model — shapes, horizon handling, and that
its body/forecaster losses match the heads' losses on identical inputs (so a
parity gate measures the same metric)."""

from __future__ import annotations

import pytest
import torch

from ledger.body_model import compute_body_losses
from ledger.forecaster import forecaster_nll
from selfq import (
    INTENT_ACTION,
    INTENT_PLAN,
    SelfPrediction,
    SelfQ,
    body_outputs,
    selfq_body_losses,
    selfq_forecaster_nll,
)

CFG = {"horizons": [1, 10], "selfq_embed": 16, "selfq_hidden": [32, 32]}
NUM_ACTIONS, NUM_PLANS, INTERO = 9, 5, 6


def _selfq() -> SelfQ:
    torch.manual_seed(0)
    return SelfQ(CFG, core_dim=16, num_actions=NUM_ACTIONS, num_plans=NUM_PLANS,
                 intero_dim=INTERO)


def test_horizon_set_always_includes_one() -> None:
    sq = SelfQ({**CFG, "horizons": [10, 100]}, 16, NUM_ACTIONS, NUM_PLANS, INTERO)
    assert sq.horizons == (1, 10, 100)


def test_query_shapes_action_and_plan() -> None:
    sq = _selfq()
    h = torch.randn(4, 16)
    a = torch.nn.functional.one_hot(torch.randint(0, NUM_ACTIONS, (4,)), NUM_ACTIONS).float()
    pred = sq.query(h, a, INTENT_ACTION, horizon=1)
    assert isinstance(pred, SelfPrediction)
    assert pred.dpos_logits.shape == (4, 5)
    assert pred.success_logit.shape == (4,) and pred.denergy.shape == (4,)
    assert pred.intero_mean.shape == (4, INTERO)
    p = torch.nn.functional.one_hot(torch.randint(0, NUM_PLANS, (4,)), NUM_PLANS).float()
    fc = sq.query(h, p, INTENT_PLAN, horizon=10)
    assert fc.intero_mean.shape == (4, INTERO) and fc.horizon == 10


def test_unknown_horizon_and_kind_raise() -> None:
    sq = _selfq()
    h = torch.randn(2, 16)
    a = torch.zeros(2, NUM_ACTIONS)
    with pytest.raises(ValueError):
        sq.query(h, a, INTENT_ACTION, horizon=7)
    with pytest.raises(ValueError):
        sq.query(h, a, "nonsense", horizon=1)


def test_body_losses_match_head_losses_on_same_prediction() -> None:
    """selfq_body_losses must be numerically identical to compute_body_losses
    given the same logits — parity is measured on one metric, not two."""
    sq = _selfq()
    h = torch.randn(8, 16)
    a = torch.nn.functional.one_hot(torch.randint(0, NUM_ACTIONS, (8,)), NUM_ACTIONS).float()
    pred = sq.query(h, a, INTENT_ACTION, horizon=1)
    dpos = torch.randint(0, 5, (8,))
    succ = torch.randint(0, 2, (8,)).float()
    den = torch.randn(8)
    a_losses = selfq_body_losses(pred, dpos, succ, den)
    b_losses = compute_body_losses(body_outputs(pred), dpos, succ, den)
    for key in ("total", "body_nll", "success_bce", "denergy_mse", "success_brier"):
        assert torch.allclose(a_losses[key], b_losses[key])


def test_forecaster_nll_matches_head_nll() -> None:
    sq = _selfq()
    h = torch.randn(8, 16)
    p = torch.nn.functional.one_hot(torch.randint(0, NUM_PLANS, (8,)), NUM_PLANS).float()
    pred = sq.query(h, p, INTENT_PLAN, horizon=10)
    target = torch.rand(8, INTERO)
    assert torch.allclose(selfq_forecaster_nll(pred, target),
                          forecaster_nll(pred.intero_mean, pred.intero_logvar, target))
