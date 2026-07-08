"""Every Ledger head returns its documented output shapes for batch sizes
1 and 32 (the two sizes that catch squeeze()/broadcast bugs)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.drives import NUM_PLANS
from ledger.attribution import AttributionHead, residual_features
from ledger.body_model import DPOS_CLASSES, BodyModel
from ledger.forecaster import Forecaster
from ledger.reliability import FEATURES, ReliabilityModel

NUM_ACTIONS = 9
CORE_DIM = 24
INTERO_DIM = 6

BATCH_SIZES = (1, 32)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_body_model_shapes(batch: int) -> None:
    model = BodyModel({"body_hidden": [16, 16]}, core_dim=CORE_DIM, num_actions=NUM_ACTIONS)
    h = torch.randn(batch, CORE_DIM)
    onehot = torch.nn.functional.one_hot(
        torch.randint(0, NUM_ACTIONS, (batch,)), NUM_ACTIONS
    ).float()

    per_action = model.predict_action(h, onehot)
    assert per_action["dpos_logits"].shape == (batch, len(DPOS_CLASSES))
    assert per_action["success_logit"].shape == (batch,)
    assert per_action["denergy"].shape == (batch,)

    all_actions = model(h)
    assert all_actions["success_prob"].shape == (batch, NUM_ACTIONS)
    assert all_actions["denergy"].shape == (batch, NUM_ACTIONS)
    assert all_actions["dpos_class"].shape == (batch, NUM_ACTIONS)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_attribution_head_shapes(batch: int) -> None:
    head = AttributionHead({"tau_pos": 0.5, "tau_energy": 0.03, "lr": 1e-3})
    feats = residual_features(
        torch.randint(0, 5, (batch,)),
        torch.randint(0, 5, (batch,)),
        torch.randn(batch),
        torch.randn(batch),
        torch.randint(0, NUM_ACTIONS, (batch,)),
        noop_action=8,
    )
    assert feats.shape == (batch, 3)
    logits = head(feats)
    assert logits.shape == (batch, 3)  # {self, world, both}


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_forecaster_shapes(batch: int) -> None:
    horizons = [1, 10]
    model = Forecaster(
        {"forecaster_hidden": [16, 16], "horizons": horizons},
        core_dim=CORE_DIM, intero_dim=INTERO_DIM, num_plans=NUM_PLANS,
    )
    h = torch.randn(batch, CORE_DIM)
    plan = torch.nn.functional.one_hot(
        torch.randint(0, NUM_PLANS, (batch,)), NUM_PLANS
    ).float()
    out = model(h, plan)
    assert set(out) == set(horizons)
    for mean, logvar in out.values():
        assert mean.shape == (batch, INTERO_DIM)
        assert logvar.shape == (batch, INTERO_DIM)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_reliability_model_shapes(batch: int) -> None:
    model = ReliabilityModel({"lr": 1e-3}, world_size=64)
    feats = model.features(
        age=np.random.default_rng(0).uniform(0, 1000, batch),
        surprise=np.ones(batch),
        revisits=np.zeros(batch),
        positions=np.zeros((batch, 2), dtype=int),
    )
    assert feats.shape == (batch, len(FEATURES))
    preds = model.predict(feats)
    assert preds.shape == (batch,)
    assert np.all((preds >= 0) & (preds <= 1))
