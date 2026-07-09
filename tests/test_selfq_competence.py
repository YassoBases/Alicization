"""Stage-E3: competence-as-aggregation. In selfq mode the per-region
forecaster NMSE is derived from SelfQ's own query errors and fed into the
competence tracker (the schema is unchanged; the field was simply never fed
per-region before)."""

from __future__ import annotations

import numpy as np
import torch

from ledger.competence import CompetenceTracker
from ledger.forecaster import ForecastTupleStore
from selfq import SelfQ
from selfq.competence import feed_forecast_competence

CFG = {"horizons": [1, 10], "selfq_embed": 16, "selfq_hidden": [32, 32]}


def _selfq() -> SelfQ:
    torch.manual_seed(0)
    return SelfQ(CFG, core_dim=16, num_actions=9, num_plans=5, intero_dim=6)


def _batch(n: int = 60) -> dict:
    torch.manual_seed(1)
    return {
        "h": torch.randn(n, 16),
        "plan": torch.nn.functional.one_hot(torch.randint(0, 5, (n,)), 5).float(),
        "intero_now": torch.randn(n, 6),
        "future": {1: torch.randn(n, 6), 10: torch.randn(n, 6)},
        # spread positions across a few regions of a 32-world.
        "pos": [(i % 32, (i * 5) % 32) for i in range(n)],
    }


def test_feed_forecast_populates_per_region_nmse() -> None:
    tracker = CompetenceTracker(world_size=32)
    fed = feed_forecast_competence(tracker, _batch(), _selfq())
    assert fed > 0
    # The fed cells carry a finite NMSE, derived from SelfQ query errors.
    nmses = [cell.nmse for cell in tracker._cells.values()]
    assert nmses and any(np.isfinite(v) for v in nmses)


def test_no_positions_is_a_noop() -> None:
    tracker = CompetenceTracker(world_size=32)
    b = _batch()
    del b["pos"]
    assert feed_forecast_competence(tracker, b, _selfq()) == 0
    assert not tracker._cells   # nothing fed


def test_tuple_store_carries_pos_additively() -> None:
    store = ForecastTupleStore(capacity=100, horizons=(1, 10))
    # heads-style add (no pos) -> batch omits pos, behavior unchanged.
    for _ in range(4):
        store.add(torch.randn(16), 0, torch.randn(6), {1: torch.randn(6), 10: torch.randn(6)})
    b = store.batch(5, torch.device("cpu"))
    assert b is not None and "pos" not in b
    # selfq-style add (with pos) -> batch surfaces it.
    store.add(torch.randn(16), 1, torch.randn(6), {1: torch.randn(6), 10: torch.randn(6)},
              pos=(3, 4))
    b = store.batch(5, torch.device("cpu"))
    assert "pos" in b and b["pos"][-1] == (3, 4)


def test_region_nmse_is_pooled_ratio_not_per_sample() -> None:
    """A region's NMSE is sum(MSE_forecast)/sum(MSE_identity) over its tuples
    — pooled, so it is a stable ratio rather than a mean of noisy per-sample
    ratios. Sanity: a region where SelfQ equals the target has NMSE ~ 0."""
    tracker = CompetenceTracker(world_size=32)
    selfq = _selfq()
    n = 20
    h = torch.randn(n, 16)
    plan = torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long), 5).float()
    with torch.no_grad():
        pred = selfq.query(h, plan, "plan", 10)
    batch = {"h": h, "plan": plan, "intero_now": torch.randn(n, 6),
             "future": {10: pred.intero_mean.clone()},  # target == forecast
             "pos": [(0, 0)] * n}
    feed_forecast_competence(tracker, batch, selfq, horizon=10)
    cell = tracker._cells[(0, 0, "all")]
    assert cell.nmse < 1e-6   # forecast is exactly the target
