"""Forecaster + arbiter tests: shapes, NLL learning, the mandatory identity
baseline (NMSE), tuple-store bookkeeping, plan executors, and epsilon-greedy
plan selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.drives import (
    NUM_PLANS,
    PLANS,
    Arbiter,
    drive_error,
    homeostatic_reward,
    plan_action,
)
from ledger.forecaster import (
    Forecaster,
    ForecastTupleStore,
    forecaster_nll,
    identity_baseline,
    nmse,
)

CORE_DIM, INTERO_DIM = 16, 6
LCFG = {"forecaster_hidden": [32, 32], "horizons": [1, 10]}


def make_forecaster() -> Forecaster:
    torch.manual_seed(0)
    return Forecaster(LCFG, core_dim=CORE_DIM, intero_dim=INTERO_DIM, num_plans=NUM_PLANS)


# ------------------------------------------------------------------ shapes


def test_forecaster_output_shapes() -> None:
    f = make_forecaster()
    h = torch.randn(5, CORE_DIM)
    plan = torch.nn.functional.one_hot(torch.randint(0, NUM_PLANS, (5,)), NUM_PLANS).float()
    out = f(h, plan)
    assert set(out.keys()) == {1, 10}
    for mean, logvar in out.values():
        assert mean.shape == (5, INTERO_DIM) and logvar.shape == (5, INTERO_DIM)


def test_forecaster_learns_plan_dependent_drift() -> None:
    """Synthetic task: future intero = now + plan-specific constant drift,
    where h ENCODES the current intero (as the trained RSSM state does — the
    encoder consumes intero and reconstruction trains it into h; a forecaster
    can only beat identity if its input carries the current state). After
    training, forecast NMSE must beat the identity predictor (< 1)."""
    torch.manual_seed(1)
    f = make_forecaster()
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    n = 512
    now = torch.rand(n, INTERO_DIM)
    h = torch.randn(n, CORE_DIM)
    h[:, :INTERO_DIM] = now  # h carries the current intero
    plans = torch.randint(0, NUM_PLANS, (n,))
    plan_oh = torch.nn.functional.one_hot(plans, NUM_PLANS).float()
    drift = torch.linspace(-0.3, 0.3, NUM_PLANS)[plans].unsqueeze(-1)
    future = {1: now + 0.1 * drift, 10: now + drift}

    for _ in range(300):
        out = f(h, plan_oh)
        loss = torch.stack([
            forecaster_nll(out[k][0], out[k][1], future[k]) for k in (1, 10)
        ]).sum()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        out = f(h, plan_oh)
    score = nmse(out[10][0], now, future[10])
    assert score < 1.0, f"forecaster did not beat identity: NMSE={score:.3f}"


# --------------------------------------------------------- identity baseline


def test_identity_baseline_and_nmse() -> None:
    now = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[2.0, 3.0]])
    assert torch.equal(identity_baseline(now, 10), now)
    perfect = nmse(target, now, target)
    assert perfect == 0.0
    same_as_identity = nmse(now, now, target)
    assert same_as_identity == pytest.approx(1.0)
    # Degenerate case: identity is exact -> ratio undefined -> inf.
    assert nmse(target, now, now) == float("inf")


# --------------------------------------------------------------- tuple store


def test_tuple_store_fifo_and_batch() -> None:
    store = ForecastTupleStore(capacity=3, horizons=(1, 10))
    for i in range(5):
        store.add(
            torch.full((CORE_DIM,), float(i)), i % NUM_PLANS,
            torch.full((INTERO_DIM,), float(i)),
            {1: torch.zeros(INTERO_DIM), 10: torch.ones(INTERO_DIM)},
        )
    assert len(store) == 3  # FIFO evicted the first two
    batch = store.batch(NUM_PLANS, torch.device("cpu"))
    assert batch is not None
    assert batch["h"].shape == (3, CORE_DIM)
    assert batch["h"][0, 0].item() == 2.0  # oldest kept is i=2
    assert batch["plan"].shape == (3, NUM_PLANS)
    assert batch["future"][10].shape == (3, INTERO_DIM)


# ------------------------------------------------------------ plan executors


def _empty_grid(channels: int = 8, window: int = 11) -> np.ndarray:
    return np.zeros((channels, window, window), dtype=np.float32)


def test_forage_nearest_eats_on_food_and_moves_toward_it() -> None:
    rng = np.random.default_rng(0)
    ch_food, ch_shelter = 3, 5
    plan = PLANS.index("forage_nearest")

    grid = _empty_grid()
    grid[ch_food, 5, 5] = 1.0  # center: standing on food
    assert plan_action(plan, grid, ch_food, ch_shelter, rng) == 4  # EAT

    grid = _empty_grid()
    grid[ch_food, 5, 8] = 1.0  # east of center
    assert plan_action(plan, grid, ch_food, ch_shelter, rng) == 2  # MOVE_E

    grid = _empty_grid()
    grid[ch_food, 2, 5] = 1.0  # north of center
    assert plan_action(plan, grid, ch_food, ch_shelter, rng) == 0  # MOVE_N


def test_rest_and_goto_shelter() -> None:
    rng = np.random.default_rng(0)
    ch_food, ch_shelter = 3, 5
    assert plan_action(PLANS.index("rest"), _empty_grid(), ch_food, ch_shelter, rng) == 5

    grid = _empty_grid()
    grid[ch_shelter, 5, 5] = 1.0  # on shelter -> REST
    assert plan_action(PLANS.index("goto_shelter"), grid, ch_food, ch_shelter, rng) == 5
    grid = _empty_grid()
    grid[ch_shelter, 5, 2] = 1.0  # west
    assert plan_action(PLANS.index("goto_shelter"), grid, ch_food, ch_shelter, rng) == 3


def test_explore_moves_toward_higher_epistemic() -> None:
    rng = np.random.default_rng(0)
    plan = PLANS.index("explore_high_epistemic")
    emap = np.zeros((16, 16))
    emap[7, 9] = 5.0  # east of pos (8, 7)
    action = plan_action(plan, _empty_grid(), 3, 5, rng, epistemic_map=emap, pos=(8, 7))
    assert action == 2  # MOVE_E
    # No map -> random move, never crashes.
    assert plan_action(plan, _empty_grid(), 3, 5, rng) in (0, 1, 2, 3)


# ------------------------------------------------------------------ arbiter


def test_arbiter_prefers_plan_with_best_forecasted_drives() -> None:
    """Stub forecaster: plan 0 forecasts perfect energy, others forecast
    depletion. With epsilon=0 the arbiter must always pick plan 0."""

    class StubForecaster:
        def __call__(self, h: torch.Tensor, plan_oh: torch.Tensor):
            b = h.shape[0]
            plan = plan_oh.argmax(dim=-1)
            mean = torch.zeros(b, INTERO_DIM)
            mean[:, 0] = torch.where(plan == 0, 1.0, 0.2)  # energy
            return {10: (mean, torch.zeros(b, INTERO_DIM))}

    arb = Arbiter({"epsilon": 0.0, "score_horizon": 10}, StubForecaster(), seed=0)
    plans = arb.select_plans(torch.randn(6, CORE_DIM))
    assert (plans == 0).all()


def test_arbiter_epsilon_explores() -> None:
    class UniformForecaster:
        def __call__(self, h, plan_oh):
            b = h.shape[0]
            return {10: (torch.ones(b, INTERO_DIM), torch.zeros(b, INTERO_DIM))}

    arb = Arbiter({"epsilon": 1.0, "score_horizon": 10}, UniformForecaster(), seed=0)
    picks = np.concatenate([arb.select_plans(torch.randn(8, CORE_DIM)) for _ in range(20)])
    assert len(np.unique(picks)) == NUM_PLANS  # pure exploration hits every plan


# ------------------------------------------------------------------- drives


def test_drive_error_and_homeostatic_reward() -> None:
    cfg = {"setpoint_energy": 1.0, "setpoint_fatigue": 0.0,
           "weight_energy": 1.0, "weight_fatigue": 0.25}
    assert drive_error(1.0, 0.0, cfg) == 0.0
    assert drive_error(0.5, 0.0, cfg) == pytest.approx(0.25)
    assert drive_error(1.0, 1.0, cfg) == pytest.approx(0.25)
    intero = np.array([0.5, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert homeostatic_reward(intero, cfg) == pytest.approx(-0.5)
