"""Mirror tests: pose head learning, the monitored-never-minimized rule,
trigger state machine, probe script vs real engine outcomes, probe-driven
body-model refresh, and MPC output validity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agent.core_rssm import RSSMCore
from ledger.body_model import BodyModel
from ledger.mirror import DELIBERATING, NORMAL, PROBE_SCRIPT, PROBING, MirrorMonitor
from world.config import load_config
from world.engine import World

ROOT = Path(__file__).resolve().parent.parent

RSSM_CFG = {
    "deter": 16, "stoch": 4, "embed": 24, "ensemble_k": 3,
    "free_nats": 1.0, "kl_balance": 0.8, "min_std": 0.1,
}
GRID_SHAPE = (8, 11, 11)
NUM_ACTIONS = 9
MIRROR_CFG = {"enabled": True, "threshold": 2.0, "mpc_ticks": 3,
              "mpc_horizon": 3, "mpc_candidates": 8}


def make_core() -> RSSMCore:
    torch.manual_seed(0)
    return RSSMCore(RSSM_CFG, input_dim=24, grid_shape=GRID_SHAPE,
                    intero_dim=6, num_actions=NUM_ACTIONS)


def make_body(core_dim: int = 20) -> BodyModel:
    torch.manual_seed(1)
    return BodyModel({"body_hidden": [16, 16]}, core_dim=core_dim, num_actions=NUM_ACTIONS)


# ------------------------------------------------------------------ pose head


def test_pose_head_learns_position_mapping() -> None:
    """world_model_loss's pose term must fall when positions are a learnable
    function of the observation embedding."""
    torch.manual_seed(2)
    core = make_core()
    opt = torch.optim.Adam(core.parameters(), lr=1e-3)
    horizon, batch = 6, 8
    positions = torch.rand(horizon, batch, 2)
    embeds = positions.repeat(1, 1, 12)  # embed encodes the position
    h0 = core.initial_state(batch, torch.device("cpu"))
    dones = torch.zeros(horizon, batch)
    actions = torch.randint(0, NUM_ACTIONS, (horizon, batch))
    grid = torch.zeros(horizon, batch, *GRID_SHAPE)
    intero = torch.zeros(horizon, batch, 6)

    torch.manual_seed(3)
    first = core.world_model_loss(embeds, h0, dones, actions, grid, intero,
                                  positions=positions)["pose_mse"].item()
    for _ in range(120):
        loss = core.world_model_loss(embeds, h0, dones, actions, grid, intero,
                                     positions=positions)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
    torch.manual_seed(3)
    last = core.world_model_loss(embeds, h0, dones, actions, grid, intero,
                                 positions=positions)["pose_mse"].item()
    assert last < first * 0.5, f"pose_mse did not learn: {first} -> {last}"


# ----------------------------------------------- monitored, never minimized


def test_divergence_is_monitored_never_minimized() -> None:
    """Structural rule: divergence is computed under no_grad and returns
    plain numpy — it CANNOT appear in any autograd graph, so no loss term can
    contain it. Feed requires_grad features and assert the output is graph-
    free numpy and that computing it left no grad on any parameter."""
    core = make_core()
    body = make_body()
    mirror = MirrorMonitor(MIRROR_CFG, num_envs=3, world_size=32)

    features = torch.randn(3, 20, requires_grad=True)
    actions = torch.randint(0, NUM_ACTIONS, (3,))
    div = mirror.divergence(core, body, features, actions, [(5, 5)] * 3)

    assert isinstance(div, np.ndarray)  # numpy: no grad_fn possible
    assert div.shape == (3,) and np.all(div >= 0)
    assert features.grad is None  # nothing backpropagated
    for p in list(core.parameters()) + list(body.parameters()):
        assert p.grad is None, "divergence computation touched a parameter grad"


# -------------------------------------------------------------- state machine


def test_state_machine_normal_probe_mpc_cycle() -> None:
    mirror = MirrorMonitor(MIRROR_CFG, num_envs=2, world_size=32)
    core, body = make_core(), make_body()
    feats = torch.randn(2, 20)

    # Env 0 crosses the threshold; env 1 stays calm.
    mirror.step_state(np.array([10.0, 0.1]))
    assert mirror.state[0] == PROBING and mirror.state[1] == NORMAL
    assert mirror.trigger_count == 1

    # Probe phase: overrides follow the script for exactly len(PROBE_SCRIPT) ticks.
    executed = []
    for _ in range(len(PROBE_SCRIPT)):
        acts = mirror.override_actions(torch.zeros(2, dtype=torch.long), core, body, feats)
        executed.append(int(acts[0]))
        mirror.step_state(np.array([10.0, 0.1]))  # still diverged; mid-probe ignored
    assert executed == [a for a, _ in PROBE_SCRIPT]
    assert mirror.state[0] == DELIBERATING
    assert mirror.just_finished_probing[0]

    # MPC phase for mpc_ticks, then back to NORMAL (divergence low now).
    for _ in range(MIRROR_CFG["mpc_ticks"]):
        acts = mirror.override_actions(torch.zeros(2, dtype=torch.long), core, body, feats)
        assert 0 <= int(acts[0]) < NUM_ACTIONS
        mirror.step_state(np.array([0.1, 0.1]))
    assert mirror.state[0] == NORMAL
    # Env 1 was never overridden.
    assert mirror.state[1] == NORMAL and mirror.trigger_count == 1


def test_disabled_mirror_never_triggers_or_overrides() -> None:
    mirror = MirrorMonitor(dict(MIRROR_CFG, enabled=False), num_envs=2, world_size=32)
    mirror.step_state(np.array([100.0, 100.0]))
    assert (mirror.state == NORMAL).all() and mirror.trigger_count == 0
    default = torch.tensor([7, 7])
    out = mirror.override_actions(default, make_core(), make_body(), torch.randn(2, 20))
    assert torch.equal(out, default)


# ------------------------------------------------------------------- probes


def test_probe_script_expected_outcomes_match_engine() -> None:
    """NOOP and REST must realize 'stay' in the real engine (the known
    expected outcomes the self-check relies on)."""
    cfg = load_config(ROOT / "configs" / "base.yaml")
    w = World(cfg)
    w.set_agent_pos(0, 32, 32)
    for action, expected_cls in PROBE_SCRIPT[:2]:  # NOOP, REST
        _, infos = w.step([action])
        assert infos[0]["realized"]["dpos"] == (0, 0)
        assert expected_cls == 0


def test_probe_refresh_updates_body_model() -> None:
    mirror = MirrorMonitor(MIRROR_CFG, num_envs=1, world_size=32)
    body = make_body()
    opt = torch.optim.Adam(body.parameters(), lr=1e-2)
    for action, cls in PROBE_SCRIPT:
        mirror.record_probe_result(0, torch.randn(20), action, cls, 1.0, -0.001)
    before = [p.detach().clone() for p in body.parameters()]
    loss = mirror.refresh_body_model(body, opt, 0)
    assert loss is not None and np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, body.parameters()))
    assert mirror.refresh_body_model(body, opt, 0) is None  # queue cleared


# ---------------------------------------------------------------------- MPC


def test_mpc_returns_valid_actions() -> None:
    mirror = MirrorMonitor(MIRROR_CFG, num_envs=4, world_size=32)
    core, body = make_core(), make_body()
    with torch.no_grad():
        actions = mirror._mpc_plan(core, body, torch.randn(4, 20))
    assert actions.shape == (4,)
    assert all(0 <= int(a) < NUM_ACTIONS for a in actions)


def test_warmup_disarms_responses_then_arms() -> None:
    """Responses stay disarmed for warmup_ticks step_state calls (divergence
    is still recorded by the caller); the first post-warmup crossing arms."""
    mirror = MirrorMonitor(dict(MIRROR_CFG, warmup_ticks=5), num_envs=1, world_size=32)
    for _ in range(5):
        mirror.step_state(np.array([100.0]))
        assert mirror.state[0] == NORMAL and mirror.trigger_count == 0
    mirror.step_state(np.array([100.0]))  # first armed tick
    assert mirror.state[0] == PROBING and mirror.trigger_count == 1
