"""Gradient-isolation tests for the Ledger (CLAUDE.md Hard rules).

Directions covered, all load-bearing:
1. The body model's own CE+BCE+MSE loss must never reach the encoder/GRU core
   (its input is h.detach()).
2. The policy/value loss must never reach the body model (its output is
   detached before being concatenated into the policy's input features).
3. The attribution classifier's own CE loss must never reach the encoder/GRU
   core OR the body model (its three scalar inputs are derived from the body
   model's already-detached prediction and from plain observed values).
4. The forecaster's NLL loss must never reach the encoder/core or the
   actor-critic heads (its input is h.detach() + a plan one-hot).

Each test also asserts the OPPOSITE side does receive gradient, to guard
against a vacuously-passing test (e.g. a broken forward pass that produces no
gradient anywhere).
"""

from __future__ import annotations

from pathlib import Path

import torch

from agent.core_gru import GRUCore
from agent.encoder import ObsEncoder
from agent.policy import ActorCritic
from ledger.attribution import AttributionHead, compute_attribution_loss, pseudo_label, residual_features
from ledger.body_model import BodyModel, build_policy_features, compute_body_losses
from training.ppo import PPOTrainer
from world.config import load_config
from world.engine import NOOP

NUM_ACTIONS = 9
ROOT = Path(__file__).resolve().parent.parent


def _no_grad_reached(params) -> bool:
    return all(p.grad is None or torch.all(p.grad == 0) for p in params)


def _some_grad_reached(params) -> bool:
    return any(p.grad is not None and torch.any(p.grad != 0) for p in params)


def test_body_model_loss_does_not_reach_encoder_or_core() -> None:
    torch.manual_seed(0)
    batch, grid_channels, window, intero_dim = 6, 5, 11, 6

    encoder = ObsEncoder(
        {"encoder_channels": [4, 8]}, grid_channels, intero_dim, embed_dim=16, window=window
    )
    core = GRUCore({"hidden_size": 16}, input_dim=16)
    body = BodyModel({"body_hidden": [32, 32]}, core_dim=core.output_dim, num_actions=NUM_ACTIONS)

    grid = torch.randn(batch, grid_channels, window, window)
    intero = torch.randn(batch, intero_dim)
    h0 = core.initial_state(batch, torch.device("cpu"))

    embed = encoder(grid, intero)
    out, _ = core(embed, h0)
    h_detached = out.detach()
    assert not h_detached.requires_grad

    action_onehot = torch.nn.functional.one_hot(torch.randint(0, NUM_ACTIONS, (batch,)), NUM_ACTIONS).float()
    outputs = body.predict_action(h_detached, action_onehot)

    real_dpos_class = torch.randint(0, 5, (batch,))
    real_success = torch.randint(0, 2, (batch,)).float()
    real_denergy = torch.randn(batch)
    losses = compute_body_losses(outputs, real_dpos_class, real_success, real_denergy)
    losses["total"].backward()

    core_and_encoder_params = list(encoder.parameters()) + list(core.parameters())
    assert _no_grad_reached(core_and_encoder_params), (
        "body-model loss leaked gradient into encoder/core parameters"
    )
    # Sanity: the body model itself must have actually received gradient,
    # otherwise the assertion above would pass vacuously.
    assert _some_grad_reached(body.parameters())


def test_policy_loss_does_not_reach_body_model() -> None:
    torch.manual_seed(1)
    batch, core_dim = 6, 16

    body = BodyModel({"body_hidden": [32, 32]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    heads = ActorCritic({}, core_dim + 2 * NUM_ACTIONS, NUM_ACTIONS)

    # Stand-in for the real encoder/core output: requires_grad, as it would in
    # training (build_policy_features must not touch its grad path either).
    core_out = torch.randn(batch, core_dim, requires_grad=True)
    features, _ = build_policy_features(core_out, body)
    dist, value = heads(features)
    action = torch.randint(0, NUM_ACTIONS, (batch,))
    loss = -dist.log_prob(action).mean() + value.pow(2).mean()
    loss.backward()

    assert core_out.grad is not None and torch.any(core_out.grad != 0), (
        "sanity check failed: core output itself received no gradient"
    )
    assert _no_grad_reached(body.parameters()), (
        "policy/value loss leaked gradient into body-model parameters"
    )


def test_attribution_loss_does_not_reach_core_or_body_model() -> None:
    torch.manual_seed(2)
    batch, grid_channels, window, intero_dim, core_dim = 6, 5, 11, 6, 16

    encoder = ObsEncoder(
        {"encoder_channels": [4, 8]}, grid_channels, intero_dim, embed_dim=core_dim, window=window
    )
    core = GRUCore({"hidden_size": core_dim}, input_dim=core_dim)
    body = BodyModel({"body_hidden": [16, 16]}, core_dim=core_dim, num_actions=NUM_ACTIONS)
    attribution = AttributionHead({"tau_pos": 0.5, "tau_energy": 0.01, "lr": 1e-3})

    grid = torch.randn(batch, grid_channels, window, window)
    intero = torch.randn(batch, intero_dim)
    h0 = core.initial_state(batch, torch.device("cpu"))
    embed = encoder(grid, intero)
    out, _ = core(embed, h0)

    action = torch.randint(0, NUM_ACTIONS, (batch,))
    action_onehot = torch.nn.functional.one_hot(action, NUM_ACTIONS).float()
    body_out = body.predict_action(out.detach(), action_onehot)  # body model's own forward, detached input
    pred_dpos_class = body_out["dpos_logits"].argmax(dim=-1).detach()
    pred_denergy = body_out["denergy"].detach()

    observed_dpos_class = torch.randint(0, 5, (batch,))
    observed_denergy = torch.randn(batch)
    features = residual_features(
        pred_dpos_class, observed_dpos_class, pred_denergy, observed_denergy, action, NOOP
    )
    labels = pseudo_label(features, tau_pos=0.5, tau_energy=0.01)
    logits = attribution(features)
    loss = compute_attribution_loss(logits, labels)
    loss.backward()

    upstream_params = list(encoder.parameters()) + list(core.parameters()) + list(body.parameters())
    assert _no_grad_reached(upstream_params), (
        "attribution loss leaked gradient into encoder/core/body-model parameters"
    )
    assert _some_grad_reached(attribution.parameters())


def test_integration_ledger_updates_never_touch_policy_params_via_real_trainer() -> None:
    """Same guarantee as the tests above, but through the REAL PPOTrainer
    wiring (collect_rollout -> update_body_model -> update_attribution_model),
    not just hand-assembled synthetic tensors — catches integration-level
    leaks a unit test could miss (e.g. an optimizer accidentally sweeping
    Ledger parameters, or a submodule registered where it shouldn't be)."""
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = 3
    cfg["device"] = "cpu"
    cfg["agent"] = {"hidden_size": 16, "gru_layers": 1, "encoder_channels": [4, 8]}
    cfg["ppo"].update(
        rollout_steps=8, seq_len=4, num_envs=2, episode_length=16,
        minibatch_transitions=8, epochs=1, total_steps=10**9, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False

    trainer = PPOTrainer(cfg)
    policy_params = list(trainer.model.parameters())
    assert all(p.grad is None for p in policy_params)  # freshly constructed, no grad yet

    buf = trainer.collect_rollout()  # @torch.no_grad(): cannot populate any .grad
    trainer.update_body_model(buf)
    trainer.update_attribution_model(buf)

    assert all(p.grad is None for p in policy_params), (
        "body-model/attribution updates left gradient on encoder/core/policy parameters "
        "-- Ledger optimizers must never touch them"
    )
    # Sanity: the Ledger heads themselves DID get gradient (not a vacuous pass).
    assert _some_grad_reached(trainer.body_model.parameters())
    assert _some_grad_reached(trainer.attribution_model.parameters())


def test_forecaster_loss_does_not_reach_core_or_heads() -> None:
    """Direction 4: forecaster NLL backward must leave encoder, core, and
    actor-critic heads untouched; the forecaster itself must train."""
    from agent.drives import NUM_PLANS
    from ledger.forecaster import Forecaster, forecaster_nll

    torch.manual_seed(4)
    batch, grid_channels, window, intero_dim = 5, 5, 11, 6
    encoder = ObsEncoder(
        {"encoder_channels": [4, 8]}, grid_channels, intero_dim, embed_dim=16, window=window
    )
    core = GRUCore({"hidden_size": 16}, input_dim=16)
    heads = ActorCritic({}, 16 + 2 * NUM_ACTIONS, NUM_ACTIONS)
    forecaster = Forecaster(
        {"forecaster_hidden": [16, 16], "horizons": [1, 10]},
        core_dim=16, intero_dim=intero_dim, num_plans=NUM_PLANS,
    )

    grid = torch.randn(batch, grid_channels, window, window)
    intero = torch.randn(batch, intero_dim)
    out, _ = core(encoder(grid, intero), core.initial_state(batch, torch.device("cpu")))
    plan_oh = torch.nn.functional.one_hot(
        torch.randint(0, NUM_PLANS, (batch,)), NUM_PLANS
    ).float()

    fc = forecaster(out.detach(), plan_oh)
    target = torch.rand(batch, intero_dim)
    loss = forecaster_nll(fc[10][0], fc[10][1], target) + forecaster_nll(
        fc[1][0], fc[1][1], target
    )
    loss.backward()

    frozen = list(encoder.parameters()) + list(core.parameters()) + list(heads.parameters())
    assert _no_grad_reached(frozen), "forecaster loss leaked into encoder/core/heads"
    assert _some_grad_reached(forecaster.parameters())
