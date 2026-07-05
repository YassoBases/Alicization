"""Gradient-isolation tests for the Ledger (CLAUDE.md Hard rules).

Two directions, both load-bearing:
1. The body model's own CE+BCE+MSE loss must never reach the encoder/GRU core
   (its input is h.detach()).
2. The policy/value loss must never reach the body model (its output is
   detached before being concatenated into the policy's input features).

Each test also asserts the OPPOSITE side does receive gradient, to guard
against a vacuously-passing test (e.g. a broken forward pass that produces no
gradient anywhere).
"""

from __future__ import annotations

import torch

from agent.core_gru import GRUCore
from agent.encoder import ObsEncoder
from agent.policy import ActorCritic
from ledger.body_model import BodyModel, build_policy_features, compute_body_losses

NUM_ACTIONS = 9


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
    features = build_policy_features(core_out, body)
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
