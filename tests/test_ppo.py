"""Recurrent-PPO unit tests targeting the classic silent-failure modes:
hidden-state masking at done boundaries, GAE bootstrapping across dones,
sequence minibatching state, and exact checkpoint-resume determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agent.core_gru import GRUCore
from agent.encoder import ObsEncoder
from training.ppo import PPOTrainer, compute_gae, replay_core
from training.reward import compute_reward
from training.vecenv import VecWorld
from world import engine
from world.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def tiny_cfg(seed: int = 5) -> dict:
    """Small but real config for fast trainer tests."""
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["agent"] = {"hidden_size": 16, "gru_layers": 1, "encoder_channels": [4, 8]}
    cfg["ppo"].update(
        rollout_steps=8, seq_len=4, num_envs=2, episode_length=16,
        minibatch_transitions=8, epochs=1, total_steps=10**9,
        anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False
    return cfg


# ------------------------------------------------------------- done masking


def test_gru_hidden_zeroed_at_done_boundary_vs_hand_forward() -> None:
    """Scripted 2-env, 3-step sequence with a mid-sequence done (env 0 at t=1).

    The hidden state entering t=2 for env 0 must be exactly zero; every output
    must match a hand-computed GRUCell forward pass.
    """
    torch.manual_seed(0)
    core = GRUCore({"hidden_size": 8}, input_dim=4)
    x = torch.randn(3, 2, 4)
    h0 = torch.randn(2, 8)
    dones = torch.zeros(3, 2)
    dones[1, 0] = 1.0  # env 0 episode ends after step 1

    outs = replay_core(core, x, h0, dones)

    with torch.no_grad():
        o0 = core.cells[0](x[0], h0)
        o1 = core.cells[0](x[1], o0)  # no done at step 0 -> hidden carried
        h2_in = o1 * (1.0 - dones[1]).unsqueeze(-1)
        o2 = core.cells[0](x[2], h2_in)

    assert torch.allclose(outs[0], o0, atol=1e-6)
    assert torch.allclose(outs[1], o1, atol=1e-6)
    assert torch.allclose(outs[2], o2, atol=1e-6)
    # Explicit boundary check: env 0's input hidden at t=2 was exactly zero...
    assert torch.equal(h2_in[0], torch.zeros(8))
    with torch.no_grad():
        from_zero = core.cells[0](x[2, 0:1], torch.zeros(1, 8))[0]
    assert torch.allclose(outs[2, 0], from_zero, atol=1e-6)
    # ...while env 1 (no done) was NOT reset.
    assert not torch.allclose(outs[2, 1], core.cells[0](x[2, 1:2], torch.zeros(1, 8))[0])


# --------------------------------------------------------------------- GAE


def test_gae_hand_computed_with_mid_sequence_done() -> None:
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    next_value = torch.tensor([0.9])
    gamma, lam = 0.9, 0.8

    adv = compute_gae(rewards, values, dones, next_value, gamma, lam)

    a2 = 2.0 + 0.9 * 0.9 - 0.3            # bootstraps into next_value
    a1 = 0.0 - 0.4                          # done: no bootstrap, no carry-over
    a0 = (1.0 + 0.9 * 0.4 - 0.5) + 0.9 * 0.8 * a1
    expected = torch.tensor([[a0], [a1], [a2]])
    assert torch.allclose(adv, expected, atol=1e-6)


# ----------------------------------------------------------------- encoder


def test_encoder_output_shape() -> None:
    enc = ObsEncoder(
        {"encoder_channels": [4, 8]},
        grid_channels=8, intero_dim=6, embed_dim=32, window=11,
    )
    out = enc(torch.zeros(5, 8, 11, 11), torch.zeros(5, 6))
    assert out.shape == (5, 32)


def test_gru_core_multi_layer_matches_stacked_cells() -> None:
    """gru_layers > 1: hidden_dim is layer_size * gru_layers; output stays
    layer_size (top layer only); zero-layers behavior matches a manual stack."""
    torch.manual_seed(1)
    core = GRUCore({"hidden_size": 6, "gru_layers": 2}, input_dim=4)
    assert core.hidden_dim == 12
    assert core.output_dim == 6

    x = torch.randn(3, 4)
    h0 = torch.zeros(3, 12)
    out, h1 = core(x, h0)
    assert out.shape == (3, 6)
    assert h1.shape == (3, 12)

    with torch.no_grad():
        layer0_out = core.cells[0](x, h0[:, :6])
        layer1_out = core.cells[1](layer0_out, h0[:, 6:])
    assert torch.allclose(out, layer1_out, atol=1e-6)
    assert torch.allclose(h1[:, :6], layer0_out, atol=1e-6)
    assert torch.allclose(h1[:, 6:], layer1_out, atol=1e-6)


# ------------------------------------------------------------------ reward


def test_reward_terms() -> None:
    rcfg = {"eat": 1.0, "step_cost": 0.001, "deficit_threshold": 0.2,
            "deficit_penalty": 0.01}
    assert compute_reward(engine.NOOP, True, 0.9, rcfg) == pytest.approx(-0.001)
    assert compute_reward(engine.EAT, True, 0.9, rcfg) == pytest.approx(0.999)
    assert compute_reward(engine.EAT, False, 0.9, rcfg) == pytest.approx(-0.001)
    assert compute_reward(engine.NOOP, True, 0.1, rcfg) == pytest.approx(-0.011)
    assert compute_reward(engine.EAT, True, 0.1, rcfg) == pytest.approx(0.989)


# ------------------------------------------------------------------ vecenv


def test_vecenv_shapes_dones_and_state_roundtrip() -> None:
    cfg = tiny_cfg()
    cfg["ppo"]["episode_length"] = 5
    vec = VecWorld(cfg)
    obs = vec.observe()
    n = cfg["ppo"]["num_envs"]
    assert obs["grid"].shape == (n, *vec.grid_shape)
    assert obs["intero"].shape == (n, 6)

    noops = np.full(n, engine.NOOP)
    for step in range(1, 6):
        obs, rewards, dones, infos = vec.step(noops)
        assert rewards.shape == (n,) and dones.shape == (n,)
        expected = 1.0 if step == 5 else 0.0
        assert list(dones) == [expected] * n, f"wrong dones at step {step}"
    assert list(vec.ep_steps) == [0] * n  # reset after episode boundary

    state = vec.get_state()
    hashes = [w.state_hash() for w in vec.worlds]
    vec.step(noops)  # mutate past the saved state
    vec.set_state(state)
    assert [w.state_hash() for w in vec.worlds] == hashes


# ------------------------------------------------- exact-resume determinism


def test_checkpoint_resume_identical_step_and_loss(tmp_path: Path) -> None:
    """Save mid-training, resume in a fresh trainer: identical global step and
    matching loss on the next batch (fixed seed, CPU)."""
    cfg = tiny_cfg()
    steps_per_update = cfg["ppo"]["rollout_steps"] * cfg["ppo"]["num_envs"]

    t1 = PPOTrainer(cfg)
    t1.train(max_updates=2)
    assert t1.global_step == 2 * steps_per_update
    ckpt_path = t1.save(tmp_path / "ck.pt")
    t1.train(max_updates=1)
    loss_a = t1.last_metrics["loss/total"]
    reward_a = t1.last_metrics["reward/rollout"]

    t2 = PPOTrainer(cfg)
    t2.load(ckpt_path)
    assert t2.global_step == 2 * steps_per_update
    t2.train(max_updates=1)

    assert t2.global_step == t1.global_step
    assert t2.last_metrics["reward/rollout"] == pytest.approx(reward_a, abs=1e-7)
    assert t2.last_metrics["loss/total"] == pytest.approx(loss_a, abs=1e-6)


# ------------------------------------------------ architecture-B (no ledger)


def test_use_ledger_features_false_runs_and_shrinks_policy_input() -> None:
    """capability_shift battery's architecture-B control: the body model still
    trains, but the policy head's input is just the raw core output."""
    cfg = tiny_cfg()
    cfg["agent"]["use_ledger_features"] = False
    t = PPOTrainer(cfg)
    assert t.model.heads.pi.in_features == t.model.core.output_dim
    t.train(max_updates=2)  # must not raise
    assert "ledger/body_nll" in t.last_metrics  # body model still trained/logged
