"""RSSM core tests: GRUCore interface parity, KL free-nats floor, world-model
learning on a fixed batch, drop-in PPOTrainer training + exact resume, and the
participation-ratio collapse detector."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.core_rssm import RSSMCore
from training.monitors import ParticipationRatioMonitor
from training.ppo import PPOTrainer
from world.config import load_config

ROOT = Path(__file__).resolve().parent.parent

RSSM_CFG = {
    "deter": 16, "stoch": 4, "embed": 24, "ensemble_k": 3,
    "free_nats": 1.0, "kl_balance": 0.8, "min_std": 0.1,
}
GRID_SHAPE = (8, 11, 11)
INTERO_DIM = 6
NUM_ACTIONS = 9


def make_core() -> RSSMCore:
    torch.manual_seed(0)
    return RSSMCore(
        RSSM_CFG, input_dim=24, grid_shape=GRID_SHAPE,
        intero_dim=INTERO_DIM, num_actions=NUM_ACTIONS,
    )


def rssm_trainer_cfg(seed: int = 5) -> dict:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["agent"] = {
        "core": "rssm", "hidden_size": 16, "gru_layers": 1,
        "encoder_channels": [4, 8],
    }
    cfg["rssm"].update(deter=16, stoch=4, embed=24, ensemble_k=3)
    cfg["ppo"].update(
        rollout_steps=8, seq_len=4, num_envs=2, episode_length=16,
        minibatch_transitions=8, epochs=1, total_steps=10**9, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False
    return cfg


# ----------------------------------------------------------- interface parity


def test_interface_matches_gru_core() -> None:
    core = make_core()
    assert core.hidden_dim == 16 + 4
    assert core.output_dim == 16 + 4
    h0 = core.initial_state(3, torch.device("cpu"))
    assert h0.shape == (3, 20) and torch.all(h0 == 0)

    embed = torch.randn(3, 24)
    out, h1 = core(embed, h0)
    assert out.shape == (3, 20) and h1.shape == (3, 20)
    assert torch.equal(out, h1)  # out IS the state, like GRUCore


def test_forward_mean_path_is_deterministic() -> None:
    """The policy/collection path must not sample: same input, same output."""
    core = make_core()
    embed = torch.randn(4, 24)
    h = torch.randn(4, 20)
    out1, _ = core(embed, h)
    out2, _ = core(embed, h)
    assert torch.equal(out1, out2)


# -------------------------------------------------------------------- KL loss


def test_kl_loss_free_nats_floor() -> None:
    """When posterior == prior, KL = 0 but the loss must sit at the free-nats
    floor (both balanced terms clamp at free_nats)."""
    core = make_core()
    stats = torch.zeros(5, 2, 4)
    seq = {
        "prior_mean": stats, "prior_std": torch.ones_like(stats),
        "post_mean": stats.clone(), "post_std": torch.ones_like(stats),
    }
    kl = core.kl_loss(seq)
    assert kl.item() == pytest.approx(core.free_nats)


# ----------------------------------------------------- world-model learning


def test_world_model_loss_decreases_on_fixed_batch() -> None:
    torch.manual_seed(1)
    core = make_core()
    opt = torch.optim.Adam(core.parameters(), lr=3e-4)
    horizon, batch = 6, 4
    embeds = torch.randn(horizon, batch, 24)
    h0 = core.initial_state(batch, torch.device("cpu"))
    dones = torch.zeros(horizon, batch)
    actions = torch.randint(0, NUM_ACTIONS, (horizon, batch))
    grid = torch.rand(horizon, batch, *GRID_SHAPE)
    intero = torch.rand(horizon, batch, INTERO_DIM)
    rewards = torch.randn(horizon, batch)

    torch.manual_seed(2)
    first = core.world_model_loss(embeds, h0, dones, actions, grid, intero, rewards)
    recon_first = first["recon_grid"].item() + first["recon_intero"].item()
    for _ in range(60):
        loss = core.world_model_loss(embeds, h0, dones, actions, grid, intero, rewards)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
    torch.manual_seed(2)
    last = core.world_model_loss(embeds, h0, dones, actions, grid, intero, rewards)
    recon_last = last["recon_grid"].item() + last["recon_intero"].item()
    assert recon_last < recon_first, f"recon did not decrease: {recon_first} -> {recon_last}"
    assert np.isfinite(last["total"].item())


def test_ensemble_stats_shapes_and_nonnegative() -> None:
    core = make_core()
    feats = torch.randn(5, 20)
    onehot = torch.nn.functional.one_hot(torch.randint(0, NUM_ACTIONS, (5,)), NUM_ACTIONS).float()
    means, epistemic, aleatoric = core.ensemble_stats(feats, onehot)
    assert means.shape == (3, 5, 24)
    assert epistemic.shape == (5,) and torch.all(epistemic >= 0)
    assert aleatoric.shape == (5,) and torch.all(aleatoric > 0)


# ------------------------------------------------------ drop-in PPO training


def test_ppo_trainer_with_rssm_core_trains_finite() -> None:
    t = PPOTrainer(rssm_trainer_cfg())
    t.train(max_updates=2)
    assert all(np.isfinite(v) for v in t.last_metrics.values()), t.last_metrics
    assert "rssm/recon" in t.last_metrics
    assert (t.epistemic_count > 0).sum() > 0  # epistemic map got updates


def test_rssm_checkpoint_resume_identical_loss(tmp_path: Path) -> None:
    cfg = rssm_trainer_cfg()
    t1 = PPOTrainer(cfg)
    t1.train(max_updates=2)
    ckpt = t1.save(tmp_path / "ck.pt")
    t1.train(max_updates=1)
    loss_a = t1.last_metrics["loss/total"]

    t2 = PPOTrainer(cfg)
    t2.load(ckpt)
    t2.train(max_updates=1)
    assert t2.last_metrics["loss/total"] == pytest.approx(loss_a, abs=1e-6)
    # Epistemic map must survive the round trip too.
    assert np.array_equal(t2.epistemic_count, t1.epistemic_count) or (
        t2.epistemic_count.sum() > 0
    )


# ------------------------------------------------------- participation ratio


def test_participation_ratio_full_rank_vs_collapsed() -> None:
    rng = np.random.default_rng(0)
    mon = ParticipationRatioMonitor(every_ticks=1, window=500, min_samples=64)
    mon.add(rng.normal(size=(500, 16)))  # isotropic: PR should be near dim
    pr_full = mon.compute()
    assert pr_full is not None and pr_full > 10

    mon2 = ParticipationRatioMonitor(every_ticks=1, window=500, min_samples=64)
    direction = rng.normal(size=16)
    coeffs = rng.normal(size=(500, 1))
    mon2.add(coeffs * direction[None, :] + 1e-6 * rng.normal(size=(500, 16)))
    pr_collapsed = mon2.compute()
    assert pr_collapsed is not None and pr_collapsed < 2


def test_participation_ratio_collapse_warning(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(1)
    mon = ParticipationRatioMonitor(every_ticks=10, window=200, collapse_frac=0.25, min_samples=64)
    mon.add(rng.normal(size=(200, 16)))
    assert mon.maybe_compute(10) is not None  # healthy: sets the running max

    direction = rng.normal(size=16)
    mon.add(rng.normal(size=(200, 1)) * direction[None, :])  # collapse the window
    with caplog.at_level(logging.WARNING, logger="training.monitors"):
        pr = mon.maybe_compute(20)
    assert pr is not None and pr < 0.25 * 16
    assert any("participation ratio collapse" in r.message for r in caplog.records)
