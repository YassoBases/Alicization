"""5k-step training smoke: tiny config completes, losses finite, reward sane."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from training.ppo import PPOTrainer
from world.config import load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_train_smoke_5k_steps_finite() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = 7
    cfg["device"] = "cpu"
    cfg["agent"] = {"hidden_size": 32, "gru_layers": 1, "encoder_channels": [8, 16]}
    cfg["ppo"].update(
        rollout_steps=32, seq_len=16, num_envs=2, episode_length=256,
        minibatch_transitions=32, epochs=1, total_steps=5_000, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False

    trainer = PPOTrainer(cfg)
    trainer.train()

    assert trainer.global_step >= 5_000
    metrics = trainer.last_metrics
    assert metrics, "no metrics recorded"
    for tag, value in metrics.items():
        assert math.isfinite(value), f"{tag} is not finite: {value}"
    assert not math.isnan(metrics["reward/rollout"])
    assert all(math.isfinite(r) for r in trainer.reward_history)
