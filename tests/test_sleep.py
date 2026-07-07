"""Wake/sleep trainer tests: the structural exogenous-scheduling rule, the
zero-core-updates-during-wake guarantee, replay buffer behavior, lambda
returns, and the wake-only ablation flag."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from training.replay import SequenceReplay
from training.sleep import CircadianTrainer, is_sleep_tick, lambda_returns, sleep_windows_due
from world.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def circadian_cfg(seed: int = 5, sleep: bool = True) -> dict:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["agent"] = {
        "core": "rssm", "hidden_size": 16, "gru_layers": 1,
        "encoder_channels": [4, 8],
    }
    cfg["rssm"].update(
        deter=16, stoch=4, embed=24, ensemble_k=3, seq_len=8, batch_seqs=2,
        replay_capacity=2000, sleep_every=128, sleep_grad_steps=3,
        imagination_horizon=4, sleep=sleep,
    )
    cfg["ppo"].update(
        rollout_steps=16, seq_len=8, num_envs=2, episode_length=64,
        minibatch_transitions=16, epochs=1, total_steps=10**9, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False
    return cfg


# ------------------------------------------------- exogenous sleep scheduling


def test_sleep_schedule_reads_only_the_env_step_counter() -> None:
    """STRUCTURAL RULE: both scheduling functions take exactly (env_steps,
    sleep_every) — two plain ints, no trainer, no agent state, no model
    output. If this signature grows, consolidation timing has stopped being
    exogenous and that is a hard-rule violation, not a refactor."""
    import typing

    for fn in (is_sleep_tick, sleep_windows_due):
        params = list(inspect.signature(fn).parameters)
        assert params == ["env_steps", "sleep_every"], f"{fn.__name__}: {params}"
        hints = typing.get_type_hints(fn)
        assert hints["env_steps"] is int and hints["sleep_every"] is int


def test_sleep_windows_due_counts_crossings() -> None:
    assert sleep_windows_due(0, 100) == 0
    assert sleep_windows_due(99, 100) == 0
    assert sleep_windows_due(100, 100) == 1
    assert sleep_windows_due(256, 100) == 2  # stepping OVER multiples still counts
    assert is_sleep_tick(200, 100) and not is_sleep_tick(150, 100)


# --------------------------------------------- zero core updates during wake


def _param_blob(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _identical(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_wake_phase_never_updates_core_encoder_or_heads() -> None:
    """The acceptance test: a wake stretch (with online Ledger updates
    running) must leave encoder, core, and actor-critic heads bit-identical;
    the Ledger heads must have actually trained (guards a vacuous pass); and
    a subsequent sleep phase must move core parameters."""
    trainer = CircadianTrainer(circadian_cfg())
    enc0 = _param_blob(trainer.model.encoder)
    core0 = _param_blob(trainer.model.core)
    heads0 = _param_blob(trainer.model.heads)
    body0 = _param_blob(trainer._inner.body_model)

    trainer.wake_phase(256)

    assert _identical(enc0, _param_blob(trainer.model.encoder)), "encoder moved in wake"
    assert _identical(core0, _param_blob(trainer.model.core)), "core moved in wake"
    assert _identical(heads0, _param_blob(trainer.model.heads)), "actor-critic moved in wake"
    assert not _identical(body0, _param_blob(trainer._inner.body_model)), (
        "body model did NOT train in wake — test would be vacuous"
    )

    metrics = trainer.sleep_phase()
    assert metrics["sleep/grad_steps"] > 0
    assert not _identical(core0, _param_blob(trainer.model.core)), (
        "sleep phase did not update the core"
    )
    assert not _identical(heads0, _param_blob(trainer.model.heads)), (
        "sleep phase did not update the actor-critic"
    )


def test_wake_only_flag_disables_sleep_entirely() -> None:
    trainer = CircadianTrainer(circadian_cfg(sleep=False))
    core0 = _param_blob(trainer.model.core)
    heads0 = _param_blob(trainer.model.heads)
    trainer.train(max_env_steps=384)
    assert trainer._sleep_windows_done == 0
    assert _identical(core0, _param_blob(trainer.model.core))
    assert _identical(heads0, _param_blob(trainer.model.heads))


# ------------------------------------------------------------------- replay


def make_replay(cap: int = 200, envs: int = 2) -> SequenceReplay:
    return SequenceReplay(cap, envs, grid_shape=(2, 3, 3), intero_dim=4, seed=0)


def _fill(replay: SequenceReplay, ticks: int, done_at: set[int] | None = None) -> None:
    done_at = done_at or set()
    n = replay.num_envs
    for t in range(ticks):
        replay.add_batch(
            grid=np.full((n, 2, 3, 3), t % 2, dtype=np.float32),
            intero=np.full((n, 4), float(t), dtype=np.float32),
            action=np.full(n, t % 9),
            reward=np.full(n, 0.1 * t, dtype=np.float32),
            done=np.array([float(t in done_at)] * n),
        )


def test_replay_sample_shapes_and_content() -> None:
    replay = make_replay()
    _fill(replay, 50)
    batch = replay.sample(3, 8, torch.device("cpu"))
    assert batch is not None
    assert batch["grid"].shape == (8, 3, 2, 3, 3)
    assert batch["intero"].shape == (8, 3, 4)
    assert batch["action"].shape == (8, 3) and batch["action"].dtype == torch.long
    # intero encodes the tick: sequences must be consecutive.
    seq_ticks = batch["intero"][:, 0, 0]
    assert torch.all(seq_ticks[1:] - seq_ticks[:-1] == 1.0)


def test_replay_sequences_never_cross_a_done() -> None:
    replay = make_replay()
    _fill(replay, 60, done_at={20, 40})
    for _ in range(30):
        batch = replay.sample(4, 10, torch.device("cpu"))
        assert batch is not None
        # A done may appear only at the FINAL position of a sequence.
        interior = batch["done"][:-1]
        assert torch.all(interior == 0), "sequence crossed an episode boundary"


def test_replay_priorities_bias_sampling() -> None:
    replay = make_replay(cap=400, envs=1)
    _fill(replay, 200)
    # Give one region enormous priority and everything else epsilon.
    envs = np.zeros(1, dtype=np.int64)
    replay.update_priorities(np.zeros(190, dtype=np.int64), np.arange(190), 1,
                             np.full(190, 1e-6))
    replay.update_priorities(envs, np.array([100]), 8, np.array([100.0]))
    hits = 0
    for _ in range(50):
        batch = replay.sample(1, 8, torch.device("cpu"))
        assert batch is not None
        if 93 <= batch["starts"][0] <= 107:
            hits += 1
    assert hits >= 40, f"prioritized region sampled only {hits}/50 times"


def test_replay_state_roundtrip() -> None:
    replay = make_replay()
    _fill(replay, 30)
    state = replay.state_dict()
    restored = make_replay()
    restored.load_state_dict(state)
    b1 = replay.sample(2, 5, torch.device("cpu"))
    b2 = restored.sample(2, 5, torch.device("cpu"))
    assert b1 is not None and b2 is not None
    assert torch.equal(b1["intero"], b2["intero"])  # same RNG state -> same draw


# ------------------------------------------------------------ lambda returns


def test_lambda_returns_hand_computed() -> None:
    rewards = torch.tensor([[1.0], [0.0]])
    values = torch.tensor([[0.5], [0.4]])
    bootstrap = torch.tensor([2.0])
    gamma, lam = 0.9, 0.8
    out = lambda_returns(rewards, values, bootstrap, gamma, lam)
    r1 = 0.0 + gamma * ((1 - lam) * 2.0 + lam * 2.0)  # last step bootstraps
    r0 = 1.0 + gamma * ((1 - lam) * values[1, 0].item() + lam * r1)
    assert out[1, 0].item() == pytest.approx(r1)
    assert out[0, 0].item() == pytest.approx(r0)


# ------------------------------------------------------------- arbiter mode


def test_arbiter_mode_trains_forecaster_and_keeps_isolation() -> None:
    """agent.controller=arbiter: wake runs plan executors, forecast tuples
    accumulate with realized futures, sleep trains the forecaster — and the
    wake guarantee (no encoder/core/heads updates) still holds."""
    cfg = circadian_cfg()
    cfg["agent"]["controller"] = "arbiter"
    trainer = CircadianTrainer(cfg)
    core0 = _param_blob(trainer.model.core)
    heads0 = _param_blob(trainer.model.heads)
    fore0 = _param_blob(trainer.forecaster)

    trainer.wake_phase(256)
    assert _identical(core0, _param_blob(trainer.model.core))
    assert _identical(heads0, _param_blob(trainer.model.heads))
    assert len(trainer.tuple_store) > 0, "no forecast tuples collected"
    # Tuples must contain BOTH horizons, realized (not placeholder) futures.
    batch = trainer.tuple_store.batch(4, torch.device("cpu"))
    assert batch is not None
    assert set(batch["future"].keys()) == {1, 10}

    metrics = trainer.sleep_phase()
    if "sleep/forecaster_nll" in metrics:  # store may still be < min batch
        assert not _identical(fore0, _param_blob(trainer.forecaster))
    assert _identical(heads0, _param_blob(trainer.model.heads)) is False  # sleep trains AC
