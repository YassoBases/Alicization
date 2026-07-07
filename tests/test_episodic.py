"""Episodic memory tests: write-gate rate control, retrieval ranking,
importance pruning, per-env isolation + memory_pressure through the real
trainer, and state round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memory.episodic import EpisodicMemory, SurpriseWriteGate
from training.ppo import PPOTrainer
from world.config import load_config

ROOT = Path(__file__).resolve().parent.parent

MEM_CFG = {
    "capacity": 50, "latent_dim": 8, "retrieve_k": 3,
    "w_sim": 1.0, "w_spatial": 1.0, "spatial_sigma": 8.0,
    "importance_tau": 1000.0, "write_rate_target": 1.0 / 200.0,
}


def memory_trainer_cfg(seed: int = 5) -> dict:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["agent"] = {
        "core": "rssm", "hidden_size": 16, "gru_layers": 1,
        "encoder_channels": [4, 8],
    }
    cfg["rssm"].update(deter=16, stoch=4, embed=24, ensemble_k=3)
    cfg["memory"] = dict(MEM_CFG, enabled=True, capacity=64)
    cfg["ppo"].update(
        rollout_steps=8, seq_len=4, num_envs=2, episode_length=16,
        minibatch_transitions=8, epochs=1, total_steps=10**9, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False
    return cfg


# -------------------------------------------------------- write-gate control


def test_write_gate_converges_to_target_rate() -> None:
    """Feed lognormal surprises for 40k ticks; the realized write rate must
    settle near 1/200 (within a factor of two) regardless of the surprise
    scale or the initial threshold."""
    rng = np.random.default_rng(0)
    for scale, init_thr in ((1.0, 1.0), (100.0, 0.001), (0.01, 50.0)):
        gate = SurpriseWriteGate(target_rate=1.0 / 200.0, init_threshold=init_thr)
        writes = 0
        total = 40_000
        burn_in = 10_000
        for i in range(total):
            s = float(rng.lognormal(mean=0.0, sigma=1.0)) * scale
            wrote = gate.update(s)
            if i >= burn_in:
                writes += int(wrote)
        rate = writes / (total - burn_in)
        assert 1 / 400 <= rate <= 1 / 100, (
            f"scale={scale} init_thr={init_thr}: rate {rate:.5f} not near 1/200"
        )


def test_write_gate_adapts_when_surprise_distribution_shifts() -> None:
    """A sharpening world model shrinks surprises; the gate must follow."""
    rng = np.random.default_rng(1)
    gate = SurpriseWriteGate(target_rate=1.0 / 200.0)
    for _ in range(20_000):
        gate.update(float(rng.lognormal(0.0, 1.0)))
    thr_before = gate.threshold
    writes = 0
    for i in range(40_000):
        wrote = gate.update(float(rng.lognormal(0.0, 1.0)) * 0.01)  # 100x smaller
        if i >= 20_000:
            writes += int(wrote)
    assert gate.threshold < thr_before  # threshold chased the distribution down
    rate = writes / 20_000
    assert 1 / 400 <= rate <= 1 / 100, f"post-shift rate {rate:.5f}"


# ---------------------------------------------------------- retrieval ranking


def test_retrieval_ranks_by_similarity_and_distance() -> None:
    mem = EpisodicMemory(dict(MEM_CFG, retrieve_k=1), core_dim=8, seed=0)
    mem.projection = np.eye(8, 8, dtype=np.float32)  # identity: latent == state

    q = np.zeros(8, dtype=np.float32); q[0] = 1.0
    near_similar = q.copy()
    far_similar = q.copy()
    near_different = np.zeros(8, dtype=np.float32); near_different[1] = 1.0

    mem._insert(near_similar, (10, 10), tick=0, surprise=1.0, summary=None)
    mem._insert(far_similar, (60, 60), tick=0, surprise=1.0, summary=None)
    mem._insert(near_different, (10, 10), tick=0, surprise=1.0, summary=None)

    scores = mem.scores(q, pos=(10, 10))
    # near+similar must beat both far+similar and near+different.
    assert scores[0] > scores[1] and scores[0] > scores[2]

    summary, top = mem.retrieve(q, pos=(10, 10))
    assert top[0] == 0
    assert np.allclose(summary, near_similar)

    # Far away, similarity dominates: far_similar outranks near_different.
    scores_far = mem.scores(q, pos=(60, 60))
    assert scores_far[1] > scores_far[2]


def test_retrieval_reliability_multiplier() -> None:
    """A reliability_fn multiplies scores (stage-5b hook)."""
    mem = EpisodicMemory(dict(MEM_CFG, retrieve_k=1, w_spatial=0.0), core_dim=8, seed=0)
    mem.projection = np.eye(8, 8, dtype=np.float32)
    q = np.ones(8, dtype=np.float32)
    mem._insert(q.copy(), (0, 0), 0, 1.0, None)   # identical latent
    mem._insert(q * 0.9, (0, 0), 0, 1.0, None)    # slightly less similar

    _, top_plain = mem.retrieve(q, (0, 0))
    assert top_plain[0] == 0
    # Kill entry 0's reliability -> entry 1 must win.
    rel = lambda idx: np.where(idx == 0, 0.01, 1.0)  # noqa: E731
    _, top_rel = mem.retrieve(q, (0, 0), reliability_fn=rel)
    assert top_rel[0] == 1


def test_empty_memory_returns_zero_summary() -> None:
    mem = EpisodicMemory(MEM_CFG, core_dim=8, seed=0)
    summary, top = mem.retrieve(np.ones(8, dtype=np.float32), (0, 0))
    assert np.all(summary == 0) and len(top) == 0
    assert mem.pressure() == 0.0


# ------------------------------------------------------------------- pruning


def test_prune_drops_lowest_importance() -> None:
    mem = EpisodicMemory(dict(MEM_CFG, capacity=10, importance_tau=100.0), core_dim=8, seed=0)
    for i in range(10):
        # Older entries with LOW surprise; newer with HIGH surprise.
        mem._insert(np.ones(8, dtype=np.float32), (0, 0), tick=i * 10,
                    surprise=0.1 if i < 5 else 5.0, summary=None)
    assert mem.pressure() == 1.0
    dropped = mem.prune(now_tick=100, keep_fraction=0.5)
    assert dropped == 5
    # Survivors are the high-surprise (recent) half.
    assert np.all(mem.surprises[: mem.size] == 5.0)


def test_insert_when_full_replaces_least_important() -> None:
    mem = EpisodicMemory(dict(MEM_CFG, capacity=3, importance_tau=1e9), core_dim=8, seed=0)
    for s in (1.0, 5.0, 3.0):
        mem._insert(np.ones(8, dtype=np.float32), (0, 0), 0, s, None)
    mem._insert(np.ones(8, dtype=np.float32), (0, 0), 0, 4.0, None)  # full: replaces s=1.0
    assert sorted(mem.surprises[:3].tolist()) == [3.0, 4.0, 5.0]


def test_state_roundtrip() -> None:
    mem = EpisodicMemory(MEM_CFG, core_dim=8, seed=0)
    rng = np.random.default_rng(0)
    for i in range(7):
        mem.maybe_write(rng.standard_normal(8), (i, i), i, float(10 + i),
                        summary={"food": np.ones((3, 3), dtype=bool),
                                 "water": np.zeros((3, 3), dtype=bool)})
    restored = EpisodicMemory(MEM_CFG, core_dim=8, seed=99)
    restored.load_state_dict(mem.state_dict())
    q = rng.standard_normal(8)
    s1, t1 = mem.retrieve(q, (3, 3))
    s2, t2 = restored.retrieve(q, (3, 3))
    assert np.allclose(s1, s2) and np.array_equal(t1, t2)
    assert restored.gate.threshold == mem.gate.threshold


# --------------------------------------------------------- trainer integration


def test_trainer_memory_integration_smoke() -> None:
    """Two updates with memory enabled: finite losses, per-env stores isolated,
    memory_pressure surfaced in the intero tensor, boundary clears happen."""
    cfg = memory_trainer_cfg()
    t = PPOTrainer(cfg)
    assert t.model.memory_dim == 8
    t.train(max_updates=2)
    assert all(np.isfinite(v) for v in t.last_metrics.values()), t.last_metrics
    assert "memory/pressure" in t.last_metrics

    # Force fill env 0 only; env 1 must remain untouched (per-env isolation).
    t.memories[0]._insert(np.ones(8, dtype=np.float32), (1, 1), 0, 1.0, None)
    assert t.memories[0].size > 0
    n_before_env1 = t.memories[1].size
    _, intero = t._obs_tensors()
    assert intero[0, 2].item() == pytest.approx(t.memories[0].pressure())
    assert intero[1, 2].item() == pytest.approx(t.memories[1].pressure())
    assert t.memories[1].size == n_before_env1

    # Episode boundary clears that env's memory.
    cfg2 = memory_trainer_cfg()
    cfg2["ppo"]["episode_length"] = 8  # done occurs inside one rollout
    t2 = PPOTrainer(cfg2)
    t2.memories[0]._insert(np.ones(8, dtype=np.float32), (1, 1), 0, 99.0, None)
    t2.collect_rollout()
    assert all(s != 99.0 for s in t2.memories[0].surprises[: t2.memories[0].size]), (
        "memory was not cleared at the episode boundary"
    )
