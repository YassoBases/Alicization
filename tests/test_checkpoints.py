"""Round-trip tests for training.checkpoints (fully implemented in the scaffold)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from training.checkpoints import (
    ConfigMismatchError,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)

CFG = {"seed": 0, "ppo": {"lr": 3e-4}}


def _make_model_and_opt() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # One step so the optimizer has real state to round-trip.
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    opt.step()
    return model, opt


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model, opt = _make_model_and_opt()
    path = save_checkpoint(tmp_path / "ck.pt", model, opt, step=1234, cfg=CFG,
                           extra={"note": "hi"})

    fresh_model = torch.nn.Linear(4, 3)
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    ckpt = load_checkpoint(path, fresh_model, fresh_opt, cfg=CFG)

    assert ckpt.step == 1234
    assert ckpt.extra == {"note": "hi"}
    for p_old, p_new in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_old, p_new)
    assert fresh_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_rng_states_roundtrip(tmp_path: Path) -> None:
    model, opt = _make_model_and_opt()
    save_checkpoint(tmp_path / "ck.pt", model, opt, step=0, cfg=CFG)
    expected = (random.random(), np.random.random(), torch.rand(1).item())

    # Perturb every stream, then restore from the checkpoint.
    random.random(), np.random.random(), torch.rand(1)
    load_checkpoint(tmp_path / "ck.pt", cfg=CFG)
    got = (random.random(), np.random.random(), torch.rand(1).item())
    assert got == expected


def test_config_hash_mismatch(tmp_path: Path) -> None:
    model, opt = _make_model_and_opt()
    save_checkpoint(tmp_path / "ck.pt", model, opt, step=0, cfg=CFG)
    with pytest.raises(ConfigMismatchError):
        load_checkpoint(tmp_path / "ck.pt", cfg={"seed": 1})
    ckpt = load_checkpoint(tmp_path / "ck.pt", cfg={"seed": 1},
                           allow_config_mismatch=True)
    assert ckpt.step == 0


def test_prune_keeps_newest(tmp_path: Path) -> None:
    model, opt = _make_model_and_opt()
    import os
    import time

    for i in range(4):
        p = save_checkpoint(tmp_path / f"ck{i}.pt", model, opt, step=i, cfg=CFG)
        past = time.time() - (4 - i) * 10
        os.utime(p, (past, past))
    deleted = prune_checkpoints(tmp_path, keep_last=2)
    assert sorted(p.name for p in deleted) == ["ck0.pt", "ck1.pt"]
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["ck2.pt", "ck3.pt"]
