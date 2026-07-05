"""Checkpoint save/load: model, optimizer, step, RNG states, config hash.

Fully implemented (not a stub). A checkpoint is a single ``.pt`` file written
atomically (tmp file + rename). ``load_checkpoint`` verifies the config hash and
restores every global RNG stream so a resumed run continues bit-identically with
respect to those streams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world.config import config_hash

FORMAT_VERSION = 1


class ConfigMismatchError(RuntimeError):
    """Raised when a checkpoint's config hash differs from the active config."""


@dataclass
class Checkpoint:
    """In-memory view of a loaded checkpoint."""

    step: int
    config_hash: str
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    rng_states: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


def _collect_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically; returns the final path.

    ``extra`` is for caller-owned state (e.g. a world snapshot blob, replay
    cursors); it is stored verbatim.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": _collect_rng_states(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    cfg: dict[str, Any] | None = None,
    restore_rng: bool = True,
    allow_config_mismatch: bool = False,
) -> Checkpoint:
    """Load a checkpoint; restore into ``model``/``optimizer`` when provided.

    If ``cfg`` is given, its hash is checked against the stored hash and a
    :class:`ConfigMismatchError` is raised on mismatch unless
    ``allow_config_mismatch`` is True.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint format {payload.get('format_version')!r} "
            f"(expected {FORMAT_VERSION})"
        )
    if cfg is not None:
        active_hash = config_hash(cfg)
        if active_hash != payload["config_hash"] and not allow_config_mismatch:
            raise ConfigMismatchError(
                f"Checkpoint config hash {payload['config_hash'][:12]}... does not "
                f"match active config {active_hash[:12]}...; pass "
                f"--allow-config-mismatch to override."
            )
    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        _restore_rng_states(payload["rng_states"])
    return Checkpoint(
        step=payload["step"],
        config_hash=payload["config_hash"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        rng_states=payload["rng_states"],
        extra=payload["extra"],
    )


def prune_checkpoints(directory: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the ``keep_last`` newest ``*.pt`` files; returns deletions."""
    directory = Path(directory)
    ckpts = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    doomed = ckpts[:-keep_last] if keep_last > 0 else ckpts
    for p in doomed:
        p.unlink()
    return doomed
