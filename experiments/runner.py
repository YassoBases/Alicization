"""Experiment runner (stub): sweeps lever conditions and collects battery metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_experiment(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Run one configured experiment condition end to end; returns summary metrics."""
    raise NotImplementedError


def run_battery(battery_name: str, checkpoint: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a trained checkpoint against a named battery in experiments/batteries."""
    raise NotImplementedError
