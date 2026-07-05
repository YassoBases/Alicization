"""Static plots (stub): publication-style figures from run logs."""

from __future__ import annotations

from pathlib import Path


def plot_calibration(run_dir: str | Path, out_path: str | Path) -> None:
    """Ledger forecast calibration vs identity baseline for one run."""
    raise NotImplementedError


def plot_training_curves(run_dirs: list[str | Path], out_path: str | Path, keys: list[str]) -> None:
    """Overlayed scalar curves (from TensorBoard logs) across runs."""
    raise NotImplementedError
