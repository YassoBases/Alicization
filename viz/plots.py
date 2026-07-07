"""Static battery/report figures. Matplotlib, 150 dpi PNGs into the run dir.

Every forecasting plot draws the identity-baseline line (CLAUDE.md: every
metric that evaluates a forecast reports the identity predictor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

DPI = 150


def _finish(fig: matplotlib.figure.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def reward_curve(
    rewards: Sequence[float] | np.ndarray,
    out_path: str | Path,
    title: str = "Reward per rollout",
    smooth: int = 10,
) -> Path:
    """Raw reward series plus a rolling mean."""
    rewards = np.asarray(rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.35, label="raw")
    if len(rewards) >= smooth:
        kernel = np.ones(smooth) / smooth
        ax.plot(np.arange(smooth - 1, len(rewards)),
                np.convolve(rewards, kernel, mode="valid"),
                label=f"rolling mean ({smooth})")
    ax.set_xlabel("rollout")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, out_path)


def metric_around_event(
    values: Sequence[float] | np.ndarray,
    event_index: int,
    out_path: str | Path,
    window: int | None = None,
    metric_name: str = "metric",
    event_name: str = "lever event",
    title: str | None = None,
) -> Path:
    """Metric series with a vertical line at the lever tick, ± window.

    ``event_index`` is the index in ``values`` where the event happened;
    ``window`` trims the plot to event ± window samples (full series if None).
    """
    values = np.asarray(values, dtype=float)
    lo, hi = 0, len(values)
    if window is not None:
        lo, hi = max(0, event_index - window), min(len(values), event_index + window)
    xs = np.arange(lo, hi) - event_index
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, values[lo:hi])
    ax.axvline(0, color="k", ls="--", lw=1, label=event_name)
    ax.set_xlabel(f"samples relative to {event_name}")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} around {event_name}")
    ax.legend()
    return _finish(fig, out_path)


def calibration_diagram(
    bin_confidence: Sequence[float],
    bin_accuracy: Sequence[float],
    out_path: str | Path,
    bin_counts: Sequence[int] | None = None,
    ece: float | None = None,
    title: str = "Reliability calibration",
) -> Path:
    """Confidence-vs-accuracy diagram with the perfect-calibration diagonal."""
    conf = np.asarray(bin_confidence, dtype=float)
    acc = np.asarray(bin_accuracy, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.bar(conf, acc, width=0.08, alpha=0.8, label="observed")
    if bin_counts is not None:
        for c, a, n in zip(conf, acc, bin_counts):
            ax.annotate(str(n), (c, a), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title + (f" (ECE={ece:.3f})" if ece is not None else ""))
    ax.legend()
    return _finish(fig, out_path)


def nmse_bars_per_horizon(
    horizons: Sequence[int],
    nmse_values: dict[int, Sequence[float]],
    out_path: str | Path,
    title: str = "Forecaster NMSE vs identity",
) -> Path:
    """Per-horizon NMSE bars with 95% CI whiskers and the MANDATORY identity
    baseline line at NMSE = 1.0."""
    means, errs = [], []
    for k in horizons:
        vals = np.asarray(list(nmse_values[k]), dtype=float)
        means.append(vals.mean())
        errs.append(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in horizons], means, yerr=errs, capsize=4)
    ax.axhline(1.0, color="k", ls="--", lw=1.2,
               label="identity baseline (NMSE = 1)")
    ax.set_xlabel("horizon k (ticks)")
    ax.set_ylabel("NMSE")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, out_path)


def ablation_boxplots(
    groups: dict[str, Sequence[float]],
    out_path: str | Path,
    metric_name: str = "metric",
    title: str = "Ablation comparison",
) -> Path:
    """Side-by-side boxplots (one box per condition) with seed scatter."""
    labels = list(groups)
    data = [np.asarray(list(groups[k]), dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(1.6 * max(4, len(labels)), 4))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.7, zorder=3)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    return _finish(fig, out_path)


def divergence_trace(
    traces: dict[str, Sequence[float]],
    out_path: str | Path,
    spike_level: float | None = None,
    event_index: int = 0,
    title: str = "Mirror divergence",
) -> Path:
    """Divergence traces (e.g. mirror vs ablation) around an event tick."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, ys in traces.items():
        ax.plot(np.asarray(ys, dtype=float), label=label)
    if spike_level is not None:
        ax.axhline(spike_level, color="k", ls=":", lw=1, label=f"spike level {spike_level:.1f}")
    if event_index:
        ax.axvline(event_index, color="k", ls="--", lw=1)
    ax.set_xlabel("tick")
    ax.set_ylabel("divergence (cells)")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, out_path)
