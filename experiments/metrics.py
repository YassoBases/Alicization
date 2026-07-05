"""Generic time-series metrics for shift-injection experiments.

All functions operate on plain numpy arrays already collected by a runner
(reward-per-rollout, body_nll-per-rollout, per-rollout action histograms,
etc.) — nothing here touches ground truth or agent-internal state directly;
they are pure post-hoc analysis of recorded series.
"""

from __future__ import annotations

import numpy as np


def rolling_zscore_detection_tick(
    series: np.ndarray, baseline: np.ndarray, z_thresh: float = 3.0
) -> int | None:
    """First index in ``series`` where the value exceeds
    ``baseline.mean() + z_thresh * baseline.std()``. None if never.

    Used for detection latency: ``series`` is post-injection body_nll (or any
    other Ledger surprise signal), ``baseline`` is the pre-injection window of
    the same signal.
    """
    std = baseline.std()
    thresh = baseline.mean() + z_thresh * std if std > 0 else baseline.mean()
    exceeds = np.where(series > thresh)[0]
    return int(exceeds[0]) if exceeds.size else None


def broken_action_count(
    pre_action: np.ndarray,
    pre_success: np.ndarray,
    post_action: np.ndarray,
    post_success: np.ndarray,
    num_actions: int,
    drop_thresh: float = 0.2,
) -> int:
    """Count actions whose empirical success rate drops by more than
    ``drop_thresh`` from the pre-window to the post-window. Purely from
    observed (action, success) pairs — no lever/ground-truth access."""
    broken = 0
    for a in range(num_actions):
        pre_mask = pre_action == a
        post_mask = post_action == a
        if pre_mask.sum() < 5 or post_mask.sum() < 5:
            continue  # too few samples to call it either way
        pre_rate = pre_success[pre_mask].mean()
        post_rate = post_success[post_mask].mean()
        if pre_rate - post_rate > drop_thresh:
            broken += 1
    return broken


def readaptation_half_life(
    pre_reward: np.ndarray, post_reward: np.ndarray, window: int = 5
) -> int | None:
    """Ticks (in post-window rollout units) until reward recovers halfway
    from its post-injection minimum back to the pre-injection mean.

    Uses a rolling mean of width ``window`` to smooth noise before finding the
    minimum and the recovery crossing. None if the dip never recovers halfway
    within the recorded post-window.
    """
    if len(post_reward) < window:
        return None
    smoothed = np.convolve(post_reward, np.ones(window) / window, mode="valid")
    r_pre = pre_reward.mean()
    dip_idx = int(np.argmin(smoothed))
    r_min = smoothed[dip_idx]
    if r_pre <= r_min:
        return 0  # no dip at all relative to pre-shift mean
    half_target = r_min + 0.5 * (r_pre - r_min)
    after_dip = smoothed[dip_idx:]
    recovered = np.where(after_dip >= half_target)[0]
    return int(recovered[0]) if recovered.size else None


def performance_recovery_ratio(
    pre_reward: np.ndarray, post_reward: np.ndarray, tail_window: int = 10
) -> float:
    """(R_final - R_min) / (R_pre - R_min): 1.0 = fully recovered to the
    pre-shift baseline by the end of the recorded post-window, 0.0 = stuck at
    the worst post-shift dip, negative = ended up worse than the dip."""
    r_pre = pre_reward.mean()
    r_min = post_reward.min()
    r_final = post_reward[-tail_window:].mean()
    denom = r_pre - r_min
    if abs(denom) < 1e-9:
        return 1.0  # no dip to recover from
    return float((r_final - r_min) / denom)


def action_distribution(action_counts: np.ndarray) -> np.ndarray:
    """(N,) raw counts -> normalized probability distribution."""
    total = action_counts.sum()
    return action_counts / total if total > 0 else np.zeros_like(action_counts, dtype=float)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence (base-2, in [0, 1]) between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mean_and_ci95(values: list[float]) -> tuple[float, float]:
    """(mean, half-width of a normal-approx 95% CI) across seeds. NaNs
    (e.g. a run where detection never happened) are dropped before
    averaging; if that empties the list, returns (nan, nan)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, 1.96 * sem
