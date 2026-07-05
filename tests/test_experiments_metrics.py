"""Unit tests for experiments/metrics.py's pure time-series analysis functions."""

from __future__ import annotations

import numpy as np

from experiments.metrics import (
    action_distribution,
    broken_action_count,
    jensen_shannon_divergence,
    mean_and_ci95,
    performance_recovery_ratio,
    readaptation_half_life,
    rolling_zscore_detection_tick,
)


def test_rolling_zscore_detection_tick_finds_spike() -> None:
    baseline = np.full(50, 0.1)
    series = np.concatenate([np.full(5, 0.1), np.full(5, 5.0)])
    idx = rolling_zscore_detection_tick(series, baseline, z_thresh=3.0)
    assert idx == 5


def test_rolling_zscore_detection_tick_none_when_no_spike() -> None:
    baseline = np.array([0.1, 0.12, 0.09, 0.11, 0.1])
    series = np.array([0.1, 0.11, 0.1])
    assert rolling_zscore_detection_tick(series, baseline) is None


def test_broken_action_count_detects_dropped_success_rate() -> None:
    # action 0: success drops from 1.0 to 0.0; action 1: unchanged at 1.0
    pre_action = np.array([0] * 10 + [1] * 10)
    pre_success = np.array([1] * 10 + [1] * 10)
    post_action = np.array([0] * 10 + [1] * 10)
    post_success = np.array([0] * 10 + [1] * 10)
    count = broken_action_count(pre_action, pre_success, post_action, post_success, num_actions=2)
    assert count == 1


def test_broken_action_count_ignores_sparse_actions() -> None:
    pre_action = np.array([0, 0, 1, 1])  # only 2 samples each: below the min-sample floor
    pre_success = np.array([1, 1, 1, 1])
    post_action = np.array([0, 0, 1, 1])
    post_success = np.array([0, 0, 1, 1])
    count = broken_action_count(pre_action, pre_success, post_action, post_success, num_actions=2)
    assert count == 0


def test_readaptation_half_life_recovers() -> None:
    pre_reward = np.full(10, 1.0)
    # dips to 0 then linearly recovers back to 1.0 over 10 steps
    post_reward = np.linspace(0.0, 1.0, 11)
    half_life = readaptation_half_life(pre_reward, post_reward, window=1)
    assert half_life is not None
    assert 4 <= half_life <= 6  # halfway point of a linear ramp


def test_readaptation_half_life_never_recovers() -> None:
    pre_reward = np.full(10, 1.0)
    post_reward = np.full(20, 0.0)  # stays at the dip forever
    assert readaptation_half_life(pre_reward, post_reward, window=1) is None


def test_performance_recovery_ratio_full_and_none() -> None:
    pre_reward = np.full(10, 1.0)
    post_full = np.concatenate([np.zeros(5), np.ones(10)])
    assert performance_recovery_ratio(pre_reward, post_full, tail_window=10) == 1.0

    post_none = np.zeros(15)
    assert performance_recovery_ratio(pre_reward, post_none, tail_window=10) == 0.0


def test_action_distribution_normalizes() -> None:
    counts = np.array([1.0, 3.0, 0.0, 0.0])
    dist = action_distribution(counts)
    assert np.isclose(dist.sum(), 1.0)
    assert np.allclose(dist, [0.25, 0.75, 0.0, 0.0])


def test_jensen_shannon_divergence_identical_is_zero() -> None:
    p = np.array([0.5, 0.5])
    assert jensen_shannon_divergence(p, p) == 0.0


def test_jensen_shannon_divergence_disjoint_is_one() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p, q) == 1.0


def test_mean_and_ci95_basic() -> None:
    mean, ci = mean_and_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert ci > 0


def test_mean_and_ci95_drops_nones_and_nans() -> None:
    mean, ci = mean_and_ci95([1.0, None, float("nan"), 3.0])
    assert mean == 2.0


def test_mean_and_ci95_empty_returns_nan() -> None:
    mean, ci = mean_and_ci95([])
    assert np.isnan(mean) and np.isnan(ci)
