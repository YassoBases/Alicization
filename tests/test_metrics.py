"""Every per-tick spec metric in experiments/metrics.py exercised on
hand-built synthetic traces with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import metrics as M

T_E = 3000  # event tick used throughout


# ------------------------------------------------------------- rolling mean


def test_rolling_mean_alignment_and_warmup() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = M.rolling_mean(x, w=3)
    # Warmup: partial means; steady state: mean of the last 3 inclusive.
    assert out[0] == 1.0
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)
    assert len(out) == len(x)


# -------------------------------------------------------- detection latency


def _nll_trace(jump_at: int | None, n: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    nll = rng.normal(1.0, 0.05, size=n)
    if jump_at is not None:
        nll[jump_at:] += 2.0  # 40 sigma jump: unambiguous
    return nll

def test_detection_latency_finds_jump_promptly() -> None:
    latency = M.detection_latency(_nll_trace(T_E + 100), t_e=T_E)
    # Jump at +100; the w=50 rolling mean crosses mu+4sd within a few ticks
    # of the jump entering the window, m=10 consecutive adds 0 (stays above).
    assert 100 <= latency <= 160, latency


def test_detection_latency_censors_when_no_change() -> None:
    assert M.detection_latency(_nll_trace(None), t_e=T_E) == float("inf")


def test_detection_latency_requires_consecutive_ticks() -> None:
    """A single-tick spike must NOT trigger detection (m=10 consecutive)."""
    nll = _nll_trace(None)
    nll[T_E + 50] = 100.0  # one-tick spike: rolling mean w=50 lifts ~2sd...
    # make the spike big enough to cross 4sd for < m ticks only:
    latency = M.detection_latency(nll, t_e=T_E, m=60)  # spike lasts 50 in-window ticks
    assert latency == float("inf")


# ------------------------------------------------------------ broken action


def test_broken_action_failures_counts_only_window_and_action() -> None:
    n = 8000
    actions = np.zeros(n, dtype=int)
    successes = np.ones(n, dtype=bool)
    actions[T_E + 10] = 2; successes[T_E + 10] = False   # counted
    actions[T_E + 20] = 2; successes[T_E + 20] = True    # success: not counted
    actions[T_E + 30] = 3; successes[T_E + 30] = False   # other action: no
    actions[T_E - 5] = 2;  successes[T_E - 5] = False    # pre-event: no
    actions[T_E + 2500] = 2; successes[T_E + 2500] = False  # past window: no
    assert M.broken_action_failures(actions, successes, 2, T_E, window=2000) == 1


def test_emission_rate_curve() -> None:
    actions = np.array([2] * 100 + [0] * 100)
    rate = M.emission_rate(actions, 2, w=200)
    assert rate[99] == pytest.approx(1.0)
    assert rate[-1] == pytest.approx(0.5)  # 100 of last 200


# ------------------------------------------------------------ re-adaptation


def _reward_trace(recover_at: int | None, n: int = 20_000) -> np.ndarray:
    r = np.full(n, 1.0)
    if recover_at is None:
        r[T_E:] = 0.0
    else:
        r[T_E:recover_at] = 0.0
    return r


def test_readaptation_half_life_exact_recovery_point() -> None:
    tau_rel = M.readaptation_half_life_ticks(_reward_trace(T_E + 4000), t_e=T_E)
    # Recovery at +4000; rolling w=500 needs ~450 more ticks to reach 0.9.
    assert 4000 <= tau_rel <= 4500, tau_rel


def test_readaptation_half_life_censored() -> None:
    assert M.readaptation_half_life_ticks(_reward_trace(None), t_e=T_E) == float("inf")
    # And the reset battery alias is literally the same construction.
    assert M.recovery_half_life_ticks is M.readaptation_half_life_ticks


def test_recovery_ratio_full_and_censored() -> None:
    reward = _reward_trace(T_E + 4000)
    tau = T_E + M.readaptation_half_life_ticks(reward, t_e=T_E)
    ratio = M.recovery_ratio(reward, T_E, tau)
    assert 0.95 <= ratio <= 1.01, ratio
    assert np.isnan(M.recovery_ratio(reward, T_E, float("inf")))


def test_reward_before_after_windows() -> None:
    reward = np.concatenate([np.full(5000, 2.0), np.full(5000, 0.5), np.full(5000, 1.5)])
    out = M.reward_before_after(reward, t_e=5000, tau=10_000, window=5000)
    assert out["before"] == pytest.approx(2.0)
    assert out["after"] == pytest.approx(0.5)
    assert out["recovered"] == pytest.approx(1.5)


# ------------------------------------------------------- body-model quality


def test_brier_score_known_values() -> None:
    assert M.brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert M.brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_roc_auc_perfect_random_and_degenerate() -> None:
    assert M.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert M.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert M.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) == pytest.approx(0.5)  # ties
    assert np.isnan(M.roc_auc(np.array([0.5, 0.6]), np.array([1, 1])))  # one class


def test_categorical_accuracy_nll() -> None:
    probs = np.array([[0.7, 0.1, 0.1, 0.05, 0.05],
                      [0.1, 0.6, 0.1, 0.1, 0.1]])
    acc, nll = M.categorical_accuracy_nll(probs, np.array([0, 1]))
    assert acc == 1.0
    assert nll == pytest.approx(-(np.log(0.7) + np.log(0.6)) / 2)
    acc_wrong, _ = M.categorical_accuracy_nll(probs, np.array([1, 0]))
    assert acc_wrong == 0.0


def test_energy_mae() -> None:
    assert M.energy_mae(np.array([0.1, -0.2]), np.array([0.0, 0.0])) == pytest.approx(0.15)


# ------------------------------------------------ action-distribution shift


def test_action_js_shift_zero_and_disjoint() -> None:
    same = np.tile(np.arange(9), 1000)
    # 2000-tick windows aren't a multiple of 9, so the tiling phase leaves a
    # ~1e-6 residual; anything at that scale is "no shift".
    assert M.action_js_shift(same, t_e=len(same) // 2) == pytest.approx(0.0, abs=1e-4)
    actions = np.concatenate([np.zeros(2000, int), np.full(2000, 5)])
    assert M.action_js_shift(actions, t_e=2000) == pytest.approx(1.0)  # disjoint: 1 bit


def test_matched_context_js_controls_position_confound() -> None:
    """Policy is a fixed function of position; only the POSITION distribution
    shifts at t_e. Naive JS reports a big shift; the matched version ~0."""
    rng = np.random.default_rng(0)
    n = 4000
    t_e = 2000
    # Pre: mostly region A (x<16); post: mostly region B (x>=16).
    x = np.concatenate([
        rng.choice([4, 20], size=t_e, p=[0.9, 0.1]),
        rng.choice([4, 20], size=t_e, p=[0.1, 0.9]),
    ])
    positions = np.stack([x, np.full(n, 4)], axis=1)
    actions = np.where(x < 16, 0, 5)  # deterministic policy per region
    naive = M.action_js_shift(actions, t_e)
    matched = M.matched_context_js(actions, positions, t_e, bin_size=16)
    assert naive > 0.5
    assert matched == pytest.approx(0.0, abs=1e-9)


def test_anticipation_divergence_zero_for_identical() -> None:
    a = np.tile(np.arange(9), 100)
    assert M.anticipation_divergence(a, a.copy()) == pytest.approx(0.0, abs=1e-12)
    assert M.anticipation_divergence(np.zeros(50, int), np.full(50, 3)) == pytest.approx(1.0)


# ----------------------------------------------------------- memory metrics


def test_ece_10bin_perfect_and_known_miscalibration() -> None:
    rng = np.random.default_rng(1)
    preds = np.full(10_000, 0.75)
    realized = (rng.random(10_000) < 0.75).astype(float)
    ece, rows = M.ece_10bin(preds, realized)
    assert ece < 0.02
    assert sum(r["count"] for r in rows) == 10_000
    # Overconfident by exactly 0.5: predicts 0.9, realizes 0.4.
    preds = np.full(1000, 0.9)
    realized = np.concatenate([np.ones(400), np.zeros(600)])
    ece, _ = M.ece_10bin(preds, realized)
    assert ece == pytest.approx(0.5)


def test_stale_trip_rate() -> None:
    assert M.stale_trip_rate(5, 1000) == 5.0
    assert M.stale_trip_rate(0, 0) == 0.0


# ------------------------------------------------------------- forecasting


def test_nmse_identity_baseline_semantics() -> None:
    current = np.zeros((100, 3))
    actual = np.ones((100, 3))
    perfect = actual.copy()
    assert M.nmse(perfect, actual, current) == 0.0
    assert M.nmse(current, actual, current) == pytest.approx(1.0)  # = identity
    worse = 3 * np.ones((100, 3))
    assert M.nmse(worse, actual, current) == pytest.approx(4.0)
    assert M.nmse(perfect, actual, actual) == float("inf")  # identity exact


# -------------------------------------------------------------- attribution


def test_attribution_metrics_confusion_and_splits() -> None:
    truth = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 0])
    out = M.attribution_metrics(pred, truth)
    assert out["accuracy"] == pytest.approx(4 / 6)
    assert out["accuracy_self"] == pytest.approx(2 / 3)
    assert out["accuracy_world"] == pytest.approx(1.0)
    conf = out["confusion"]
    assert conf[0, 0] == 2 and conf[0, 1] == 1 and conf[2, 0] == 1
    assert conf.sum() == 6


# ------------------------------------------------ proposal-quality metrics


def test_realized_benefit_ab_in_control_std_units() -> None:
    control = np.array([1.0, 1.2, 0.8, 1.0])  # mean 1.0, std ~0.141
    treated = control + 0.5
    benefit = M.realized_benefit_ab(treated, control)
    assert benefit == pytest.approx(0.5 / control.std(), rel=1e-6)
    assert M.realized_benefit_ab(control, control) == pytest.approx(0.0)


def test_realized_benefit_ab_degenerate_control_is_nan() -> None:
    # A 1-point or constant control series cannot normalize a difference —
    # dividing by epsilon manufactured a ~5e11-sd "benefit" in the first
    # battery run. Must be NaN, never an astronomical number.
    good = np.array([1.0, 1.2, 0.8, 1.0])
    assert math.isnan(M.realized_benefit_ab(np.array([2.0]), np.array([1.0])))
    assert math.isnan(M.realized_benefit_ab(good, np.array([1.0, 1.0, 1.0])))
    assert math.isnan(M.realized_benefit_ab(np.array([2.0]), good))


def test_realized_benefit_pre_post_drift_corrected() -> None:
    # Pre-trend rises 0.1/step; post continues the SAME trend -> benefit ~ 0
    # (a naive pre/post mean diff would falsely report ~+1.0).
    pre = 0.1 * np.arange(10)
    post = 0.1 * np.arange(10, 20)
    assert abs(M.realized_benefit_pre_post(pre, post)) < 1e-6
    # A true level jump above the trend IS credited.
    assert M.realized_benefit_pre_post(pre, post + 1.0) > 1.0


def test_success_criteria_hit_direction_and_window() -> None:
    series = np.array([0.0, 0.0, 5.0, 0.0])
    assert M.success_criteria_hit(series, threshold=4.0, direction="up", window=4)
    assert not M.success_criteria_hit(series, 4.0, "up", window=2)  # outside window
    assert M.success_criteria_hit(-series, -4.0, "down", window=4)
    assert M.hit_rate([True, False, True, False]) == pytest.approx(0.5)


def test_acceptance_rate_is_over_decided_only() -> None:
    assert M.acceptance_rate(
        ["approved", "rejected", "pending", "partially_approved", "postponed"]
    ) == pytest.approx(2 / 3)
    assert np.isnan(M.acceptance_rate(["pending"]))


def test_usefulness_divergence_cells() -> None:
    stats = M.usefulness_stats(
        ratings=[5, 4, 2, None], benefits=[-0.1, 0.5, 0.7, 1.0])
    assert stats["mean_rating"] == pytest.approx((5 + 4 + 2) / 3)
    assert stats["rated_useful_no_benefit"] == 1      # the 5-rated, -0.1
    assert stats["unrated_or_low_but_beneficial"] == 1  # the 2-rated, +0.7


def test_repeated_after_denial_rate_with_rewording() -> None:
    events = [
        {"generator": "g", "type": "hp", "target": "lr", "tick": 100,
         "kind": "rejected", "rationale_hash": "aaa"},
        {"generator": "g", "type": "hp", "target": "lr", "tick": 200,
         "kind": "proposed", "rationale_hash": "bbb"},  # reworded repeat
        {"generator": "g", "type": "hp", "target": "kl", "tick": 300,
         "kind": "rejected", "rationale_hash": "ccc"},  # never re-proposed
        {"generator": "quiet", "type": "mem", "target": "x", "tick": 10,
         "kind": "proposed", "rationale_hash": "ddd"},  # no denials at all
    ]
    out = M.repeated_after_denial_rate(events, k_ticks=1000)
    assert out["g"]["denials"] == 2
    assert out["g"]["repeat_rate"] == pytest.approx(0.5)
    assert out["g"]["reworded_rate"] == pytest.approx(1.0)
    assert np.isnan(out["quiet"]["repeat_rate"])


def test_time_to_first_useful_ticks() -> None:
    assert M.time_to_first_useful_ticks([100, 50, 200], [None, -0.2, 0.4]) == 200
    assert M.time_to_first_useful_ticks([100], [None]) == float("inf")
