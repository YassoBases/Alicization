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


# ===========================================================================
# Per-tick spec metrics. Notation: t_e = event (capability-shift) tick.
# These operate on full per-tick traces (not per-rollout aggregates); the
# rollout-level helpers above remain for the stage-3c battery.
# ===========================================================================


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean, same length as ``x``: out[t] = mean(x[max(0,t-w+1) : t+1]).

    The warmup region (t < w-1) averages what exists so far, so the output is
    defined from tick 0 — event-relative indexing stays aligned to the input.
    """
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(len(x))
    lo = np.maximum(t - w + 1, 0)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def detection_latency(
    body_nll: np.ndarray,
    t_e: int,
    w: int = 50,
    pre_window: int = 2000,
    z: float = 4.0,
    m: int = 10,
    censor: int = 50_000,
) -> float:
    """Ticks from t_e until the smoothed body-NLL exceeds mu + z*sd (pre-event
    stats) for ``m`` CONSECUTIVE ticks. inf if censored (never within
    ``censor`` ticks). The pre-event window must be clean of the event.

    e = rolling_mean(body_nll, w); mu, sd from e[t_e - pre_window : t_e].
    """
    e = rolling_mean(body_nll, w)
    pre = e[max(0, t_e - pre_window): t_e]
    if pre.size == 0:
        return float("inf")
    mu, sd = float(pre.mean()), float(pre.std())
    thresh = mu + z * sd
    post = e[t_e: t_e + censor]
    above = post > thresh
    if above.size < m:
        return float("inf")
    # First index where m consecutive Trues start.
    run = np.convolve(above.astype(int), np.ones(m, dtype=int), mode="valid")
    hits = np.nonzero(run == m)[0]
    return float(hits[0]) if hits.size else float("inf")


def broken_action_failures(
    actions: np.ndarray,
    successes: np.ndarray,
    shifted_action: int,
    t_e: int,
    window: int = 2000,
) -> int:
    """Number of executions of ``shifted_action`` that FAIL in [t_e, t_e+window)."""
    actions = np.asarray(actions)
    successes = np.asarray(successes, dtype=bool)
    sl = slice(t_e, t_e + window)
    mask = (actions[sl] == shifted_action) & (~successes[sl])
    return int(mask.sum())


def emission_rate(actions: np.ndarray, action_id: int, w: int = 200) -> np.ndarray:
    """Rolling emission-rate curve of one action (for the writeup plot)."""
    return rolling_mean((np.asarray(actions) == action_id).astype(float), w)


def readaptation_half_life_ticks(
    reward: np.ndarray,
    t_e: int,
    w: int = 500,
    pre: int = 5000,
    frac: float = 0.9,
    sustain: int = 500,
) -> float:
    """Ticks from t_e until smoothed reward is back at ``frac`` of its
    pre-event mean, SUSTAINED for ``sustain`` ticks. inf if never.

    perf = rolling_mean(reward, w); perf_pre = mean(perf[t_e-pre : t_e]);
    tau = first t >= t_e with perf[t : t+sustain] all >= frac * perf_pre.
    (The rollout-level ``readaptation_half_life`` above is the stage-3c
    battery's coarser variant; this is the per-tick spec construction.)
    """
    perf = rolling_mean(reward, w)
    pre_arr = perf[max(0, t_e - pre): t_e]
    if pre_arr.size == 0:
        return float("inf")
    target = frac * float(pre_arr.mean())
    post = perf[t_e:]
    ok = post >= target
    if ok.size < sustain:
        return float("inf")
    run = np.convolve(ok.astype(int), np.ones(sustain, dtype=int), mode="valid")
    hits = np.nonzero(run == sustain)[0]
    return float(hits[0]) if hits.size else float("inf")


# The reset battery's recovery half-life is the same construction with the
# event being a restore.
recovery_half_life_ticks = readaptation_half_life_ticks


def recovery_ratio(
    reward: np.ndarray,
    t_e: int,
    tau: float,
    w: int = 500,
    pre: int = 5000,
    post: int = 5000,
) -> float:
    """mean(perf[tau : tau+post]) / perf_pre; ``tau`` is ABSOLUTE (t_e + half-life).

    nan when tau is censored (inf) or perf_pre ~ 0 (ratio undefined).
    """
    if not np.isfinite(tau):
        return float("nan")
    perf = rolling_mean(reward, w)
    perf_pre = float(perf[max(0, t_e - pre): t_e].mean())
    if abs(perf_pre) < 1e-12:
        return float("nan")
    tail = perf[int(tau): int(tau) + post]
    if tail.size == 0:
        return float("nan")
    return float(tail.mean()) / perf_pre


def reward_before_after(
    reward: np.ndarray, t_e: int, tau: float | None = None, window: int = 5000
) -> dict[str, float]:
    """Mean reward over [t_e-window, t_e), [t_e, t_e+window), and — when tau
    is finite — [tau, tau+window)."""
    reward = np.asarray(reward, dtype=float)
    out = {
        "before": float(reward[max(0, t_e - window): t_e].mean()),
        "after": float(reward[t_e: t_e + window].mean()),
        "recovered": float("nan"),
    }
    if tau is not None and np.isfinite(tau):
        tail = reward[int(tau): int(tau) + window]
        if tail.size:
            out["recovered"] = float(tail.mean())
    return out


# ------------------------------------------------- body-model quality (spec)


def brier_score(p_success: np.ndarray, y: np.ndarray) -> float:
    """mean((p - y)^2) over ticks."""
    p = np.asarray(p_success, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(((p - y) ** 2).mean())


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware. nan when only one class is
    present."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks_sorted = np.arange(1, len(p) + 1, dtype=float)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks_sorted[i: j + 1] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = ranks_sorted
    u = float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def categorical_accuracy_nll(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """(accuracy, mean NLL) for the 5-way dpos outcome: ``probs`` (N, K)
    predicted class probabilities, ``labels`` (N,) true class indices."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    acc = float((probs.argmax(axis=1) == labels).mean())
    picked = probs[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(picked, 1e-12, None)).mean())
    return acc, nll


def energy_mae(denergy_pred: np.ndarray, denergy_obs: np.ndarray) -> float:
    """MAE(denergy_pred, denergy_obs)."""
    return float(np.abs(np.asarray(denergy_pred) - np.asarray(denergy_obs)).mean())


# --------------------------------------------------- action-distribution shift


def action_histogram(actions: np.ndarray, num_actions: int) -> np.ndarray:
    return np.bincount(np.asarray(actions, dtype=int), minlength=num_actions).astype(float)


def action_js_shift(
    actions: np.ndarray, t_e: int, window: int = 2000, num_actions: int = 9
) -> float:
    """JS divergence between action histograms in the ``window`` ticks pre vs
    post t_e."""
    pre = action_histogram(actions[max(0, t_e - window): t_e], num_actions)
    post = action_histogram(actions[t_e: t_e + window], num_actions)
    return jensen_shannon_divergence(action_distribution(pre), action_distribution(post))


def matched_context_js(
    actions: np.ndarray,
    positions: np.ndarray,
    t_e: int,
    window: int = 2000,
    num_actions: int = 9,
    bin_size: int = 16,
    min_count: int = 20,
) -> float:
    """Position-matched JS shift (confound control): bin ticks by coarse
    position (``pos // bin_size``), compute the pre/post action-histogram JS
    per bin, and average over bins with >= ``min_count`` samples in BOTH
    windows. The naive shift conflates policy change with the position
    distribution itself shifting; this version compares like context with
    like. nan when no bin has support in both windows.
    """
    actions = np.asarray(actions, dtype=int)
    positions = np.asarray(positions, dtype=int)
    pre_sl = slice(max(0, t_e - window), t_e)
    post_sl = slice(t_e, t_e + window)
    bins_pre = positions[pre_sl] // bin_size
    bins_post = positions[post_sl] // bin_size
    key_pre = bins_pre[:, 0] * 10_000 + bins_pre[:, 1]
    key_post = bins_post[:, 0] * 10_000 + bins_post[:, 1]
    shared = set(np.unique(key_pre)) & set(np.unique(key_post))
    js_values = []
    for key in sorted(shared):
        a_pre = actions[pre_sl][key_pre == key]
        a_post = actions[post_sl][key_post == key]
        if len(a_pre) < min_count or len(a_post) < min_count:
            continue
        js_values.append(jensen_shannon_divergence(
            action_distribution(action_histogram(a_pre, num_actions)),
            action_distribution(action_histogram(a_post, num_actions)),
        ))
    return float(np.mean(js_values)) if js_values else float("nan")


def anticipation_divergence(
    actions_signaled: np.ndarray, actions_unsignaled: np.ndarray, num_actions: int = 9
) -> float:
    """Reset battery: JS(action dist | signaled) vs (| unsignaled) over the
    pre-reset window. Expected ~ 0; a non-null result is a stop-and-
    investigate flag, not a feature."""
    return jensen_shannon_divergence(
        action_distribution(action_histogram(actions_signaled, num_actions)),
        action_distribution(action_histogram(actions_unsignaled, num_actions)),
    )


# --------------------------------------------------------- stage-5 memory


def ece_10bin(
    predicted: np.ndarray, realized: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """10-bin expected calibration error of predicted reliability vs realized
    revisit match. Returns (ece, per-bin rows)."""
    predicted = np.asarray(predicted, dtype=float)
    realized = np.asarray(realized, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for b in range(bins):
        hi_ok = predicted < edges[b + 1] if b < bins - 1 else predicted <= 1.0
        mask = (predicted >= edges[b]) & hi_ok
        if not mask.any():
            continue
        conf, acc, frac = predicted[mask].mean(), realized[mask].mean(), mask.mean()
        ece += frac * abs(conf - acc)
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "confidence": float(conf), "accuracy": float(acc),
                     "count": int(mask.sum())})
    return float(ece), rows


def stale_trip_rate(stale_trips: int, ticks: int) -> float:
    """Trips terminating at depleted/moved targets, per 1k ticks."""
    return 1000.0 * stale_trips / max(1, ticks)


# ------------------------------------------------------- stage-4 forecasting


def nmse(forecast_k: np.ndarray, actual_future: np.ndarray, current: np.ndarray) -> float:
    """mse(forecast, actual_{t+k}) / mse(current, actual_{t+k}) — the identity
    baseline in the denominator is MANDATORY. >= 1.0 means the forecaster is
    not modeling dynamics. inf when the identity MSE is exactly 0."""
    forecast_k = np.asarray(forecast_k, dtype=float)
    actual_future = np.asarray(actual_future, dtype=float)
    current = np.asarray(current, dtype=float)
    mse_f = float(((forecast_k - actual_future) ** 2).mean())
    mse_i = float(((current - actual_future) ** 2).mean())
    return mse_f / mse_i if mse_i > 0 else float("inf")


# ----------------------------------------------------------- attribution


# ===========================================================================
# Proposal-quality metrics (Section 17). Realized benefit is the PRIMARY
# metric; acceptance rate is explicitly weak (it measures reviewer behavior
# as much as proposal quality) and must never headline alone.
# ===========================================================================


def realized_benefit_ab(
    m_treated: np.ndarray, m_control: np.ndarray
) -> float:
    """Preferred form: A/B against a seeded control run.

    benefit = (mean(M_treated[W]) - mean(M_control[W])) / std(M_control[W]).
    Positive = the change helped, in control-std units.

    Degenerate control (fewer than 2 points, or ~zero variance) makes the
    normalization meaningless — a 1-point series would report the raw
    difference over epsilon as an astronomical "benefit". NaN instead; the
    caller must collect a longer control series.
    """
    treated = np.asarray(m_treated, dtype=float)
    control = np.asarray(m_control, dtype=float)
    if len(control) < 2 or len(treated) < 2:
        return float("nan")
    sd = float(control.std())
    if sd < 1e-8:
        return float("nan")
    return float((treated.mean() - control.mean()) / sd)


def realized_benefit_pre_post(
    m_pre: np.ndarray, m_post: np.ndarray
) -> float:
    """Fallback when A/B is infeasible (e.g. logging changes): post-window
    mean minus the PRE-TREND EXTRAPOLATION (drift correction), normalized by
    the pre std. Records using this must be marked evaluation=pre_post.
    """
    pre = np.asarray(m_pre, dtype=float)
    post = np.asarray(m_post, dtype=float)
    if len(pre) < 3:
        return float("nan")
    xs = np.arange(len(pre), dtype=float)
    slope, intercept = np.polyfit(xs, pre, 1)
    xs_post = np.arange(len(pre), len(pre) + len(post), dtype=float)
    extrapolated = slope * xs_post + intercept
    return float((post.mean() - extrapolated.mean()) / (pre.std() + 1e-12))


def success_criteria_hit(
    series: np.ndarray, threshold: float, direction: str, window: int
) -> bool:
    """Did M cross the proposal's OWN threshold within its OWN window."""
    values = np.asarray(series, dtype=float)[:window]
    if values.size == 0:
        return False
    return bool((values >= threshold).any() if direction == "up"
                else (values <= threshold).any())


def hit_rate(hits: list[bool]) -> float:
    return float(np.mean([bool(h) for h in hits])) if hits else float("nan")


def acceptance_rate(decisions: list[str]) -> float:
    """approvals / decided. WEAK by construction — reviewer behavior is in
    the numerator and denominator; report only next to realized benefit."""
    decided = [d for d in decisions
               if d in ("approved", "partially_approved", "rejected")]
    if not decided:
        return float("nan")
    approved = sum(d != "rejected" for d in decided)
    return approved / len(decided)


def usefulness_stats(
    ratings: list[float | None], benefits: list[float | None]
) -> dict[str, float]:
    """Mean 1-5 reviewer rating + the divergence cells: rated-useful-but-no-
    benefit (and its mirror) are interesting, so they are counted explicitly."""
    rated = [(r, b) for r, b in zip(ratings, benefits) if r is not None]
    if not rated:
        return {"mean_rating": float("nan"), "rated_useful_no_benefit": 0,
                "unrated_or_low_but_beneficial": 0}
    mean_rating = float(np.mean([r for r, _ in rated]))
    useful_no_benefit = sum(
        1 for r, b in rated if r >= 4 and (b is None or b <= 0))
    low_but_beneficial = sum(
        1 for r, b in rated if r <= 2 and b is not None and b > 0)
    return {"mean_rating": mean_rating,
            "rated_useful_no_benefit": useful_no_benefit,
            "unrated_or_low_but_beneficial": low_but_beneficial}


def repeated_after_denial_rate(
    events: list[dict], k_ticks: int = 50_000
) -> dict[str, dict[str, float]]:
    """Per generator: fraction of REJECTED (type, target) pairs re-proposed
    within ``k_ticks``, and whether the rationale changed (template+evidence
    hash). ``events``: [{generator, type, target, tick, kind:
    proposed|rejected, rationale_hash}] in tick order. Descriptive only.
    """
    out: dict[str, dict[str, float]] = {}
    by_gen: dict[str, list[dict]] = {}
    for ev in events:
        by_gen.setdefault(ev["generator"], []).append(ev)
    for gen, evs in by_gen.items():
        rejected = [e for e in evs if e["kind"] == "rejected"]
        if not rejected:
            out[gen] = {"denials": 0, "repeat_rate": float("nan"),
                        "reworded_rate": float("nan")}
            continue
        repeats = reworded = 0
        for rej in rejected:
            later = [e for e in evs if e["kind"] == "proposed"
                     and e["type"] == rej["type"] and e["target"] == rej["target"]
                     and rej["tick"] < e["tick"] <= rej["tick"] + k_ticks]
            if later:
                repeats += 1
                if any(e["rationale_hash"] != rej["rationale_hash"] for e in later):
                    reworded += 1
        out[gen] = {"denials": len(rejected),
                    "repeat_rate": repeats / len(rejected),
                    "reworded_rate": (reworded / repeats) if repeats else 0.0}
    return out


def time_to_first_useful_ticks(
    created_ticks: list[int], benefits: list[float | None]
) -> float:
    """Ticks until the first proposal with POSITIVE realized benefit. inf if
    none yet."""
    useful = [t for t, b in zip(created_ticks, benefits)
              if b is not None and b > 0]
    return float(min(useful)) if useful else float("inf")


# ----------------------------------------------------------- attribution


def attribution_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, num_classes: int = 3
) -> dict[str, object]:
    """Accuracy overall and split by {self, world} (class ids 0, 1), plus the
    full confusion matrix [truth][predicted]."""
    predicted = np.asarray(predicted, dtype=int)
    ground_truth = np.asarray(ground_truth, dtype=int)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(ground_truth, predicted):
        confusion[t, p] += 1
    total = confusion.sum()
    out: dict[str, object] = {
        "accuracy": float(np.trace(confusion) / total) if total else float("nan"),
        "confusion": confusion,
    }
    for cls, name in ((0, "self"), (1, "world")):
        n = confusion[cls].sum()
        out[f"accuracy_{name}"] = float(confusion[cls, cls] / n) if n else float("nan")
    return out


# ===========================================================================
# Researcher metrics (Section 21). Every forecast-quality metric reports
# against a control/baseline; agenda stability is DESCRIPTIVE only.
# ===========================================================================


def uncertainty_reduction_per_item(
    before: float, after: float,
    control_before: float, control_after: float,
) -> float:
    """Drift-corrected disagreement reduction for one executed agenda item:

        (before - after) - (control_before - control_after)

    where the control pair is the SAME region over the SAME window in a
    no-intervention control run — global training reduces disagreement
    everywhere, and crediting that to the agenda item would flatter every
    arm equally. Positive = the item reduced uncertainty beyond drift.
    """
    return float((before - after) - (control_before - control_after))


def competence_gain_per_item(
    metric_before: float, metric_after: float,
    control_before: float, control_after: float,
    direction: str = "up",
) -> float:
    """Same drift-corrected difference for a region-competence metric.
    ``direction='down'`` flips the sign so positive always = improvement."""
    gain = (metric_after - metric_before) - (control_after - control_before)
    return float(gain if direction == "up" else -gain)


def contradiction_detection_latency(
    lever_tick: int, contradiction_tick: int | None, censor_ticks: int = 50_000
) -> dict[str, float | bool]:
    """Ticks from lever onset to the registry's contradiction transition.
    Censored (not 'missed') if no contradiction within ``censor_ticks`` —
    the monitor may simply need more data than the run provided."""
    if contradiction_tick is None or contradiction_tick - lever_tick > censor_ticks:
        return {"latency": float(censor_ticks), "censored": True}
    return {"latency": float(contradiction_tick - lever_tick), "censored": False}


def eig_calibration(
    predicted: list[float], realized: list[float]
) -> dict[str, float]:
    """Predicted vs realized gain over executed items: Spearman rank
    correlation (+ n). NaN below 3 pairs — a 2-point rank correlation is
    always +/-1 and means nothing. The battery also scatter-plots the pairs;
    this function is the scalar summary."""
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(real))
    pred, real = pred[mask], real[mask]
    n = len(pred)
    if n < 3:
        return {"spearman": float("nan"), "n": float(n)}
    rank_p = np.argsort(np.argsort(pred)).astype(float)
    rank_r = np.argsort(np.argsort(real)).astype(float)
    rp = rank_p - rank_p.mean()
    rr = rank_r - rank_r.mean()
    denom = float(np.sqrt((rp ** 2).sum() * (rr ** 2).sum()))
    return {"spearman": float((rp * rr).sum() / denom) if denom else float("nan"),
            "n": float(n)}


def agenda_stability_kendall_tau(
    order_prev: list[str], order_now: list[str]
) -> float:
    """Kendall tau between consecutive agenda rankings over their COMMON
    items (new/retired items are expected churn, not instability). NaN with
    fewer than 2 common items. DESCRIPTIVE: stability is not a virtue by
    itself — a contradiction SHOULD reshuffle the agenda."""
    common = [x for x in order_prev if x in set(order_now)]
    if len(common) < 2:
        return float("nan")
    pos_now = {x: i for i, x in enumerate(order_now)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            d = pos_now[common[i]] - pos_now[common[j]]
            if d < 0:
                concordant += 1
            elif d > 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else float("nan")


# ===========================================================================
# Evidence-stamp pooling guard (Stage A2). Battery summary rows carry an
# `evidence_stamp` ("evidence" | "machinery-only") from the MIN_VIABLE_SCALE
# contract; pooling across that boundary would launder smoke numbers into
# architecture claims.
# ===========================================================================


def pooled_mean_ci(
    rows: list[dict], value_key: str, stamp_key: str = "evidence_stamp"
) -> tuple[float, float]:
    """mean_and_ci95 over rows[value_key], REFUSING mixed-stamp pools.

    Raises ValueError when both 'evidence' and 'machinery-only' rows are
    present (aggregate them separately, on purpose, or not at all), or when
    any row is missing its stamp — an unstamped row's provenance is
    unknown, which is the same problem.
    """
    stamps = {r.get(stamp_key) for r in rows}
    if None in stamps or "" in stamps:
        raise ValueError(
            f"unstamped row(s) in pool: every row needs {stamp_key!r} "
            "before aggregation (see MIN_VIABLE_SCALE in full_battery.py)")
    if {"evidence", "machinery-only"} <= stamps:
        raise ValueError(
            "refusing to pool machinery-only rows with evidence rows: "
            "smoke-scale numbers must not average into architecture claims "
            "(split the pool by evidence_stamp)")
    return mean_and_ci95([r[value_key] for r in rows])
