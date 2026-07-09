"""Competence as aggregation from SelfQ query errors (stage-E3).

In heads mode the competence tracker's per-region body-Brier comes from the
body model's telemetry and its forecaster-NMSE from the forecaster's — two
separate heads. In selfq mode BOTH derive from one model's logged query
errors: the wake body-Brier already flows from SelfQ (the body adapter is
SelfQ), and this module adds the missing per-region forecaster-NMSE feed,
computed from SelfQ's horizon queries against the stored realized futures.

The CompetenceReport schema is UNCHANGED (forecaster_nmse_ema already
exists; it was simply never fed per-region before). Pure aggregation over
detached tensors — no autograd, no new head telemetry.
"""

from __future__ import annotations

from typing import Any

import torch

from selfq.model import INTENT_PLAN, SelfQ


def feed_forecast_competence(tracker: Any, batch: dict[str, Any], selfq: SelfQ,
                             horizon: int | None = None) -> int:
    """Feed per-region forecaster NMSE (SelfQ vs the identity baseline) into
    ``tracker`` from a ForecastTupleStore batch carrying positions. Returns
    the number of regions fed. No-op without positions (heads-mode batches)."""
    positions = batch.get("pos")
    if not positions:
        return 0
    futures = batch["future"]
    horizon = horizon if horizon in futures else max(futures)
    with torch.no_grad():
        pred = selfq.query(batch["h"], batch["plan"], INTENT_PLAN, horizon)
        forecast = pred.intero_mean
        target = futures[horizon]
        now = batch["intero_now"]
        mse_f = ((forecast - target) ** 2).mean(dim=-1)   # (B,)
        mse_i = ((now - target) ** 2).mean(dim=-1)        # identity baseline

    # Aggregate MSEs per region, then form the region NMSE = sum_f / sum_i
    # (a ratio of pooled MSEs is far less noisy than per-sample ratios).
    agg: dict[tuple[int, int], list[float]] = {}
    rep_pos: dict[tuple[int, int], tuple[int, int]] = {}
    for i, pos in enumerate(positions):
        if pos is None:
            continue
        region = tracker.region_of(pos)
        acc = agg.setdefault(region, [0.0, 0.0])
        acc[0] += float(mse_f[i])
        acc[1] += float(mse_i[i])
        rep_pos.setdefault(region, tuple(int(x) for x in pos))

    fed = 0
    for region, (sum_f, sum_i) in agg.items():
        if sum_i <= 0.0:
            continue  # identity is unbeatable (nothing changed) — skip
        tracker.update_forecast(rep_pos[region], sum_f / sum_i)
        fed += 1
    return fed
