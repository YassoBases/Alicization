"""Simulated continuity metric suite — a DEPENDENT variable, nothing more.

Composite diagnostic per eval window: weighted z-scores of
  preference persistence   1 - JS(action histogram, previous window)
  revisit efficiency       distinct previously-known cells re-reached per
                           movement tick
  forecaster NMSE trend    -slope of the sleep forecaster NMSE (falling =
                           better dynamics modeling)
  reliability ECE          -ECE (better calibration)
  adaptation half-life     -recovery ticks from the window's deepest reward
                           dip (censored at the window length)

Weights and window come from config (``continuity.weights``,
``continuity.window_ticks``); components are ALWAYS reported individually
alongside the composite. z-scores are computed across the pooled windows of
the runs under comparison, so composites are comparable within one
``compare_runs`` call and meaningless across calls by construction.

HARD RULES (CLAUDE.md; tests/test_continuity_enforcement.py):
- Never couples to process lifetime, shutdown, or researcher actions: this
  module reads run logs only. Its imports are json/pathlib/numpy + the
  in-repo TB reader — no os, no process or system-time access, no world or
  training imports.
- Appears in no loss, reward, or policy-input construction anywhere: it is
  used to compare learning mechanisms across runs and to score proposal
  outcomes, and for nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


import numpy as np

COMPONENTS = ("preference_persistence", "revisit_efficiency",
              "forecaster_nmse_trend", "reliability_ece",
              "adaptation_half_life")

DEFAULT_WEIGHTS = {name: 1.0 for name in COMPONENTS}


@dataclass
class ContinuityResult:
    run_id: str
    window_ticks: int
    components: dict[str, list[float]] = field(default_factory=dict)
    composite: list[float] = field(default_factory=list)

    def mean_ci(self) -> tuple[float, float]:
        arr = np.asarray(self.composite, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        if arr.size == 1:
            return float(arr[0]), float("nan")
        return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


# ------------------------------------------------------------- raw readers


def _read_jsonl(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ticks: list[int] = []
    positions: list[list[int]] = []
    actions: list[int] = []
    rewards: list[float] = []
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        with open(chunk, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ticks.append(rec["tick"])
                positions.append(rec["pos"])
                actions.append(rec["action"])
                rewards.append(rec["reward"])
    del ticks
    return (np.asarray(positions, dtype=int), np.asarray(actions, dtype=int),
            np.asarray(rewards, dtype=float))


def _read_scalar(run_dir: Path, tag_prefix: str) -> np.ndarray:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return np.zeros(0)
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags().get("scalars", []):
        if tag.startswith(tag_prefix):
            return np.asarray([e.value for e in acc.Scalars(tag)], dtype=float)
    return np.zeros(0)


# ------------------------------------------------------------- components


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _window_components(
    win_actions: np.ndarray,
    prev_actions: np.ndarray | None,
    win_positions: np.ndarray,
    known_cells: set[tuple[int, int]],
    win_rewards: np.ndarray,
    nmse_series: np.ndarray,
    ece_series: np.ndarray,
    num_actions: int = 9,
) -> dict[str, float]:
    out: dict[str, float] = {}

    if prev_actions is None or not len(prev_actions):
        out["preference_persistence"] = float("nan")
    else:
        hist_now = np.bincount(win_actions, minlength=num_actions).astype(float)
        hist_prev = np.bincount(prev_actions, minlength=num_actions).astype(float)
        out["preference_persistence"] = 1.0 - _js(hist_now, hist_prev)

    moved = np.any(np.diff(win_positions, axis=0) != 0, axis=1)
    move_ticks = int(moved.sum())
    revisited = {tuple(p) for p in win_positions if tuple(p) in known_cells}
    out["revisit_efficiency"] = (
        len(revisited) / move_ticks if move_ticks else float("nan")
    )

    if len(nmse_series) >= 3:
        xs = np.arange(len(nmse_series), dtype=float)
        out["forecaster_nmse_trend"] = -float(np.polyfit(xs, nmse_series, 1)[0])
    else:
        out["forecaster_nmse_trend"] = float("nan")

    out["reliability_ece"] = -float(ece_series[-1]) if len(ece_series) else float("nan")

    # Deepest dip in the window's reward; recovery ticks back to 90% of the
    # window-start level (censored at the window length).
    w = max(10, len(win_rewards) // 20)
    kernel = np.ones(w) / w
    smoothed = np.convolve(win_rewards, kernel, mode="valid")
    if len(smoothed) < 3:
        out["adaptation_half_life"] = float("nan")
        return out
    start_level = smoothed[: max(1, len(smoothed) // 10)].mean()
    dip = int(np.argmin(smoothed))
    target = 0.9 * start_level
    after = np.nonzero(smoothed[dip:] >= target)[0]
    half_life = float(after[0]) if after.size else float(len(win_rewards))
    out["adaptation_half_life"] = -half_life
    return out


# ------------------------------------------------------------------- main


def compute_run(run_dir: str | Path, window_ticks: int = 4096) -> ContinuityResult:
    """Per-window raw components for one run (z-scoring happens in
    compare_runs, across the pool)."""
    run_dir = Path(run_dir)
    positions, actions, rewards = _read_jsonl(run_dir)
    nmse = _read_scalar(run_dir, "sleep/forecaster_nmse")
    ece = _read_scalar(run_dir, "ledger/reliability_ece")

    result = ContinuityResult(run_id=run_dir.name, window_ticks=window_ticks,
                              components={name: [] for name in COMPONENTS})
    n_windows = len(actions) // window_ticks
    known_cells: set[tuple[int, int]] = set()
    prev_actions: np.ndarray | None = None
    for k in range(n_windows):
        sl = slice(k * window_ticks, (k + 1) * window_ticks)
        # Per-window slices of the sleep-cadence scalars (approximate: they
        # are far sparser than ticks; take the proportional slice).
        def _slice_scalar(series: np.ndarray) -> np.ndarray:
            if not len(series):
                return series
            lo = int(len(series) * k / n_windows)
            hi = max(lo + 1, int(len(series) * (k + 1) / n_windows))
            return series[lo:hi]

        comps = _window_components(
            actions[sl], prev_actions, positions[sl], known_cells,
            rewards[sl], _slice_scalar(nmse), _slice_scalar(ece),
        )
        for name in COMPONENTS:
            result.components[name].append(comps[name])
        known_cells.update(tuple(p) for p in positions[sl])
        prev_actions = actions[sl]
    return result


def compare_runs(
    run_dirs: list[str | Path],
    weights: dict[str, float] | None = None,
    window_ticks: int = 4096,
) -> dict[str, ContinuityResult]:
    """Compute components for every run, z-score each component across the
    POOLED windows, and fill in per-window composites."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    results = {Path(d).name: compute_run(d, window_ticks) for d in run_dirs}

    for name in COMPONENTS:
        pooled = np.asarray(
            [v for r in results.values() for v in r.components[name]], dtype=float)
        finite = pooled[np.isfinite(pooled)]
        mu = finite.mean() if finite.size else 0.0
        sd = finite.std(ddof=1) if finite.size > 1 else 1.0
        sd = sd if sd > 1e-12 else 1.0
        for r in results.values():
            z = [(v - mu) / sd if np.isfinite(v) else np.nan
                 for v in r.components[name]]
            r.components[f"z_{name}"] = z

    total_w = sum(abs(w) for w in weights.values()) or 1.0
    for r in results.values():
        n = len(next(iter(r.components.values()), []))
        composites = []
        for i in range(n):
            terms = [weights[name] * r.components[f"z_{name}"][i]
                     for name in COMPONENTS
                     if np.isfinite(r.components[f"z_{name}"][i])]
            composites.append(sum(terms) / total_w if terms else float("nan"))
        r.composite = composites
    return results
