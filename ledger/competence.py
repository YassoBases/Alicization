"""Level-6 competence tracker: per-region, per-task rolling self-assessment.

Maintains, per 8x8 world region and per task key (macro-plan name, or "all"),
rolling estimates (EMA + trailing best) of: world-model loss, body-model
Brier, forecaster NMSE, reward rate, and learning progress (negative
derivative of the world-model-loss EMA over a window). Every sleep phase it
emits a typed CompetenceReport (dataclass -> JSON): current-vs-trailing-best
ratios, an adaptation-status flag per region (stable / degrading /
mid-adaptation), and replay coverage per region.

Reports are logged (runs/<id>/competence/), checkpointed, and READ-ONLY to
everything except the proposal layer and the dashboard.

GRADIENT ISOLATION: this module computes from detached logs only — it
imports numpy, never torch, so autograd cannot exist here (a test asserts
the import list). Trainers pass plain floats/arrays.

Flag semantics (loss-like metrics; lower is better):
  ratio      = wm_loss_ema / trailing_min(wm_loss_ema)   (1.0 = at best)
  degrading      : ratio > degrade_ratio AND learning progress <= 0
                   (worse than we have ever been here, and not improving)
  mid-adaptation : ratio > degrade_ratio AND learning progress > 0
                   (still worse, but the loss is falling — adapting)
  stable         : ratio <= degrade_ratio
The spec's "current-vs-trailing-max" is the reward-side phrasing; for
loss-like metrics the trailing BEST is a minimum, reported symmetrically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STABLE, DEGRADING, MID_ADAPTATION = "stable", "degrading", "mid-adaptation"

REPORT_SCHEMA_VERSION = 1


@dataclass
class RegionCompetence:
    """One region x task cell of a competence report."""

    region: tuple[int, int]          # (row, col) in region units
    task: str
    n_samples: int
    wm_loss_ema: float
    wm_loss_ratio: float             # ema / trailing min (1.0 = at best)
    body_brier_ema: float
    body_brier_ratio: float
    forecaster_nmse_ema: float       # nan when never fed (non-arbiter runs)
    reward_rate_ema: float
    reward_ratio: float              # ema / trailing max (1.0 = at best)
    learning_progress: float         # -d(wm_loss_ema)/dt over the window
    adaptation_status: str           # stable | degrading | mid-adaptation
    replay_coverage: float           # fraction of replay transitions here


@dataclass
class CompetenceReport:
    """Emitted every sleep phase; serialized to JSON next to the run."""

    schema_version: int
    tick: int
    run_id: str
    regions: list[RegionCompetence] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "CompetenceReport":
        raw = json.loads(text)
        if raw.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported competence schema: {raw.get('schema_version')}")
        regions = [RegionCompetence(**{**r, "region": tuple(r["region"])})
                   for r in raw.pop("regions")]
        return CompetenceReport(regions=regions, **raw)


class _Cell:
    """Rolling state for one (region, task) cell. Plain floats only.

    Trailing bests are WINDOWED (min/max over the last ``trail_window`` EMA
    snapshots), not all-time: the world-model surprise proxy is artificially
    LOW at init (posterior ~ prior before training), so an all-time best
    anchors on the untrained artifact and flags every region degraded
    forever. A trailing window lets early artifacts age out, so the flags
    reflect recent worsening vs recent best — which is what adaptation
    tracking needs.
    """

    __slots__ = ("n", "wm", "wm_snaps", "wm_history", "brier", "brier_snaps",
                 "nmse", "reward", "reward_snaps")

    def __init__(self) -> None:
        self.n = 0
        self.wm = float("nan")
        self.wm_snaps: list[float] = []
        self.wm_history: list[float] = []
        self.brier = float("nan")
        self.brier_snaps: list[float] = []
        self.nmse = float("nan")
        self.reward = float("nan")
        self.reward_snaps: list[float] = []


class CompetenceTracker:
    """Feed per-tick observations; emit a CompetenceReport per sleep phase."""

    def __init__(
        self,
        world_size: int,
        region_size: int = 8,
        ema_decay: float = 0.99,
        progress_window: int = 20,
        trail_window: int = 50,
        degrade_ratio: float = 1.5,
        min_samples: int = 50,
    ) -> None:
        self.region_size = region_size
        self.n_regions = (world_size + region_size - 1) // region_size
        self.ema_decay = ema_decay
        self.progress_window = progress_window
        self.trail_window = trail_window
        self.degrade_ratio = degrade_ratio
        self.min_samples = min_samples
        self._cells: dict[tuple[int, int, str], _Cell] = {}

    # ------------------------------------------------------------------ feed

    def _cell(self, region: tuple[int, int], task: str) -> _Cell:
        key = (region[0], region[1], task)
        if key not in self._cells:
            self._cells[key] = _Cell()
        return self._cells[key]

    def region_of(self, pos: tuple[int, int]) -> tuple[int, int]:
        return int(pos[1]) // self.region_size, int(pos[0]) // self.region_size

    def update_tick(
        self,
        pos: tuple[int, int],
        wm_loss: float,
        body_brier: float,
        reward: float,
        task: str = "all",
    ) -> None:
        """One tick of detached observations at world position ``pos``."""
        cell = self._cell(self.region_of(pos), task)
        d = self.ema_decay
        cell.n += 1

        def ema(old: float, new: float) -> float:
            return new if np.isnan(old) else d * old + (1 - d) * new

        cell.wm = ema(cell.wm, wm_loss)
        cell.brier = ema(cell.brier, body_brier)
        cell.reward = ema(cell.reward, reward)

    def update_forecast(
        self, pos: tuple[int, int], nmse: float, task: str = "all"
    ) -> None:
        """Per-region forecaster NMSE (arbiter runs; optional elsewhere)."""
        cell = self._cell(self.region_of(pos), task)
        d = self.ema_decay
        cell.nmse = nmse if np.isnan(cell.nmse) else d * cell.nmse + (1 - d) * nmse

    def snapshot_progress(self) -> None:
        """Record one snapshot of each cell's EMAs (call on a fixed cadence,
        e.g. once per rollout): learning progress = -slope of the wm EMA over
        the last ``progress_window`` snapshots, and the TRAILING bests are
        min/max over the last ``trail_window`` snapshots."""
        for cell in self._cells.values():
            if np.isnan(cell.wm):
                continue
            cell.wm_history.append(cell.wm)
            if len(cell.wm_history) > self.progress_window:
                cell.wm_history.pop(0)
            if cell.n < self.min_samples:
                continue
            for value, snaps in ((cell.wm, cell.wm_snaps),
                                 (cell.brier, cell.brier_snaps),
                                 (cell.reward, cell.reward_snaps)):
                if not np.isnan(value):
                    snaps.append(float(value))
                    if len(snaps) > self.trail_window:
                        snaps.pop(0)

    # ---------------------------------------------------------------- report

    @staticmethod
    def _progress(history: list[float]) -> float:
        if len(history) < 3:
            return 0.0
        xs = np.arange(len(history), dtype=float)
        slope = float(np.polyfit(xs, np.asarray(history), 1)[0])
        return -slope  # falling loss = positive progress

    def report(
        self,
        tick: int,
        run_id: str,
        replay_positions: np.ndarray | None = None,
        world_size: int | None = None,
    ) -> CompetenceReport:
        """Build the report. ``replay_positions`` (N, 2) NORMALIZED coords
        from the replay buffer give per-region replay coverage."""
        coverage: dict[tuple[int, int], float] = {}
        if replay_positions is not None and len(replay_positions) and world_size:
            cells = (replay_positions * world_size).astype(int) // self.region_size
            keys, counts = np.unique(cells[:, [1, 0]], axis=0, return_counts=True)
            total = counts.sum()
            coverage = {
                (int(r), int(c)): float(n) / total for (r, c), n in zip(keys, counts)
            }

        out = CompetenceReport(schema_version=REPORT_SCHEMA_VERSION,
                               tick=tick, run_id=run_id)
        for (r, c, task), cell in sorted(self._cells.items()):
            if cell.n < self.min_samples:
                continue
            wm_best = min(cell.wm_snaps) if cell.wm_snaps else float("nan")
            brier_best = min(cell.brier_snaps) if cell.brier_snaps else float("nan")
            reward_best = max(cell.reward_snaps) if cell.reward_snaps else float("nan")
            wm_ratio = (cell.wm / wm_best
                        if np.isfinite(wm_best) and wm_best > 0 else 1.0)
            brier_ratio = (cell.brier / brier_best
                           if np.isfinite(brier_best) and brier_best > 0 else 1.0)
            reward_ratio = (cell.reward / reward_best
                            if np.isfinite(reward_best) and abs(reward_best) > 1e-12
                            else 1.0)
            progress = self._progress(cell.wm_history)
            if wm_ratio > self.degrade_ratio:
                status = MID_ADAPTATION if progress > 0 else DEGRADING
            else:
                status = STABLE
            out.regions.append(RegionCompetence(
                region=(r, c), task=task, n_samples=cell.n,
                wm_loss_ema=float(cell.wm), wm_loss_ratio=float(wm_ratio),
                body_brier_ema=float(cell.brier), body_brier_ratio=float(brier_ratio),
                forecaster_nmse_ema=float(cell.nmse),
                reward_rate_ema=float(cell.reward), reward_ratio=float(reward_ratio),
                learning_progress=float(progress),
                adaptation_status=status,
                replay_coverage=coverage.get((r, c), 0.0),
            ))
        return out

    def write_report(self, report: CompetenceReport, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "competence"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"report-{report.tick:012d}.json"
        path.write_text(report.to_json(), encoding="utf-8")
        return path

    # ----------------------------------------------------------------- state

    def state_dict(self) -> dict[str, Any]:
        return {
            "cells": {
                "|".join(map(str, key)): {
                    "n": cell.n, "wm": cell.wm,
                    "wm_snaps": list(cell.wm_snaps),
                    "wm_history": list(cell.wm_history),
                    "brier": cell.brier, "brier_snaps": list(cell.brier_snaps),
                    "nmse": cell.nmse, "reward": cell.reward,
                    "reward_snaps": list(cell.reward_snaps),
                }
                for key, cell in self._cells.items()
            }
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._cells.clear()
        for key_str, raw in state["cells"].items():
            r, c, task = key_str.split("|", 2)
            cell = _Cell()
            cell.n = raw["n"]
            cell.wm = raw["wm"]
            cell.wm_snaps = list(raw["wm_snaps"])
            cell.wm_history = list(raw["wm_history"])
            cell.brier = raw["brier"]
            cell.brier_snaps = list(raw["brier_snaps"])
            cell.nmse, cell.reward = raw["nmse"], raw["reward"]
            cell.reward_snaps = list(raw["reward_snaps"])
            self._cells[(int(r), int(c), task)] = cell
