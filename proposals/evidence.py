"""Evidence bundles for the proposal generators.

Two variants of the same run's diagnostics (the control condition for the
whole stage):
  source="ledger"    — competence reports + Ledger scalars + raw logs.
  source="logs_only" — the SAME bundle with every Ledger-derived field
                       stripped; only raw training logs (reward/loss curves,
                       action histograms, positions) remain.

This module reads runs/<id>/ artifacts directly (TB event files, competence
JSON, per-tick JSONL). Import rules (tests/test_proposals_no_execution.py):
no world/training/agent imports — the tensorboard reader and json/numpy are
the whole toolbox.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ledger.competence import CompetenceReport

# Scalar tags that are Ledger self-model outputs: stripped in the logs_only
# variant. Raw task/optimization logs (reward, loss/*, entropy, clip_frac,
# sps, sleep/grad_steps, phase markers) remain in both.
LEDGER_TAG_PREFIXES = ("ledger/", "rssm/", "mirror/", "memory/", "sleep/forecaster")


@dataclass
class Evidence:
    """Everything a generator may read. Ledger fields are None in the
    logs_only variant."""

    source: str
    run_id: str
    tick: int
    scalars: dict[str, tuple[list[int], list[float]]]
    competence: CompetenceReport | None = None
    positions: np.ndarray | None = None      # (N, 2) world coords from JSONL
    actions: np.ndarray | None = None        # (N,) from JSONL
    config: dict[str, Any] = field(default_factory=dict)

    def series(self, tag: str) -> np.ndarray:
        steps_values = self.scalars.get(tag)
        return np.asarray(steps_values[1], dtype=float) if steps_values else np.zeros(0)

    def first_tag(self, *candidates: str) -> str:
        """First candidate tag with data — the two trainers name equivalent
        scalars differently (rssm/* vs sleep/*); citations must reference a
        record that exists in THIS run."""
        for tag in candidates:
            if self.scalars.get(tag, ([], []))[0]:
                return tag
        return candidates[0]

    def ref(self, tag: str, index: int = -1) -> str:
        """A citable log-record reference for supporting_observations."""
        steps_values = self.scalars.get(tag)
        if not steps_values or not steps_values[0]:
            return f"tb:{tag}"
        idx = index if index >= 0 else len(steps_values[0]) + index
        return f"tb:{tag}@step={steps_values[0][idx]}"


def _read_tb_scalars(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return {}
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def _read_jsonl(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[int]] = []
    actions: list[int] = []
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        with open(chunk, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                positions.append(rec["pos"])
                actions.append(rec["action"])
    return np.asarray(positions, dtype=int), np.asarray(actions, dtype=int)


def _read_latest_competence(run_dir: Path) -> CompetenceReport | None:
    reports = sorted(run_dir.glob("competence/report-*.json"))
    if not reports:
        return None
    return CompetenceReport.from_json(reports[-1].read_text(encoding="utf-8"))


def evidence_from_run(run_dir: str | Path, source: str) -> Evidence:
    """Build one evidence bundle from a run dir. ``source`` selects the
    variant; logs_only strips every Ledger-derived signal."""
    run_dir = Path(run_dir)
    scalars = _read_tb_scalars(run_dir)
    positions, actions = _read_jsonl(run_dir)
    config: dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    tick = max((sv[0][-1] for sv in scalars.values() if sv[0]), default=0)

    if source == "logs_only":
        scalars = {
            tag: sv for tag, sv in scalars.items()
            if not tag.startswith(LEDGER_TAG_PREFIXES)
        }
        competence = None
    else:
        competence = _read_latest_competence(run_dir)
    return Evidence(
        source=source, run_id=run_dir.name, tick=tick, scalars=scalars,
        competence=competence,
        positions=positions if len(positions) else None,
        actions=actions if len(actions) else None,
        config=config,
    )
