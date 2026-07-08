"""Question generation: diagnostics -> structured research questions.

Four question types, each carrying evidence refs and candidate experiments
from a FIXED menu (probe-action batch, directed visit, targeted replay,
named battery) — plus every pending Stage-7 proposal, which enters the same
agenda as a candidate stream (researcher/agenda.py).

The SCOPE RULE applies to questions exactly as to hypotheses: statements
may concern the agent-in-the-world and its own models only (validated with
the registry's forbidden-pattern list).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from researcher.registry import _FORBIDDEN_PATTERNS, HypothesisRegistry

QUESTION_TYPES = ("world_uncertainty", "capability_gap",
                  "assumption_violation", "model_misfit")

# The fixed experiment menu: name -> (cost in budget units, params template).
EXPERIMENT_MENU = {
    "probe_action_batch": {"cost": 1.0},
    "directed_visit": {"cost": 2.0},
    "targeted_replay": {"cost": 1.5},
    "run_battery": {"cost": 4.0},
}


@dataclass
class Question:
    id: str
    type: str
    statement: str
    evidence_refs: list[str]
    candidate_experiments: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0        # normalized [0,1]; the v1 value term
    region: tuple[int, int] | None = None  # for tractability lookup

    def validate(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {self.type!r}")
        text = (self.statement + " " + json.dumps(self.params)).lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in text:
                raise ValueError(
                    f"SCOPE RULE: question references {pattern!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------- run inputs


def _read_viz_state(run_dir: Path) -> dict[str, Any] | None:
    """Local pickle reader (training.loggers has one, but researcher/ may
    not import training — structural rule)."""
    path = run_dir / "viz_state.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None


def _action_success_shift(run_dir: Path, num_actions: int = 9,
                          window: int = 4000) -> dict[int, float]:
    """Per-action success-rate shift (sd units) between the last two windows
    of the JSONL stream — the capability_gap signal."""
    records = []
    for chunk in sorted(run_dir.glob("events-*.jsonl")):
        with open(chunk, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    out: dict[int, float] = {}
    if not records:
        return out
    last = records[-1]["tick"]
    for action in range(num_actions):
        now = [float(r["success"]) for r in records
               if r["action"] == action and r["tick"] >= last - window]
        prev = [float(r["success"]) for r in records
                if r["action"] == action
                and last - 2 * window <= r["tick"] < last - window]
        if len(now) >= 10 and len(prev) >= 10:
            # Same std floor as the registry's mean_shift: a constant prev
            # window (all-success) must not turn one change into an
            # astronomical sd count.
            denom = max(float(np.std(prev)), 0.05)
            out[action] = float(abs(np.mean(now) - np.mean(prev)) / denom)
    return out


def generate_questions(
    run_dir: str | Path,
    registry: HypothesisRegistry | None = None,
    top_k_cells: int = 3,
    region_size: int = 8,
) -> list[Question]:
    """Convert one run's diagnostics into validated questions."""
    run_dir = Path(run_dir)
    questions: list[Question] = []

    # world_uncertainty: top-k epistemic-map cells.
    state = _read_viz_state(run_dir)
    emap = state.get("epistemic_map") if state else None
    if emap is not None and np.any(emap > 0):
        peak = float(emap.max())
        flat = emap.flatten()
        order = np.argsort(-flat)[:top_k_cells]
        for rank, idx in enumerate(order):
            y, x = divmod(int(idx), emap.shape[1])
            region = (y // region_size, x // region_size)
            level = float(flat[idx]) / peak
            q = Question(
                id=f"q-world-{y}-{x}",
                type="world_uncertainty",
                statement=(f"what are the dynamics of region {region} "
                           f"(cell {x},{y})? ensemble disagreement is at "
                           f"{level:.0%} of the map peak"),
                evidence_refs=[f"viz_state:epistemic_map:cell=({x},{y})"],
                candidate_experiments=[
                    {"name": "directed_visit", "region": region,
                     **EXPERIMENT_MENU["directed_visit"]},
                    {"name": "targeted_replay", "region": region,
                     **EXPERIMENT_MENU["targeted_replay"]},
                ],
                params={"cell": [x, y], "rank": rank},
                uncertainty=level, region=region,
            )
            q.validate()
            questions.append(q)

    # capability_gap: actions whose success rate shifted most.
    shifts = _action_success_shift(run_dir)
    for action, shift in sorted(shifts.items(), key=lambda kv: -kv[1])[:2]:
        if shift < 1.0:
            continue
        q = Question(
            id=f"q-capability-{action}",
            type="capability_gap",
            statement=(f"is capability (action {action}) degraded or "
                       f"mis-modeled? success rate shifted {shift:.1f} sd "
                       f"between recent windows"),
            evidence_refs=[f"jsonl:action_success:action={action}"],
            candidate_experiments=[
                {"name": "probe_action_batch", "action": action, "n": 32,
                 **EXPERIMENT_MENU["probe_action_batch"]},
            ],
            params={"action": action, "shift_sd": shift},
            uncertainty=float(min(1.0, shift / 5.0)),
        )
        q.validate()
        questions.append(q)

    # assumption_violation: one per weakening/contradicted hypothesis.
    if registry is not None:
        for h in registry.hypotheses.values():
            if h.status not in ("weakening", "contradicted"):
                continue
            q = Question(
                id=f"q-violation-{h.id}",
                type="assumption_violation",
                statement=(f"standing assumption {h.status}: "
                           f"{h.statement()} — what changed?"),
                evidence_refs=[f"hypothesis:{h.id}"]
                + [f"transition:{t['tick']}:{t['to']}" for t in h.transitions[-2:]],
                candidate_experiments=[
                    {"name": "directed_visit",
                     "region": (h.params.get("r", 0), h.params.get("c", 0)),
                     **EXPERIMENT_MENU["directed_visit"]}
                    if h.scope == "world" else
                    {"name": "probe_action_batch",
                     "action": h.params.get("action", 0), "n": 32,
                     **EXPERIMENT_MENU["probe_action_batch"]},
                ],
                params=dict(h.params),
                uncertainty=1.0 if h.status == "contradicted" else 0.7,
                region=((h.params.get("r"), h.params.get("c"))
                        if "r" in h.params else None),
            )
            q.validate()
            questions.append(q)

    # model_misfit: forecaster NMSE / reliability residual patterns from TB.
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        tb_dir = run_dir / "tb"
        if tb_dir.exists():
            acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                if not tag.startswith("sleep/forecaster_nmse_k"):
                    continue
                values = [e.value for e in acc.Scalars(tag)]
                if len(values) >= 3 and np.mean(values[-3:]) > 2.0:
                    horizon = tag.rsplit("k", 1)[-1]
                    q = Question(
                        id=f"q-misfit-forecaster-k{horizon}",
                        type="model_misfit",
                        statement=(f"the forecaster at horizon {horizon} is "
                                   f"not modeling dynamics (NMSE "
                                   f"{np.mean(values[-3:]):.1f} >= 2)"),
                        evidence_refs=[f"tb:{tag}"],
                        candidate_experiments=[
                            {"name": "run_battery", "battery": "forecaster_nmse",
                             **EXPERIMENT_MENU["run_battery"]},
                        ],
                        params={"horizon": horizon},
                        uncertainty=float(min(1.0, np.mean(values[-3:]) / 20.0)),
                    )
                    q.validate()
                    questions.append(q)
    except Exception:  # tb log absent/corrupt: model_misfit questions skipped
        pass

    return questions
