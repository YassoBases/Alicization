"""Versioned proposal schema: dataclass <-> JSON, validated on load.

Proposals are DATA, NEVER CODE (CLAUDE.md Hard rules): this module defines
records for a human to read and decide on; nothing here executes anything,
and the only write target is runs/<id>/proposals/ (path-validated).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

TYPES = (
    "retraining", "training_schedule", "hyperparameter", "memory_policy",
    "checkpoint_schedule", "evaluation", "logging_change", "compute_budget",
    "dataset_extension", "visualization",
)
SOURCES = ("ledger", "logs_only")
STATUSES = ("pending", "approved", "partially_approved", "modified",
            "rejected", "postponed", "executed", "evaluated")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{7,63}$")


@dataclass
class Proposal:
    """One proposal record. See TYPES/SOURCES/STATUSES for the enums."""

    schema_version: int
    id: str
    type: str
    created_tick: int
    run_id: str
    source: str                      # ledger | logs_only (BLIND in review)
    rationale: str                   # templated text citing log records
    expected_benefit: dict[str, Any]  # {metric, direction, magnitude_estimate}
    confidence: float                # [0, 1]
    supporting_observations: list[str]  # log-record refs
    estimated_cost: dict[str, float]    # {human_hours, gpu_hours}
    risks: list[str]
    success_criteria: dict[str, Any]    # {metric, threshold, eval_window_ticks}
    status: str = "pending"
    decision: dict[str, Any] = field(default_factory=dict)
    #   {timestamp, note, usefulness_rating (1-5, optional), human_diff?}
    linked_experiment_id: str | None = None
    realized_benefit: dict[str, Any] | None = None  # filled after evaluation
    target: str = ""                 # dedup key component (e.g. region/knob)

    # ------------------------------------------------------------ lifecycle

    @staticmethod
    def new(**kwargs: Any) -> "Proposal":
        kwargs.setdefault("schema_version", SCHEMA_VERSION)
        kwargs.setdefault("id", f"prop-{uuid.uuid4().hex[:12]}")
        p = Proposal(**kwargs)
        p.validate()
        return p

    def dedup_hash(self) -> str:
        """Proposals with the same (type, target) are duplicates."""
        return hashlib.sha256(f"{self.type}|{self.target}".encode()).hexdigest()[:16]

    # ----------------------------------------------------------- validation

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        if self.type not in TYPES:
            raise ValueError(f"unknown type {self.type!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of [0,1]: {self.confidence}")
        for key in ("metric", "direction", "magnitude_estimate"):
            if key not in self.expected_benefit:
                raise ValueError(f"expected_benefit missing {key!r}")
        for key in ("metric", "threshold", "eval_window_ticks"):
            if key not in self.success_criteria:
                raise ValueError(f"success_criteria missing {key!r}")
        for key in ("human_hours", "gpu_hours"):
            if key not in self.estimated_cost:
                raise ValueError(f"estimated_cost missing {key!r}")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be non-empty text")

    # ------------------------------------------------------------------ json

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Proposal":
        p = Proposal(**json.loads(text))
        p.validate()
        return p


# ------------------------------------------------------------- persistence


def proposals_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "proposals"


def save_proposal(proposal: Proposal, run_dir: str | Path) -> Path:
    """Write runs/<id>/proposals/<proposal-id>.json — the ONLY write target
    this package has (CLAUDE.md Hard rules). Path-validated: a proposal id
    that would escape the proposals dir raises instead of writing."""
    proposal.validate()
    if not _ID_RE.match(proposal.id):
        raise ValueError(f"unsafe proposal id {proposal.id!r}")
    out_dir = proposals_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = (out_dir / f"{proposal.id}.json").resolve()
    if path.parent != out_dir.resolve():
        raise ValueError("write outside the proposals dir refused")
    path.write_text(proposal.to_json(), encoding="utf-8")
    return path


def load_proposal(path: str | Path) -> Proposal:
    return Proposal.from_json(Path(path).read_text(encoding="utf-8"))


def load_all(run_dir: str | Path) -> list[Proposal]:
    d = proposals_dir(run_dir)
    if not d.exists():
        return []
    return sorted(
        (load_proposal(p) for p in d.glob("prop-*.json")
         if not p.name.endswith(".edit.json")),  # review's in-flight edit copies
        key=lambda p: (p.created_tick, p.id),
    )
