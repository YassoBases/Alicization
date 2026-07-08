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

SCHEMA_VERSION = 2

TYPES = (
    "retraining", "training_schedule", "hyperparameter", "memory_policy",
    "checkpoint_schedule", "evaluation", "logging_change", "compute_budget",
    "dataset_extension", "visualization",
)
# What KIND of intervention the proposal is (stage-C2): a config knob, an
# experiment to run, or an architecture change (stage-D Architect). The
# evaluation ladder (experiments/runner.py) keys tier-0 auto-A/B off "config".
INTERVENTION_CLASSES = ("config", "experiment", "architecture")
# KNOWN sources (the ledger-vs-logs control condition, plus stage-D
# "architect"); NOT an enforced enum — source is validated non-empty so new
# generators (e.g. "architect:sonnet") need no schema bump. Blinding keys off
# status ("not yet evaluated"), never off source membership.
SOURCES = ("ledger", "logs_only", "architect")
STATUSES = ("pending", "approved", "partially_approved", "modified",
            "rejected", "postponed", "executed", "evaluated")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{7,63}$")


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an older serialized proposal to the current schema, filling
    defaults. v1 -> v2: add intervention_class (knob proposals are config
    interventions, everything else is an experiment), provenance, artifacts."""
    version = data.get("schema_version", 1)
    if version == SCHEMA_VERSION:
        return data
    if version == 1:
        data = dict(data)
        data["schema_version"] = SCHEMA_VERSION
        data.setdefault("provenance", {})
        data.setdefault("artifacts", [])
        data.setdefault(
            "intervention_class",
            "config" if data.get("proposed_change") else "experiment")
        return data
    raise ValueError(f"cannot migrate proposal schema_version {version!r}")


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
    # Machine-readable form of the recommendation WHEN it is a config knob:
    # {"config_path": "rssm.free_nats", "new_value": 0.5}. None for proposals
    # with no single-knob form (retraining, logging changes, ...). Enables
    # A/B realized-benefit evaluation (Section 17); the human still executes.
    proposed_change: dict[str, Any] | None = None
    # --- schema v2 (stage-C2) ---
    intervention_class: str = "config"   # config | experiment | architecture
    # Reproducibility (standing rule): where this proposal came from.
    # {evidence_bundle_hash, generator_id} for rule generators; adds
    # {prompt_hash, model_id} for LLM-drafted ones (stage-D).
    provenance: dict[str, Any] = field(default_factory=dict)
    # Run-relative paths to attached data files (e.g. an UNAPPLIED diff).
    # Never absolute, never containing "..": these live under runs/<id>/.
    artifacts: list[str] = field(default_factory=list)

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
        # source is an OPEN string (v2): validated non-empty, not enum-checked
        # — blinding keys off status, not off a known-source list.
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.intervention_class not in INTERVENTION_CLASSES:
            raise ValueError(f"unknown intervention_class {self.intervention_class!r}")
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
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dict")
        for art in self.artifacts:
            if not isinstance(art, str) or not art:
                raise ValueError("artifacts must be non-empty run-relative paths")
            parts = Path(art).parts
            if art[:1] in ("/", "\\") or ".." in parts or (
                    len(art) > 1 and art[1] == ":"):
                raise ValueError(f"artifact path escapes the run dir: {art!r}")

    # ------------------------------------------------------------------ json

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Proposal":
        p = Proposal(**_migrate(json.loads(text)))
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
