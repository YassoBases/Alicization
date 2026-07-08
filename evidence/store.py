"""EvidenceStore: load one run dir's artifacts once, expose source-scoped
views and immutable content-hashed bundles.

Generalizes what proposals/evidence.py did (ledger vs logs_only variants of
the same run) into a shared, reusable store, and adds:
  - a repo snapshot reference {git_sha, dirty} (injected or read from
    runs/<id>/repo_snapshot.json — never captured by shelling out here),
  - code:<path>@<sha>#Lx-Ly citation refs,
  - bundle() -> an immutable, content-hashed EvidenceBundle whose hash is
    the provenance.evidence_bundle_hash a proposal records (schema v2).
"""

from __future__ import annotations

import hashlib
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


@dataclass(frozen=True)
class RepoSnapshot:
    """Which repo state produced this run. Injected or read from
    runs/<id>/repo_snapshot.json; unknown (None, None) when neither is
    available. Captured OUTSIDE this package (experiments/provenance.py) so
    the evidence plane never runs git."""

    git_sha: str | None
    dirty: bool | None

    @staticmethod
    def unknown() -> "RepoSnapshot":
        return RepoSnapshot(git_sha=None, dirty=None)

    def as_dict(self) -> dict[str, Any]:
        return {"git_sha": self.git_sha, "dirty": self.dirty}


@dataclass(frozen=True)
class EvidenceBundle:
    """An immutable snapshot of the evidence a proposal was derived from,
    identified by a content hash. The hash digests the source, run, tick,
    a per-tag summary of every scalar series, the competence tick, the
    raw-log sizes, a config digest, and the repo snapshot — enough that two
    materially different evidence states cannot collide, without hashing
    full float arrays."""

    source: str
    run_id: str
    tick: int
    content_hash: str
    repo_snapshot: RepoSnapshot
    tags: tuple[str, ...]


@dataclass
class EvidenceView:
    """Everything a generator/researcher may read from one run, scoped to a
    source variant. Ledger fields are None/empty in the logs_only variant.

    This is the pre-stage-C ``Evidence`` dataclass (kept as an alias): the
    generators' interface is unchanged. New: ``bundle_hash`` (the provenance
    stamp) and ``code_ref`` (source-line citations)."""

    source: str
    run_id: str
    tick: int
    scalars: dict[str, tuple[list[int], list[float]]]
    competence: CompetenceReport | None = None
    positions: np.ndarray | None = None      # (N, 2) world coords from JSONL
    actions: np.ndarray | None = None        # (N,) from JSONL
    config: dict[str, Any] = field(default_factory=dict)
    bundle_hash: str = ""                     # provenance.evidence_bundle_hash
    repo_snapshot: RepoSnapshot = field(default_factory=RepoSnapshot.unknown)

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

    def code_ref(self, path: str, line_start: int, line_end: int) -> str:
        """A source-line citation: code:<path>@<sha>#Lx-Ly. The sha is this
        run's repo snapshot; 'unknown' when it was never captured (degrade,
        don't lie). Validated against the working tree in stage-D."""
        sha = self.repo_snapshot.git_sha or "unknown"
        return f"code:{path}@{sha}#L{line_start}-L{line_end}"


# ---------------------------------------------------------------- readers


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


def _read_repo_snapshot(run_dir: Path) -> RepoSnapshot:
    path = run_dir / "repo_snapshot.json"
    if not path.exists():
        return RepoSnapshot.unknown()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RepoSnapshot(git_sha=raw.get("git_sha"), dirty=raw.get("dirty"))


# ------------------------------------------------------------------ store


class EvidenceStore:
    """One run dir's artifacts, loaded once, viewable per source. Read-only.

    ``repo_snapshot`` may be injected (the harness that knows the git state);
    otherwise it is read from runs/<id>/repo_snapshot.json, else unknown.
    """

    def __init__(self, run_dir: str | Path,
                 repo_snapshot: RepoSnapshot | None = None) -> None:
        self.run_dir = Path(run_dir)
        self._scalars = _read_tb_scalars(self.run_dir)
        self._positions, self._actions = _read_jsonl(self.run_dir)
        self._competence = _read_latest_competence(self.run_dir)
        cfg_path = self.run_dir / "config.json"
        self._config: dict[str, Any] = (
            json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg_path.exists() else {})
        self.repo_snapshot = (repo_snapshot if repo_snapshot is not None
                              else _read_repo_snapshot(self.run_dir))
        self._tick = max((sv[0][-1] for sv in self._scalars.values() if sv[0]),
                         default=int(self._actions.shape[0]) - 1
                         if self._actions.size else 0)

    # ------------------------------------------------------------- scoping

    def _scoped_scalars(self, source: str) -> dict[str, tuple[list[int], list[float]]]:
        if source == "logs_only":
            return {tag: sv for tag, sv in self._scalars.items()
                    if not tag.startswith(LEDGER_TAG_PREFIXES)}
        return dict(self._scalars)

    def view(self, source: str) -> EvidenceView:
        """A source-scoped view over the store; logs_only strips every
        Ledger-derived signal (scalars + competence)."""
        scalars = self._scoped_scalars(source)
        competence = None if source == "logs_only" else self._competence
        return EvidenceView(
            source=source, run_id=self.run_dir.name, tick=self._tick,
            scalars=scalars, competence=competence,
            positions=self._positions if self._positions.size else None,
            actions=self._actions if self._actions.size else None,
            config=self._config, bundle_hash=self._content_hash(source),
            repo_snapshot=self.repo_snapshot,
        )

    # ------------------------------------------------------------- bundling

    def _manifest(self, source: str) -> dict[str, Any]:
        """Canonical, float-noise-tolerant summary of the scoped evidence.
        Per tag: (n, first step, last step, rounded sum, rounded last) — a
        material change moves the digest; a re-read of the same logs does
        not."""
        scalars = self._scoped_scalars(source)
        tag_summary = {}
        for tag in sorted(scalars):
            steps, values = scalars[tag]
            arr = np.asarray(values, dtype=float)
            tag_summary[tag] = [
                len(values),
                steps[0] if steps else None,
                steps[-1] if steps else None,
                round(float(arr.sum()), 6) if arr.size else 0.0,
                round(float(arr[-1]), 6) if arr.size else 0.0,
            ]
        competence = None if source == "logs_only" else self._competence
        return {
            "source": source, "run_id": self.run_dir.name, "tick": self._tick,
            "tags": tag_summary,
            "competence_tick": competence.tick if competence else None,
            "n_positions": int(self._positions.shape[0]) if self._positions.size else 0,
            "n_actions": int(self._actions.shape[0]) if self._actions.size else 0,
            "config_digest": hashlib.sha256(
                json.dumps(self._config, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "repo_snapshot": self.repo_snapshot.as_dict(),
        }

    def _content_hash(self, source: str) -> str:
        blob = json.dumps(self._manifest(source), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def bundle(self, source: str) -> EvidenceBundle:
        """An immutable, content-hashed snapshot of the scoped evidence."""
        return EvidenceBundle(
            source=source, run_id=self.run_dir.name, tick=self._tick,
            content_hash=self._content_hash(source),
            repo_snapshot=self.repo_snapshot,
            tags=tuple(sorted(self._scoped_scalars(source))),
        )


def evidence_from_run(run_dir: str | Path, source: str,
                      repo_snapshot: RepoSnapshot | None = None) -> EvidenceView:
    """Compatibility entry point (pre-stage-C signature): build one
    source-scoped view from a run dir."""
    return EvidenceStore(run_dir, repo_snapshot=repo_snapshot).view(source)
