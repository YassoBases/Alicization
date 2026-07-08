"""The evidence plane (stage-C): one read-only store over a run directory,
source-scoped views, and immutable content-hashed bundles.

STRUCTURAL RULES (same as proposals/ and researcher/, enforced by
tests/test_proposals_no_execution.py, which scans this package):

- Data, never code. Nothing here executes anything: no subprocess, exec,
  eval, importlib, os.system. The repo snapshot's git_sha/dirty are
  INJECTED or read from runs/<id>/repo_snapshot.json — this package never
  shells out (capture lives in experiments/provenance.py, outside the
  trust boundary).
- Imports: json/hashlib/pathlib (stdlib), numpy, the tensorboard event
  reader, and ledger.competence only. No world/training/agent/memory/torch.
- Reads only; the store never writes.

proposals/ and researcher/ consume this package; proposals/evidence.py is
a thin compatibility shim re-exporting EvidenceView (as Evidence) and
evidence_from_run.
"""

from __future__ import annotations

from evidence.store import (
    LEDGER_TAG_PREFIXES,
    EvidenceBundle,
    EvidenceStore,
    EvidenceView,
    RepoSnapshot,
    evidence_from_run,
)

# Back-compat alias: the pre-stage-C name for a source-scoped view.
Evidence = EvidenceView

__all__ = [
    "LEDGER_TAG_PREFIXES",
    "Evidence",
    "EvidenceBundle",
    "EvidenceStore",
    "EvidenceView",
    "RepoSnapshot",
    "evidence_from_run",
]
