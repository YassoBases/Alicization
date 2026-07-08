"""DEPRECATED compatibility shim (stage-C).

The evidence plane moved to the top-level ``evidence`` package
(evidence/store.py): one EvidenceStore per run dir, source-scoped views,
and content-hashed bundles. This module now re-exports the pieces callers
used before the move so existing imports keep working:

    from proposals.evidence import Evidence, evidence_from_run

Prefer importing from ``evidence`` directly in new code:

    from evidence import EvidenceStore, EvidenceView, evidence_from_run

``Evidence`` is an alias for ``evidence.EvidenceView``. The logs_only strip
list (LEDGER_TAG_PREFIXES) lives in evidence/store.py. This shim will be
removed once no caller imports it.
"""

from __future__ import annotations

from evidence import (
    LEDGER_TAG_PREFIXES,
    Evidence,
    EvidenceView,
    evidence_from_run,
)

__all__ = ["LEDGER_TAG_PREFIXES", "Evidence", "EvidenceView", "evidence_from_run"]
