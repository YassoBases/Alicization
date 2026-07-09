"""Self-critique pass (stage-D4): attack each drafted proposal before it can
be emitted.

Two layers:
- DETERMINISTIC citation validation (offline, always on): every
  supporting_observation ref must resolve against the run's evidence (a real
  tb tag/step, a present competence report, or an in-range code:<path>@sha
  line span). A proposal with no refs, or any ref that does not resolve, is
  DISCARDED — not emitted. This is the hard gate.
- ADVERSARIAL LLM review (online only): a second call attacks whether the
  cited records actually show what the rationale claims, whether the success
  criterion is entailed by the change itself (the free_nats tautology class),
  and whether the cost is realistic — producing a critique record attached
  to the proposal and a revised confidence.

The tautology rule is duplicated from experiments/runner.py (a 3-line rule)
because architect/ may not import experiments/; both cite the same
proposal_quality ANALYSIS caveat.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from architect.draft import LLMClient
from evidence import EvidenceView
from proposals.schema import Proposal

_TB_RE = re.compile(r"^tb:(?P<tag>[^@]+)(?:@step=(?P<step>\d+))?$")
_CODE_RE = re.compile(r"^code:(?P<path>[^@]+)@(?P<sha>[^#]+)#L(?P<a>\d+)-L(?P<b>\d+)$")
_COMPETENCE_RE = re.compile(r"^competence:report-(?P<tick>\d+)")


def resolve_citation(ref: str, view: EvidenceView, repo_root: Path) -> bool:
    """True iff the ref points at a record that actually exists."""
    m = _TB_RE.match(ref)
    if m:
        tag = m.group("tag")
        steps_values = view.scalars.get(tag)
        if not steps_values or not steps_values[0]:
            return False
        if m.group("step") is not None:
            return int(m.group("step")) in steps_values[0]
        return True
    m = _COMPETENCE_RE.match(ref)
    if m:
        return view.competence is not None
    m = _CODE_RE.match(ref)
    if m:
        path = repo_root / m.group("path")
        if not path.exists():
            return False
        loc = path.read_text(encoding="utf-8").count("\n") + 1
        a, b = int(m.group("a")), int(m.group("b"))
        return 1 <= a <= b <= loc
    return False


def validate_citations(proposal: Proposal, view: EvidenceView,
                       repo_root: Path) -> tuple[bool, list[str]]:
    """(ok, unresolved). A proposal must cite at least one ref and every ref
    must resolve."""
    obs = proposal.supporting_observations
    unresolved = [r for r in obs if not resolve_citation(r, view, repo_root)]
    return (bool(obs) and not unresolved), unresolved


def is_tautological(proposal: Proposal) -> str | None:
    change = proposal.proposed_change or {}
    metric = proposal.success_criteria.get("metric", "")
    if change.get("config_path") == "rssm.free_nats" and metric in ("rssm/kl", "sleep/kl"):
        return ("success criterion is the KL the free_nats knob directly "
                "clamps — near-tautological (proposal_quality ANALYSIS caveat)")
    return None


_CRITIQUE_SYSTEM = (
    "You are an adversarial reviewer of a single experiment proposal. Attack "
    "it: do the cited records actually show what the rationale claims? Is the "
    "success criterion trivially entailed by the change itself? Is the cost "
    "realistic? Reply ONLY with a JSON object "
    "{\"revised_confidence\": 0..1, \"critique\": \"...\"}."
)


def _adversarial_review(proposal: Proposal, view: EvidenceView,
                        client: LLMClient, cfg: dict[str, Any]) -> dict[str, Any]:
    acfg = cfg.get("architect", {})
    prompt = (f"Proposal: {proposal.rationale}\n"
              f"target={proposal.target} class={proposal.intervention_class}\n"
              f"criterion={json.dumps(proposal.success_criteria)}\n"
              f"cost={json.dumps(proposal.estimated_cost)}\n"
              f"cites={json.dumps(proposal.supporting_observations)}")
    raw = client.complete(system=_CRITIQUE_SYSTEM, prompt=prompt,
                          model=acfg.get("model_id", "claude-sonnet-5"),
                          temperature=0.0,
                          max_tokens=int(acfg.get("max_tokens", 4096)))
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        rc = data.get("revised_confidence")
        return {"critique": str(data.get("critique", "")),
                "revised_confidence": (float(rc) if rc is not None else None)}
    except (ValueError, KeyError):
        return {"critique": "adversarial review returned malformed JSON",
                "revised_confidence": None}


def critique_proposals(proposals: list[Proposal], view: EvidenceView,
                       repo_root: str | Path, cfg: dict[str, Any],
                       client: LLMClient | None = None
                       ) -> tuple[list[Proposal], list[dict[str, Any]]]:
    """Discard proposals whose citations do not resolve; attach a critique
    record (+ revised confidence when online) to the survivors."""
    repo_root = Path(repo_root)
    offline = cfg.get("architect", {}).get("offline", True)
    kept: list[Proposal] = []
    decisions: list[dict[str, Any]] = []
    for p in proposals:
        ok, unresolved = validate_citations(p, view, repo_root)
        if not ok:
            decisions.append({"action": "discard", "proposal_id": p.id,
                "reason": f"citations do not resolve: {unresolved or 'no refs'}"})
            continue
        record: dict[str, Any] = {"citation_ok": True,
                                  "tautological_criterion": is_tautological(p)}
        if client is not None and not offline:
            review = _adversarial_review(p, view, client, cfg)
            record["adversarial"] = review["critique"]
            if review["revised_confidence"] is not None:
                p.confidence = float(np.clip(review["revised_confidence"], 0.0, 1.0))
        p.provenance = {**p.provenance, "critique": record}
        kept.append(p)
        decisions.append({"action": "keep", "proposal_id": p.id,
                          "tautological": bool(record["tautological_criterion"])})
    return kept, decisions
