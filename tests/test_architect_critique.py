"""Stage-D4: the self-critique gate. Deterministic citation validation
(offline) discards proposals whose refs don't resolve; the adversarial LLM
review (injected client) attaches a critique and can revise confidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from architect.critique import (
    critique_proposals,
    is_tautological,
    resolve_citation,
    validate_citations,
)
from architect.draft import StubClient
from evidence import EvidenceView, RepoSnapshot
from proposals.schema import Proposal

ROOT = Path(__file__).resolve().parent.parent


def _view() -> EvidenceView:
    return EvidenceView(
        source="ledger", run_id="r", tick=100,
        scalars={"reward/rollout": ([0, 1, 2], [0.1, 0.2, 0.3]),
                 "rssm/kl": ([0], [1.0])},
        config={}, bundle_hash="b0",
        repo_snapshot=RepoSnapshot("abc1234", False))


def _proposal(obs: list[str], *, metric: str = "rssm/recon",
              change: dict | None = None) -> Proposal:
    return Proposal.new(
        type="hyperparameter", created_tick=100, run_id="r",
        source="architect:test", intervention_class="config",
        rationale="drafted", expected_benefit={"metric": "rssm/recon",
            "direction": "down", "magnitude_estimate": 0.05}, confidence=0.6,
        supporting_observations=obs,
        estimated_cost={"human_hours": 0.25, "gpu_hours": 1.0}, risks=[],
        success_criteria={"metric": metric, "threshold": 0.5,
                          "eval_window_ticks": 10000},
        target="rssm.free_nats", proposed_change=change,
        provenance={"generator_id": "architect"})


# ------------------------------------------------------ citation resolution


def test_resolve_tb_competence_and_code_refs() -> None:
    view = _view()
    assert resolve_citation("tb:reward/rollout", view, ROOT)
    assert resolve_citation("tb:reward/rollout@step=2", view, ROOT)
    assert not resolve_citation("tb:reward/rollout@step=99", view, ROOT)  # no such step
    assert not resolve_citation("tb:no/such/tag", view, ROOT)
    # code ref in range vs out of range vs missing file.
    assert resolve_citation("code:architect/constitution.py@abc#L1-L5", view, ROOT)
    assert not resolve_citation("code:architect/constitution.py@abc#L1-L999999", view, ROOT)
    assert not resolve_citation("code:does/not/exist.py@abc#L1-L2", view, ROOT)
    # competence ref needs a report present.
    assert not resolve_citation("competence:report-100", view, ROOT)
    view.competence = object()  # type: ignore[assignment]
    assert resolve_citation("competence:report-100", view, ROOT)
    assert not resolve_citation("mystery:ref", view, ROOT)


def test_validate_citations_requires_at_least_one_resolving_ref() -> None:
    view = _view()
    ok, unresolved = validate_citations(_proposal(["tb:reward/rollout"]), view, ROOT)
    assert ok and unresolved == []
    ok, unresolved = validate_citations(_proposal([]), view, ROOT)  # no refs
    assert not ok
    ok, unresolved = validate_citations(
        _proposal(["tb:reward/rollout", "tb:ghost/tag"]), view, ROOT)
    assert not ok and unresolved == ["tb:ghost/tag"]


def test_tautology_rule_matches_runner() -> None:
    assert is_tautological(_proposal(["tb:rssm/kl"], metric="rssm/kl",
        change={"config_path": "rssm.free_nats", "new_value": 0.5})) is not None
    assert is_tautological(_proposal(["tb:rssm/kl"], metric="rssm/recon",
        change={"config_path": "rssm.free_nats", "new_value": 0.5})) is None


# ------------------------------------------------------------ the gate


def test_critique_discards_unresolved_and_keeps_valid_offline() -> None:
    view = _view()
    good = _proposal(["tb:reward/rollout"])
    bad = _proposal(["tb:ghost/tag"])
    kept, decisions = critique_proposals([good, bad], view, ROOT,
                                         {"architect": {"offline": True}})
    assert [p.id for p in kept] == [good.id]
    assert kept[0].provenance["critique"]["citation_ok"] is True
    assert any(d["action"] == "discard" and d["proposal_id"] == bad.id
               for d in decisions)


def test_critique_flags_tautology_offline() -> None:
    view = _view()
    taut = _proposal(["tb:rssm/kl"], metric="rssm/kl",
                     change={"config_path": "rssm.free_nats", "new_value": 0.5})
    kept, _ = critique_proposals([taut], view, ROOT,
                                 {"architect": {"offline": True}})
    assert kept[0].provenance["critique"]["tautological_criterion"] is not None


def test_adversarial_review_revises_confidence_online() -> None:
    view = _view()
    client = StubClient([json.dumps(
        {"revised_confidence": 0.2, "critique": "the cited tag is flat"})])
    p = _proposal(["tb:reward/rollout"])
    kept, decisions = critique_proposals(
        [p], view, ROOT, {"architect": {"offline": False, "model_id": "m"}},
        client=client)
    assert kept[0].confidence == pytest.approx(0.2)
    assert "flat" in kept[0].provenance["critique"]["adversarial"]
    assert decisions[0]["action"] == "keep"
