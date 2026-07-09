"""Stage-D3: LLM drafting with an INJECTED stub client (offline). Covers the
offline kill switch, strict-JSON parsing + one repair round-trip, schema-v2
construction with prompt-hash provenance and a diff artifact, and
constitutional rejection of drafted proposals."""

from __future__ import annotations

import json
from pathlib import Path

from architect.analysis import analyze
from architect.draft import StubClient, draft_proposals
from evidence import EvidenceView, RepoSnapshot

ROOT = Path(__file__).resolve().parent.parent


def _view() -> EvidenceView:
    return EvidenceView(
        source="ledger", run_id="r", tick=100,
        scalars={"reward/rollout": ([0, 1], [0.1, 0.2]),
                 "rssm/kl": ([0], [1.0])},
        config={"rssm": {"free_nats": 1.0}}, bundle_hash="bundlehash000001",
        repo_snapshot=RepoSnapshot("abc1234", False))


def _valid_json(target: str = "rssm.free_nats", diff: str | None = None) -> str:
    obj = {
        "type": "hyperparameter", "intervention_class": "config",
        "target": target, "rationale": "KL sits at the free-nats floor",
        "supporting_observations": ["tb:rssm/kl@step=0"],
        "expected_benefit": {"metric": "rssm/recon", "direction": "down",
                             "magnitude_estimate": 0.05},
        "success_criteria": {"metric": "rssm/kl", "threshold": 0.5,
                             "eval_window_ticks": 10000},
        "estimated_cost": {"human_hours": 0.25, "gpu_hours": 1.0},
        "risks": ["can destabilize early training"], "confidence": 0.55,
        "proposed_change": {"config_path": "rssm.free_nats", "new_value": 0.5},
    }
    if diff is not None:
        obj["diff"] = diff
    return "```json\n" + json.dumps([obj]) + "\n```"


ONLINE_CFG = {"architect": {"offline": False, "model_id": "claude-test"}}


def test_offline_is_a_logged_noop(tmp_path: Path) -> None:
    report = analyze(ROOT)
    result = draft_proposals(report, _view(), tmp_path / "runs" / "r",
                             {"architect": {"offline": True}})
    assert result.proposals == []
    assert result.decisions[0]["action"] == "skip"


def test_valid_output_becomes_v2_proposal_with_provenance(tmp_path: Path) -> None:
    diff = "--- a/configs/base.yaml\n+++ b/configs/base.yaml\n@@ -1 +1 @@\n-a\n+b\n"
    client = StubClient([_valid_json(diff=diff)])
    run_dir = tmp_path / "runs" / "r"
    result = draft_proposals(analyze(ROOT), _view(), run_dir, ONLINE_CFG,
                             client=client)
    assert len(result.proposals) == 1
    p = result.proposals[0]
    assert p.schema_version == 2 and p.intervention_class == "config"
    assert p.source == "architect:claude-test"
    assert p.provenance["prompt_hash"] == result.prompt_hash
    assert p.provenance["model_id"] == "claude-test"
    assert p.provenance["evidence_bundle_hash"] == "bundlehash000001"
    # The diff was written as an artifact under runs/<id>/architect/.
    assert p.artifacts == [f"architect/diffs/{p.id}.diff"]
    assert (run_dir / p.artifacts[0]).exists()
    # The client saw the system prompt + a prompt citing evidence.
    assert "JSON array" in client.calls[0]["system"]
    assert "rssm/kl" in client.calls[0]["prompt"]


def test_malformed_json_gets_one_repair(tmp_path: Path) -> None:
    client = StubClient(["I think you should lower the learning rate.",
                         _valid_json()])
    result = draft_proposals(analyze(ROOT), _view(), tmp_path / "runs" / "r",
                             ONLINE_CFG, client=client)
    assert len(result.proposals) == 1
    assert any(d["action"] == "repair" for d in result.decisions)
    assert len(client.calls) == 2  # original + one repair


def test_malformed_twice_discards_batch(tmp_path: Path) -> None:
    client = StubClient(["not json", "still not json"])
    result = draft_proposals(analyze(ROOT), _view(), tmp_path / "runs" / "r",
                             ONLINE_CFG, client=client)
    assert result.proposals == []
    assert result.decisions[0]["action"] == "discard_batch"
    assert result.prompt_hash  # still recorded for reproducibility


def test_constitutional_target_is_discarded(tmp_path: Path) -> None:
    client = StubClient([_valid_json(target="review/queue.py")])
    result = draft_proposals(analyze(ROOT), _view(), tmp_path / "runs" / "r",
                             ONLINE_CFG, client=client)
    assert result.proposals == []
    assert any(d["action"] == "discard" and "constitution" in d["reason"]
               for d in result.decisions)


def test_constitutional_diff_is_discarded(tmp_path: Path) -> None:
    bad = "--- a/CLAUDE.md\n+++ b/CLAUDE.md\n@@ -1 +1 @@\n-a\n+b\n"
    client = StubClient([_valid_json(diff=bad)])
    result = draft_proposals(analyze(ROOT), _view(), tmp_path / "runs" / "r",
                             ONLINE_CFG, client=client)
    assert result.proposals == []
    assert any("constitution" in d["reason"] for d in result.decisions
               if d["action"] == "discard")
