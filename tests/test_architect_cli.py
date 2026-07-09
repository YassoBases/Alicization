"""Stage-D5: the `python -m architect` pipeline. Offline end-to-end on the
live repo (the Gate-D path), and an online emit via an injected stub client.
No network, no API key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from architect.__main__ import main, run_architect
from architect.draft import StubClient

ROOT = Path(__file__).resolve().parent.parent


def _run_dir(tmp_path: Path, offline: bool = True) -> Path:
    run_dir = tmp_path / "runs" / "20990101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"architect": {"offline": offline, "model_id": "m"}}),
        encoding="utf-8")
    (run_dir / "repo_snapshot.json").write_text(
        json.dumps({"git_sha": "abc1234", "dirty": False}), encoding="utf-8")
    return run_dir


def _valid_json() -> str:
    return json.dumps([{
        "type": "hyperparameter", "intervention_class": "config",
        "target": "rssm.free_nats", "rationale": "KL at the free-nats floor",
        "supporting_observations": ["tb:rssm/kl@step=0"],
        "expected_benefit": {"metric": "rssm/recon", "direction": "down",
                             "magnitude_estimate": 0.05},
        "success_criteria": {"metric": "rssm/recon", "threshold": 0.5,
                             "eval_window_ticks": 10000},
        "estimated_cost": {"human_hours": 0.25, "gpu_hours": 1.0},
        "risks": ["destabilize"], "confidence": 0.55,
        "proposed_change": {"config_path": "rssm.free_nats", "new_value": 0.5}}])


def test_offline_end_to_end_on_live_repo(tmp_path: Path,
                                         capsys: pytest.CaptureFixture) -> None:
    run_dir = _run_dir(tmp_path, offline=True)
    rc = main(["--run", str(run_dir), "--offline", "--repo-root", str(ROOT)])
    assert rc == 0
    # analysis.json written, valid JSON, names real modules.
    analysis = json.loads((run_dir / "architect" / "analysis.json").read_text())
    assert analysis["modules"] and "invariants" in analysis
    # decisions.jsonl records the offline skip; nothing emitted.
    decisions = [json.loads(x) for x in
                 (run_dir / "architect" / "decisions.jsonl").read_text().splitlines()]
    assert any(d["action"] == "skip" for d in decisions)
    assert not list((run_dir / "proposals").glob("prop-*.json")) \
        if (run_dir / "proposals").exists() else True
    assert "offline" in capsys.readouterr().out


def test_online_emits_into_queue_via_injected_client(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path, offline=False)
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {"rssm/kl": ([0], [1.0]),
                                    "reward/rollout": ([0, 1], [0.1, 0.2])})
    cfg = {"architect": {"offline": False, "model_id": "m"}}
    emitted = run_architect(run_dir, ROOT, cfg,
                            client=StubClient([_valid_json()]))
    assert len(emitted) == 1
    p = emitted[0]
    assert p.source == "architect:m" and p.intervention_class == "config"
    # Emitted into the standard queue.
    from proposals.schema import load_all
    assert [x.id for x in load_all(run_dir)] == [p.id]
    # Provenance carries the reproducibility stamp; critique attached.
    assert p.provenance["prompt_hash"] and p.provenance["model_id"] == "m"
    assert p.provenance["critique"]["citation_ok"] is True
    # decisions.jsonl records draft -> keep -> emit with timestamps.
    decisions = [json.loads(x) for x in
                 (run_dir / "architect" / "decisions.jsonl").read_text().splitlines()]
    actions = {d["action"] for d in decisions}
    assert {"draft", "keep", "emit"} <= actions
    assert all("timestamp" in d for d in decisions)


def test_unresolved_citation_emits_nothing(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path, offline=False)
    # No rssm/kl in the store: the drafted proposal's citation cannot resolve.
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {"reward/rollout": ([0, 1], [0.1, 0.2])})
    emitted = run_architect(run_dir, ROOT, {"architect": {"offline": False}},
                            client=StubClient([_valid_json()]))
    assert emitted == []
    decisions = [json.loads(x) for x in
                 (run_dir / "architect" / "decisions.jsonl").read_text().splitlines()]
    assert any(d["action"] == "discard" and "citations" in d["reason"]
               for d in decisions)
