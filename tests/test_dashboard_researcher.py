"""Dashboard page 6 (Research Agenda) loaders on synthetic run fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viz.dashboard import (
    load_agenda_table,
    load_contradiction_events,
    load_executed_items,
    load_hypotheses_table,
)


def _agenda_proposal(pid: str, target: str, statement: str, score: float,
                     predicted_gain: float | None = None,
                     hyp_links: list[str] | None = None) -> dict:
    """A researcher-emitted experiment proposal as it lands in the queue
    (stage-C3): the agenda score/decomposition live in provenance."""
    return {
        "schema_version": 2, "id": pid, "type": "evaluation",
        "intervention_class": "experiment", "created_tick": 2000,
        "run_id": "run-a", "source": "researcher", "rationale": statement,
        "expected_benefit": {"metric": "researcher/predicted_gain",
                             "direction": "up",
                             "magnitude_estimate": predicted_gain or score},
        "confidence": 0.5, "supporting_observations": [],
        "estimated_cost": {"human_hours": 0.5, "gpu_hours": 2.0}, "risks": [],
        "success_criteria": {"metric": "researcher/predicted_gain",
                             "threshold": 0.0, "eval_window_ticks": 20000},
        "status": "pending", "target": target,
        "provenance": {"agenda_score": score,
                       "agenda_decomposition": {"value": 1.0,
                           "tractability": 1.0, "novelty": 1.0, "cost": 2.0},
                       "predicted_gain": predicted_gain,
                       "hypothesis_links": hyp_links or [],
                       "experiment": {"name": "directed_visit", "cost": 2.0}},
    }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run-a"
    rdir = run / "researcher"
    rdir.mkdir(parents=True)

    # The agenda is now the unified queue: researcher experiment proposals,
    # ranked by provenance.agenda_score (loader must sort, not assume order).
    pdir = run / "proposals"
    pdir.mkdir(parents=True)
    (pdir / "prop-agendalower001.json").write_text(json.dumps(
        _agenda_proposal("prop-agendalower001", "prop-abc", "lower the lr",
                         score=0.2)), encoding="utf-8")
    (pdir / "prop-agendatop00001.json").write_text(json.dumps(
        _agenda_proposal("prop-agendatop00001", "q-1",
                         "what are the dynamics of region (1, 1)?",
                         score=0.5, predicted_gain=0.32,
                         hyp_links=["hyp-region-1-1"])), encoding="utf-8")
    # A co-listed rule-generator proposal (no agenda_score) must be IGNORED
    # by the agenda loader — it belongs to the proposals page.
    (pdir / "prop-generator00001.json").write_text(json.dumps({
        "schema_version": 2, "id": "prop-generator00001",
        "type": "hyperparameter", "intervention_class": "config",
        "created_tick": 1, "run_id": "run-a", "source": "ledger",
        "rationale": "knob", "expected_benefit": {"metric": "m",
            "direction": "up", "magnitude_estimate": 0.1}, "confidence": 0.5,
        "supporting_observations": [], "estimated_cost": {"human_hours": 0,
            "gpu_hours": 0}, "risks": [], "success_criteria": {"metric": "m",
            "threshold": 0, "eval_window_ticks": 1}, "status": "pending",
        "provenance": {"generator_id": "propose_hyperparameter"}},
    ), encoding="utf-8")

    hdir = rdir / "hypotheses"
    hdir.mkdir()
    (hdir / "hyp-capability-success-2.json").write_text(json.dumps({
        "schema_version": 1, "id": "hyp-capability-success-2",
        "statement_template": "success rate of action {action} is stable",
        "params": {"action": 2}, "scope": "self_capability",
        "monitor": {}, "status": "contradicted", "last_checked": 5000,
        "transitions": [
            {"tick": 4750, "from": "supported", "to": "weakening"},
            {"tick": 5000, "from": "weakening", "to": "contradicted"}],
    }), encoding="utf-8")

    with open(rdir / "contradiction_events.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"hypothesis_id": "hyp-capability-success-2",
                            "tick": 4750, "from": "supported",
                            "to": "weakening", "statistic": 7.4,
                            "evidence": "mean shift 7.4 sd"}) + "\n")
        f.write(json.dumps({"hypothesis_id": "hyp-capability-success-2",
                            "tick": 5000, "from": "weakening",
                            "to": "contradicted", "statistic": 10.7,
                            "evidence": "mean shift 10.7 sd"}) + "\n")
    return run


def test_agenda_loader_reads_queue_ranked_by_score(run_dir: Path) -> None:
    table = load_agenda_table(run_dir)
    # Ranked by agenda_score (0.5 > 0.2); the generator proposal is excluded.
    assert list(table["ref"]) == ["q-1", "prop-abc"]
    assert list(table["rank"]) == [1, 2]
    assert table.iloc[0]["predicted_gain"] == pytest.approx(0.32)
    assert "hyp-region-1-1" in table.iloc[0]["hypothesis_links"]
    assert "knob" not in set(table["statement"])  # generator proposal excluded


def test_agenda_loader_empty_run(tmp_path: Path) -> None:
    assert load_agenda_table(tmp_path / "nothing").empty


def test_hypotheses_loader_renders_statement(run_dir: Path) -> None:
    table = load_hypotheses_table(run_dir)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["status"] == "contradicted"
    assert row["statement"] == "success rate of action 2 is stable"
    assert row["transitions"] == 2


def test_contradiction_events_loader(run_dir: Path) -> None:
    events = load_contradiction_events(run_dir)
    assert list(events["tick"]) == [4750, 5000]
    assert list(events["to"]) == ["weakening", "contradicted"]


def test_executed_items_loader(tmp_path: Path) -> None:
    csv = tmp_path / "items.csv"
    csv.write_text(
        "seed,arm,region,reduction,predicted_gain\n"
        "0,agenda,\"(1, 1)\",0.002,0.3\n"
        "0,random,\"(0, 0)\",0.001,\n",
        encoding="utf-8")
    items = load_executed_items(csv)
    assert len(items) == 2
    assert items[items["arm"] == "agenda"].iloc[0]["reduction"] == pytest.approx(0.002)
    assert load_executed_items(tmp_path / "missing.csv").empty
