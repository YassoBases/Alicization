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


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run-a"
    rdir = run / "researcher"
    rdir.mkdir(parents=True)

    # Two agendas: the loader must pick the LATEST.
    (rdir / "agenda_000000001000.json").write_text(json.dumps([
        {"id": "agenda-old", "kind": "question", "ref": "q-old",
         "statement": "stale", "experiment": {}, "score": 0.1,
         "decomposition": {}}]), encoding="utf-8")
    (rdir / "agenda_000000002000.json").write_text(json.dumps([
        {"id": "agenda-q-1", "kind": "question", "ref": "q-1",
         "statement": "what are the dynamics of region (1, 1)?",
         "experiment": {"name": "directed_visit", "cost": 2.0},
         "score": 0.5,
         "decomposition": {"value": 1.0, "tractability": 1.0,
                           "novelty": 1.0, "cost": 2.0},
         "hypothesis_links": ["hyp-region-1-1"], "predicted_gain": 0.32},
        {"id": "agenda-prop-1", "kind": "proposal", "ref": "prop-abc",
         "statement": "lower the lr", "experiment": {"name": "proposal_ticket"},
         "score": 0.2, "decomposition": {"value": 0.4, "tractability": 0.5,
                                         "novelty": 1.0, "cost": 1.0}},
    ]), encoding="utf-8")

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


def test_agenda_loader_picks_latest_and_keeps_order(run_dir: Path) -> None:
    table = load_agenda_table(run_dir)
    assert list(table["ref"]) == ["q-1", "prop-abc"]  # agenda order, not score-resorted
    assert list(table["rank"]) == [1, 2]
    assert table.iloc[0]["predicted_gain"] == pytest.approx(0.32)
    assert "hyp-region-1-1" in table.iloc[0]["hypothesis_links"]
    assert "stale" not in set(table["statement"])  # older agenda ignored


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
