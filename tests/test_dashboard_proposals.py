"""Proposals-page data loaders on a fixture runs tree: blind masking in the
table, acceptance-rate series, time-to-first-useful, confidence calibration,
and the repeated-after-denial statistic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proposals.schema import Proposal, save_proposal
from review.queue import ReviewQueue
from viz.dashboard import (
    acceptance_rate_over_time,
    confidence_calibration,
    load_decisions,
    load_proposals_table,
    repeated_after_denial,
    time_to_first_useful,
)


def _proposal(run_id: str, ptype: str = "hyperparameter", tick: int = 100,
              confidence: float = 0.5, target: str = "knob") -> Proposal:
    return Proposal.new(
        type=ptype, created_tick=tick, run_id=run_id, source="ledger",
        rationale="fixture", expected_benefit={"metric": "reward/rollout",
                                               "direction": "up",
                                               "magnitude_estimate": 0.1},
        confidence=confidence, supporting_observations=["tb:reward/rollout"],
        estimated_cost={"human_hours": 0.1, "gpu_hours": 0.1}, risks=[],
        success_criteria={"metric": "reward/rollout", "threshold": 0.0,
                          "eval_window_ticks": 100},
        target=target,
    )


@pytest.fixture()
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    run = root / "run-a"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")  # marks it as a run dir

    # pending (blinded), evaluated hit, evaluated miss, rejected.
    save_proposal(_proposal("run-a", tick=50, target="pending-one"), run)

    hit = _proposal("run-a", tick=200, confidence=0.9, target="hit-one")
    hit.status = "evaluated"
    hit.realized_benefit = {"metric": "reward/rollout", "observed": 1.0,
                            "threshold": 0.0, "direction": "up",
                            "met_success_criteria": True}
    save_proposal(hit, run)

    miss = _proposal("run-a", tick=300, confidence=0.9, target="miss-one")
    miss.status = "evaluated"
    miss.realized_benefit = {"metric": "reward/rollout", "observed": -1.0,
                             "threshold": 0.0, "direction": "up",
                             "met_success_criteria": False}
    save_proposal(miss, run)

    denied = _proposal("run-a", tick=400, target="denied-knob")
    save_proposal(denied, run)
    queue = ReviewQueue(run)
    queue.decide(denied.id, "reject", note="not now")
    queue.decide(hit.id, "approve", note="ok")
    # Generator re-recommends the denied target afterwards (suppressed dup).
    gd = run / "proposals" / "generator_decisions.jsonl"
    with open(gd, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tick": 999, "generator": "propose_hyperparameter",
            "source": "ledger", "decision": "SUPPRESSED",
            "reason": "duplicate of existing (type=hyperparameter, "
                      "target=denied-knob)", "proposal_id": None,
            "timestamp": 9e12,  # strictly after the rejection
        }) + "\n")
    # Clean the approval ticket the fixture created.
    ticket = Path("experiments/tickets") / f"{hit.id}.md"
    if ticket.exists():
        ticket.unlink()
    return root


def test_table_blinds_unevaluated_sources(runs_root: Path) -> None:
    table = load_proposals_table(runs_root)
    assert len(table) == 4
    blinded = table[table["status"] != "evaluated"]["source"]
    assert (blinded == "<blinded>").all()
    unblinded = table[table["status"] == "evaluated"]["source"]
    assert (unblinded == "ledger").all()


def test_acceptance_rate_and_time_to_first_useful(runs_root: Path) -> None:
    decisions = load_decisions(runs_root)
    rate = acceptance_rate_over_time(decisions)
    # reject then approve -> rates [0.0, 0.5].
    assert rate["acceptance_rate"].tolist() == [0.0, 0.5]
    table = load_proposals_table(runs_root)
    assert time_to_first_useful(table) == 200  # the evaluated hit


def test_confidence_calibration_bins(runs_root: Path) -> None:
    calib = confidence_calibration(load_proposals_table(runs_root))
    # Both evaluated proposals sit in the 0.8-1.0 bin: one hit, one miss.
    assert len(calib) == 1
    row = calib.iloc[0]
    assert row["count"] == 2 and row["hit_rate"] == pytest.approx(0.5)


def test_repeated_after_denial_counts_resurrections(runs_root: Path) -> None:
    rad = repeated_after_denial(runs_root)
    assert len(rad) == 1
    row = rad.iloc[0]
    assert row["target"] == "denied-knob"
    assert row["rejections"] == 1
    assert row["recommendations_after_denial"] == 1


def test_loaders_tolerate_empty_tree(tmp_path: Path) -> None:
    empty = tmp_path / "no-runs"
    assert load_proposals_table(empty).empty
    assert load_decisions(empty).empty
    assert np.isnan(time_to_first_useful(load_proposals_table(empty)))
    assert repeated_after_denial(empty).empty
