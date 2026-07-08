"""Review state machine: legal/illegal status transitions and
blind-until-evaluated (Section 18 delta)."""

from __future__ import annotations

from pathlib import Path

import pytest

from proposals.schema import Proposal, save_proposal
from review.queue import ReviewQueue, blind_view


def _proposal(run_dir: Path, status: str = "pending") -> Proposal:
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id=run_dir.name,
        source="logs_only", rationale="fixture",
        expected_benefit={"metric": "m", "direction": "up",
                          "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0,
                          "eval_window_ticks": 10},
    )
    p.status = status
    save_proposal(p, run_dir)
    return p


LEGAL = [
    ("pending", "approve", "approved"),
    ("pending", "reject", "rejected"),
    ("pending", "postpone", "postponed"),
    ("pending", "partial", "partially_approved"),
    ("postponed", "approve", "approved"),
    ("modified", "approve", "approved"),
    ("modified", "reject", "rejected"),
]

ILLEGAL = [
    ("approved", "approve"),   # already decided; evaluation is the only exit
    ("approved", "reject"),    # no flip-flopping an emitted ticket
    ("rejected", "approve"),   # terminal: re-propose, don't re-decide
    ("evaluated", "reject"),   # terminal
    ("evaluated", "approve"),
    ("partially_approved", "postpone"),
]


@pytest.mark.parametrize("start,action,end", LEGAL)
def test_legal_transitions(tmp_path: Path, start: str, action: str, end: str) -> None:
    run_dir = tmp_path / "run"
    p = _proposal(run_dir, status=start)
    queue = ReviewQueue(run_dir)
    result = queue.decide(p.id, action)
    assert result.status == end
    ticket = Path("experiments/tickets") / f"{p.id}.md"
    if end in ("approved", "partially_approved"):
        assert ticket.exists()
        ticket.unlink()


@pytest.mark.parametrize("start,action", ILLEGAL)
def test_illegal_transitions_raise(tmp_path: Path, start: str, action: str) -> None:
    run_dir = tmp_path / "run"
    p = _proposal(run_dir, status=start)
    queue = ReviewQueue(run_dir)
    with pytest.raises(ValueError, match="illegal transition"):
        queue.decide(p.id, action)
    assert queue.get(p.id).status == start  # unchanged on refusal


def test_modify_only_from_open_states(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    p = _proposal(run_dir, status="approved")
    with pytest.raises(ValueError, match="illegal transition"):
        ReviewQueue(run_dir).modify_start(p.id)


def test_blind_until_evaluated_all_states(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for status in ("pending", "approved", "rejected", "postponed", "modified"):
        p = _proposal(run_dir, status=status)
        assert blind_view(p)["source"] == "<blinded until evaluated>", status
    p = _proposal(run_dir, status="evaluated")
    assert blind_view(p)["source"] == "logs_only"


def test_blinding_keys_off_status_not_source_enum(tmp_path: Path) -> None:
    """v2: source is an open string. Blinding must still hide it while
    unevaluated (keying off status), and reveal the exact string — even a
    novel one — once evaluated."""
    run_dir = tmp_path / "run"
    p = _proposal(run_dir, status="pending")
    p.source = "architect:sonnet"       # not in the known-sources list
    save_proposal(p, run_dir)
    assert blind_view(ReviewQueue(run_dir).get(p.id))["source"] == \
        "<blinded until evaluated>"
    p.status = "evaluated"
    save_proposal(p, run_dir)
    assert blind_view(ReviewQueue(run_dir).get(p.id))["source"] == "architect:sonnet"
