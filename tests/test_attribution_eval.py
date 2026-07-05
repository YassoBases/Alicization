"""Unit tests for training/attribution_eval.py: ground-truth cause labeling
from synthetic event lists, and the accuracy tracker. This module is
deliberately outside ledger/ (evaluation-only; never used in any loss) — see
the module docstring."""

from __future__ import annotations

from ledger.attribution import BOTH, SELF, WORLD
from training.attribution_eval import AttributionAccuracyTracker, ground_truth_label
from world.engine import EAT, MOVE_E, NOOP


def test_ground_truth_noop_is_always_world() -> None:
    assert ground_truth_label(NOOP, []) == WORLD
    assert ground_truth_label(NOOP, [{"type": "agent_moved", "cause": "self"}]) == WORLD


def test_ground_truth_self_only_event() -> None:
    events = [{"type": "agent_moved", "cause": "self"}]
    assert ground_truth_label(MOVE_E, events) == SELF


def test_ground_truth_no_relevant_events_defaults_self() -> None:
    # e.g. a failed/blocked move: no agent_moved event logged at all.
    assert ground_truth_label(MOVE_E, []) == SELF


def test_ground_truth_world_only_event() -> None:
    events = [{"type": "agent_moved", "cause": "world"}]
    assert ground_truth_label(MOVE_E, events) == WORLD


def test_ground_truth_both_self_and_world_events() -> None:
    events = [
        {"type": "agent_moved", "cause": "self"},
        {"type": "agent_moved", "cause": "world"},
    ]
    assert ground_truth_label(MOVE_E, events) == BOTH


def test_ground_truth_ignores_irrelevant_event_types() -> None:
    # food_regrown/capability_shift_start are world-state/meta events, not
    # direct effects on the agent's own realized transition this tick.
    events = [
        {"type": "food_regrown", "cause": "world"},
        {"type": "capability_shift_start", "cause": "world"},
    ]
    assert ground_truth_label(EAT, events) == SELF


def test_accuracy_tracker() -> None:
    tracker = AttributionAccuracyTracker()
    tracker.update(predicted=SELF, ground_truth=SELF, action=MOVE_E)
    tracker.update(predicted=WORLD, ground_truth=SELF, action=MOVE_E)
    tracker.update(predicted=WORLD, ground_truth=WORLD, action=NOOP)
    tracker.update(predicted=SELF, ground_truth=WORLD, action=NOOP)  # violation

    assert tracker.total == 4
    assert tracker.correct == 2
    assert tracker.accuracy == 0.5
    assert tracker.noop_self_violations == 1
    assert tracker.confusion[SELF][SELF] == 1
    assert tracker.confusion[SELF][WORLD] == 1
    assert tracker.confusion[WORLD][WORLD] == 1
    assert tracker.confusion[WORLD][SELF] == 1


def test_accuracy_tracker_state_dict_roundtrip() -> None:
    tracker = AttributionAccuracyTracker()
    tracker.update(predicted=SELF, ground_truth=SELF, action=MOVE_E)
    tracker.update(predicted=SELF, ground_truth=WORLD, action=NOOP)

    restored = AttributionAccuracyTracker()
    restored.load_state_dict(tracker.state_dict())
    assert restored.total == tracker.total
    assert restored.correct == tracker.correct
    assert restored.noop_self_violations == tracker.noop_self_violations
    assert restored.confusion == tracker.confusion
