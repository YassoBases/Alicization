"""Ground-truth cause-attribution scoring — EVALUATION ONLY.

Deliberately outside ledger/: nothing that trains any Ledger head may import
this module or use its outputs in a loss. It exists solely to score
ledger.attribution's self-supervised predictions against the world's
ground-truth cause labels (world/levers.py's ghost events), which per
CLAUDE.md's hard rules must never enter any observation or loss.

training/ppo.py calls ``ground_truth_label`` to build a read-only accuracy
counter; the resulting labels never touch a tensor that flows into
``.backward()``.
"""

from __future__ import annotations

from typing import Any

from ledger.attribution import BOTH, SELF, WORLD
from world.engine import NOOP

# Event types that directly change the agent's OWN realized (dpos, denergy)
# this tick. food_regrown/food_relocated/seasonal_shift/capability_shift_*
# are world-state or meta events that don't themselves move/feed the agent,
# so they're irrelevant to attributing THIS tick's observed transition.
_RELEVANT_EVENT_TYPES = ("agent_moved", "food_consumed")


def ground_truth_label(action: int, tick_events: list[dict[str, Any]]) -> int:
    """3-way ground-truth label for one agent's tick, from its raw event list.

    NOOP is a structural exception, mirroring ledger.attribution.pseudo_label:
    the fixed action table guarantees NOOP has no self-caused effect, so it is
    always WORLD, regardless of what (if anything) the event log shows.
    """
    if action == NOOP:
        return WORLD
    relevant = [e for e in tick_events if e["type"] in _RELEVANT_EVENT_TYPES]
    self_effect = any(e["cause"] == "self" for e in relevant)
    world_effect = any(e["cause"] == "world" for e in relevant)
    if self_effect and world_effect:
        return BOTH
    if world_effect:
        return WORLD
    return SELF


class AttributionAccuracyTracker:
    """Cumulative (predicted vs. ground-truth) accuracy over a whole run.

    Read-only bookkeeping: ``update`` takes plain Python ints, never tensors,
    so there is no way for this to end up inside an autograd graph.
    """

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.noop_self_violations = 0
        self.confusion = [[0] * 3 for _ in range(3)]  # [ground_truth][predicted]

    def update(self, predicted: int, ground_truth: int, action: int) -> None:
        self.total += 1
        if predicted == ground_truth:
            self.correct += 1
        self.confusion[ground_truth][predicted] += 1
        if action == NOOP and predicted == SELF:
            self.noop_self_violations += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "total": self.total,
            "noop_self_violations": self.noop_self_violations,
            "confusion": [row[:] for row in self.confusion],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.correct = state["correct"]
        self.total = state["total"]
        self.noop_self_violations = state["noop_self_violations"]
        self.confusion = [row[:] for row in state["confusion"]]
