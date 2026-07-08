"""Review queue: list, inspect, and decide on proposals — blind to source.

BLIND REVIEW: the ``source`` field (ledger vs logs_only) is hidden from
every rendering until a proposal reaches status=evaluated. This keeps the
P8.6 ledger-vs-logs comparison honest: the reviewer cannot favor a variant
they cannot see.

Approval emits an experiment TICKET — a markdown stub under
experiments/tickets/ containing the exact runner command the HUMAN should
execute. Nothing auto-executes (structural tests cover this package).

Every decision is appended to an immutable decisions.jsonl — the proposal
history is itself a dataset.

The spec's "modify opens $EDITOR" collides with this package's own
no-subprocess rule; modify is therefore two-step and execution-free:
``modify <id>`` copies the JSON to <id>.edit.json for the human to edit
with whatever they like; ``modify <id> --apply`` diffs, records the human's
diff in the decision record, and applies it.
"""

from __future__ import annotations

import difflib
import json
import time
from pathlib import Path
from typing import Any

from proposals.schema import Proposal, load_all, proposals_dir, save_proposal

TICKETS_DIR = Path("experiments") / "tickets"

_DECISION_STATUS = {
    "approve": "approved",
    "reject": "rejected",
    "postpone": "postponed",
    "partial": "partially_approved",
}

# Legal state machine: a human may decide only records that are still open.
# approved/partially_approved advance to evaluated ONLY via the runner
# (--ticket); rejected and evaluated are terminal for decisions (rejected
# targets may be RE-PROPOSED by generators — that path is measured by the
# repeated-after-denial statistic, not smuggled through re-decisions).
_DECIDABLE_STATUSES = ("pending", "postponed", "modified")
_MODIFIABLE_STATUSES = ("pending", "postponed")


def blind_view(p: Proposal) -> dict[str, Any]:
    """Public rendering: source hidden until evaluated."""
    view = json.loads(p.to_json())
    if p.status != "evaluated":
        view["source"] = "<blinded until evaluated>"
    return view


class ReviewQueue:
    """All proposal records under one run dir, plus the decision log."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.decisions_path = proposals_dir(self.run_dir) / "decisions.jsonl"

    # ------------------------------------------------------------- reading

    def proposals(self, status: str | None = None,
                  ptype: str | None = None) -> list[Proposal]:
        out = load_all(self.run_dir)
        if status:
            out = [p for p in out if p.status == status]
        if ptype:
            out = [p for p in out if p.type == ptype]
        return out

    def get(self, proposal_id: str) -> Proposal:
        for p in load_all(self.run_dir):
            if p.id == proposal_id:
                return p
        raise KeyError(f"no proposal {proposal_id!r} under {self.run_dir}")

    # ------------------------------------------------------------ decisions

    def _append_decision(self, record: dict[str, Any]) -> None:
        """Immutable log: append-only, one JSON object per line."""
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.decisions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def decide(self, proposal_id: str, action: str, note: str = "") -> Proposal:
        if action not in _DECISION_STATUS:
            raise ValueError(f"unknown action {action!r}")
        p = self.get(proposal_id)
        if p.status not in _DECIDABLE_STATUSES:
            raise ValueError(
                f"illegal transition: cannot {action} a proposal in status "
                f"{p.status!r} (decidable: {_DECIDABLE_STATUSES})"
            )
        p.status = _DECISION_STATUS[action]
        p.decision = {**p.decision, "timestamp": time.time(), "note": note}
        save_proposal(p, self.run_dir)
        self._append_decision({
            "proposal_id": p.id, "action": action, "status": p.status,
            "note": note, "timestamp": p.decision["timestamp"],
        })
        if p.status in ("approved", "partially_approved"):
            self._emit_ticket(p)
        return p

    def rate(self, proposal_id: str, rating: int, note: str = "") -> Proposal:
        if not 1 <= rating <= 5:
            raise ValueError("usefulness_rating is 1-5")
        p = self.get(proposal_id)
        p.decision = {**p.decision, "usefulness_rating": rating,
                      "rating_note": note, "rated_at": time.time()}
        save_proposal(p, self.run_dir)
        self._append_decision({
            "proposal_id": p.id, "action": "rate", "rating": rating,
            "note": note, "timestamp": p.decision["rated_at"],
        })
        return p

    # --------------------------------------------------------------- modify

    def edit_path(self, proposal_id: str) -> Path:
        return proposals_dir(self.run_dir) / f"{proposal_id}.edit.json"

    def modify_start(self, proposal_id: str) -> Path:
        """Step 1: copy the record for the human to edit (no editor spawn —
        this package never executes anything)."""
        p = self.get(proposal_id)
        if p.status not in _MODIFIABLE_STATUSES:
            raise ValueError(
                f"illegal transition: cannot modify a proposal in status "
                f"{p.status!r} (modifiable: {_MODIFIABLE_STATUSES})"
            )
        path = self.edit_path(proposal_id)
        path.write_text(p.to_json(), encoding="utf-8")
        return path

    def modify_apply(self, proposal_id: str) -> Proposal:
        """Step 2: diff the edited copy against the record, store the human's
        diff in the decision, apply it (status=modified)."""
        original = self.get(proposal_id)
        edit_file = self.edit_path(proposal_id)
        if not edit_file.exists():
            raise FileNotFoundError(f"run `modify {proposal_id}` first: {edit_file}")
        edited = Proposal.from_json(edit_file.read_text(encoding="utf-8"))
        if edited.id != original.id:
            raise ValueError("the proposal id may not be edited")
        diff = "\n".join(difflib.unified_diff(
            original.to_json().splitlines(), edited.to_json().splitlines(),
            fromfile="original", tofile="modified", lineterm="",
        ))
        edited.status = "modified"
        edited.decision = {**edited.decision, "timestamp": time.time(),
                           "human_diff": diff}
        save_proposal(edited, self.run_dir)
        edit_file.unlink()
        self._append_decision({
            "proposal_id": edited.id, "action": "modify", "status": "modified",
            "human_diff": diff, "timestamp": edited.decision["timestamp"],
        })
        return edited

    # -------------------------------------------------------------- tickets

    def _emit_ticket(self, p: Proposal) -> Path:
        """The human-execution handoff: an exact command, never an execution."""
        TICKETS_DIR.mkdir(parents=True, exist_ok=True)
        eval_ticks = p.success_criteria["eval_window_ticks"]
        command = (
            f"python -m experiments.runner --ticket {p.id} "
            f"--run-dir {self.run_dir} --eval-ticks {eval_ticks}"
        )
        body = "\n".join([
            f"# Experiment ticket: {p.id}",
            "",
            f"- proposal type: `{p.type}`  status: `{p.status}`",
            f"- created at tick {p.created_tick} in run `{p.run_id}`",
            f"- success criteria: `{p.success_criteria['metric']}` vs threshold "
            f"`{p.success_criteria['threshold']}` over {eval_ticks} ticks",
            "",
            "## Rationale",
            "",
            p.rationale,
            "",
            "## To execute (HUMAN runs this by hand — nothing auto-executes)",
            "",
            "```bash",
            command,
            "```",
            "",
            "The runner writes realized_benefit back into the proposal record "
            "and flips it to status=evaluated (unblinding its source).",
        ])
        path = TICKETS_DIR / f"{p.id}.md"
        path.write_text(body, encoding="utf-8")
        return path
