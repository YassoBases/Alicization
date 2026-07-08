"""The ranked research agenda — the single top-level research artifact.

Ranker v1 (heuristic; the EIG upgrade is researcher/eig.py, config-
selectable):

    score = value * tractability * novelty_decay / cost

  value        questions: normalized uncertainty. proposals:
               expected_benefit.magnitude_estimate * confidence.
  tractability learning progress in the relevant region (competence
               report). THE NOISY-TV GUARD: high uncertainty with zero or
               negative learning progress must rank LOW — an irreducibly
               random region maxes out disagreement forever and teaches
               nothing; tractability floors at 0.05 there.
  novelty      1 / (1 + recent near-duplicate executions), from the
               executed-item history.
  cost         from the experiment menu / the proposal's estimated cost.

Output per sleep phase: runs/<id>/researcher/agenda_<tick>.json plus a
rendered research_agenda.md — top 10 items with the question, the proposed
experiment, the score decomposition, and which hypothesis a result would
move. Deterministic given a fixed log store (tested).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any



from ledger.competence import CompetenceReport
from proposals.schema import Proposal
from researcher.questions import Question

TRACTABILITY_FLOOR = 0.05


@dataclass
class AgendaItem:
    id: str
    kind: str                      # question | proposal
    ref: str                       # question/proposal id
    statement: str
    experiment: dict[str, Any]
    score: float
    decomposition: dict[str, float]
    hypothesis_links: list[str] = field(default_factory=list)
    predicted_gain: float | None = None   # filled by the EIG ranker (v2)


def _tractability(region: tuple[int, int] | None,
                  competence: CompetenceReport | None) -> float:
    """Learning progress in the relevant region, squashed to (floor, 1].

    No region / no report -> neutral 0.5. Zero or negative progress with
    the region present -> the noisy-TV floor.
    """
    if region is None or competence is None:
        return 0.5
    for r in competence.regions:
        if tuple(r.region) == tuple(region):
            if r.learning_progress <= 0:
                return TRACTABILITY_FLOOR
            return float(min(1.0, 0.5 + 10.0 * r.learning_progress))
    return 0.5


def _novelty(item_key: str, history: list[dict[str, Any]],
             window: int = 10) -> float:
    recent = [h for h in history[-window:] if h.get("item_key") == item_key]
    return 1.0 / (1.0 + len(recent))


def rank_v1(
    questions: list[Question],
    proposals: list[Proposal],
    competence: CompetenceReport | None,
    history: list[dict[str, Any]] | None = None,
) -> list[AgendaItem]:
    """Deterministic heuristic ranking over questions + pending proposals."""
    history = history or []
    items: list[AgendaItem] = []

    for q in questions:
        experiment = (q.candidate_experiments[0] if q.candidate_experiments
                      else {"name": "run_battery", "cost": 4.0})
        value = float(q.uncertainty)
        tract = _tractability(q.region, competence)
        novelty = _novelty(q.id, history)
        cost = float(experiment.get("cost", 1.0))
        score = value * tract * novelty / cost
        items.append(AgendaItem(
            id=f"agenda-{q.id}", kind="question", ref=q.id,
            statement=q.statement, experiment=experiment, score=score,
            decomposition={"value": value, "tractability": tract,
                           "novelty": novelty, "cost": cost},
            hypothesis_links=[ref.split(":", 1)[1] for ref in q.evidence_refs
                              if ref.startswith("hypothesis:")],
        ))

    for p in proposals:
        if p.status != "pending":
            continue
        value = float(abs(p.expected_benefit.get("magnitude_estimate", 0.0))
                      * p.confidence)
        novelty = _novelty(p.id, history)
        cost = float(p.estimated_cost.get("gpu_hours", 0.0)
                     + p.estimated_cost.get("human_hours", 0.0)) or 1.0
        score = value * 0.5 * novelty / cost  # neutral tractability
        items.append(AgendaItem(
            id=f"agenda-{p.id}", kind="proposal", ref=p.id,
            statement=p.rationale.splitlines()[0][:160],
            experiment={"name": "proposal_ticket", "proposal": p.id,
                        "cost": cost},
            score=score,
            decomposition={"value": value, "tractability": 0.5,
                           "novelty": novelty, "cost": cost},
        ))

    items.sort(key=lambda i: (-i.score, i.id))  # deterministic tiebreak
    return items


# ------------------------------------------------------------------ output


def write_agenda(
    items: list[AgendaItem], run_dir: str | Path, tick: int, top: int = 10
) -> tuple[Path, Path]:
    out_dir = Path(run_dir) / "researcher"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agenda_{tick:012d}.json"
    json_path.write_text(json.dumps([asdict(i) for i in items], indent=2),
                         encoding="utf-8")

    lines = [f"# Research agenda @ tick {tick}", "",
             "The single top-level research artifact: what the agent does "
             "not understand, and which experiment reduces that uncertainty "
             "most efficiently. Nothing here executes; the human picks.", ""]
    for rank, item in enumerate(items[:top], start=1):
        d = item.decomposition
        lines += [
            f"## {rank}. [{item.kind}] {item.statement}",
            "",
            f"- experiment: `{json.dumps(item.experiment)}`",
            f"- score {item.score:.4f} = value {d['value']:.3f} "
            f"x tractability {d['tractability']:.3f} "
            f"x novelty {d['novelty']:.3f} / cost {d['cost']:.2f}",
        ]
        if item.predicted_gain is not None:
            lines.append(f"- predicted gain (EIG v2): {item.predicted_gain:.4f}")
        if item.hypothesis_links:
            lines.append(f"- a result would move: "
                         f"{', '.join(item.hypothesis_links)}")
        lines.append("")
    md_path = out_dir / "research_agenda.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
