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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



from ledger.competence import CompetenceReport
from proposals.schema import Proposal, load_all, save_proposal
from researcher.questions import Question

TRACTABILITY_FLOOR = 0.05

# Researcher-emitted agenda items live in the SAME proposal queue as the
# rule-generator recommendations (stage-C3); this source marks them. It is
# NOT a blind-review control variant (that is ledger vs logs_only), so it
# may be shown freely — but research_agenda.md still prints no source at all,
# so a co-listed generator proposal's ledger/logs_only origin never leaks.
RESEARCHER_SOURCE = "researcher"


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


# ---------------------------------------------------- queue emission (C3)


def agenda_item_to_proposal(
    item: AgendaItem, run_dir: str | Path, *, ranker_id: str,
    bundle_hash: str, evidence_refs: list[str], tick: int,
) -> Proposal:
    """One ranked QUESTION item -> an intervention_class=experiment proposal
    in the shared queue. The agenda score decomposition, predicted gain, and
    hypothesis links live in provenance (schema has no score field); the
    predicted gain is also the expected_benefit magnitude, per spec."""
    d = item.decomposition
    magnitude = (item.predicted_gain if item.predicted_gain is not None
                 else item.score)
    return Proposal.new(
        type="evaluation",              # the researcher recommends an experiment
        intervention_class="experiment",
        created_tick=tick, run_id=Path(run_dir).name, source=RESEARCHER_SOURCE,
        rationale=item.statement,
        expected_benefit={"metric": "researcher/predicted_gain",
                          "direction": "up",
                          "magnitude_estimate": float(magnitude)},
        confidence=0.5,
        supporting_observations=list(evidence_refs),
        estimated_cost={"human_hours": 0.5,
                        "gpu_hours": float(item.experiment.get("cost", 1.0))},
        risks=[],
        success_criteria={"metric": "researcher/predicted_gain",
                          "threshold": 0.0, "eval_window_ticks": 20_000},
        target=item.ref,                # dedup key = the question id
        provenance={
            "evidence_bundle_hash": bundle_hash,
            "generator_id": f"researcher:agenda:{ranker_id}",
            "agenda_score": float(item.score),
            "agenda_decomposition": d,
            "predicted_gain": item.predicted_gain,
            "hypothesis_links": list(item.hypothesis_links),
            "experiment": item.experiment,
        },
    )


def emit_agenda(
    items: list[AgendaItem], questions: list[Question], run_dir: str | Path,
    *, ranker_id: str = "v1", bundle_hash: str = "", tick: int = 0,
) -> list[Proposal]:
    """Emit each ranked question item into the queue as an experiment
    proposal, deduped by (type=evaluation, target=question id) so repeated
    sleep-phase passes never duplicate. Returns the newly emitted proposals
    (proposal-kind agenda items are already in the queue and are skipped)."""
    ev_by_ref = {q.id: q.evidence_refs for q in questions}
    existing = {(p.type, p.target) for p in load_all(run_dir)}
    emitted: list[Proposal] = []
    for item in items:
        if item.kind != "question":
            continue
        key = ("evaluation", item.ref)
        if key in existing:
            continue
        proposal = agenda_item_to_proposal(
            item, run_dir, ranker_id=ranker_id, bundle_hash=bundle_hash,
            evidence_refs=ev_by_ref.get(item.ref, []), tick=tick)
        save_proposal(proposal, run_dir)
        existing.add(key)
        emitted.append(proposal)
    return emitted


# ------------------------------------------------------------------ output


def _render_score(p: Proposal) -> float:
    """Ranking score for research_agenda.md: the stored agenda score for
    researcher items; |magnitude| x confidence for co-listed generator
    proposals (matching rank_v1's proposal value term)."""
    if "agenda_score" in p.provenance:
        return float(p.provenance["agenda_score"])
    return abs(p.expected_benefit.get("magnitude_estimate", 0.0)) * p.confidence


def render_agenda_md(run_dir: str | Path, tick: int | None = None,
                     top: int = 10) -> Path:
    """Render research_agenda.md FROM the unified queue (stage-C3): all
    PENDING proposals, sorted by score. Source is deliberately never
    printed, so a generator proposal's ledger/logs_only origin cannot leak
    through this artifact (blind review stays intact)."""
    out_dir = Path(run_dir) / "researcher"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in load_all(run_dir) if p.status == "pending"]
    pending.sort(key=lambda p: (-_render_score(p), p.id))

    header = "# Research agenda" + (f" @ tick {tick}" if tick is not None else "")
    lines = [header, "",
             "The single top-level research artifact, generated from the "
             "proposal queue: what the agent does not understand and which "
             "experiment reduces that uncertainty most efficiently, alongside "
             "the rule-generator recommendations. Nothing here executes; the "
             "human picks. (Source is intentionally omitted — blind review.)",
             ""]
    for rank, p in enumerate(pending[:top], start=1):
        prov = p.provenance
        lines += [f"## {rank}. [{p.intervention_class}] {p.rationale[:160]}", ""]
        d = prov.get("agenda_decomposition")
        if d:
            lines.append(
                f"- score {_render_score(p):.4f} = value {d['value']:.3f} "
                f"x tractability {d['tractability']:.3f} "
                f"x novelty {d['novelty']:.3f} / cost {d['cost']:.2f}")
        else:
            lines.append(f"- score {_render_score(p):.4f} "
                         f"(type `{p.type}`, confidence {p.confidence:.2f})")
        if prov.get("experiment"):
            lines.append(f"- experiment: `{json.dumps(prov['experiment'])}`")
        if prov.get("predicted_gain") is not None:
            lines.append(f"- predicted gain (EIG v2): {prov['predicted_gain']:.4f}")
        if prov.get("hypothesis_links"):
            lines.append(f"- a result would move: "
                         f"{', '.join(prov['hypothesis_links'])}")
        lines.append("")
    md_path = out_dir / "research_agenda.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_agenda(
    items: list[AgendaItem], run_dir: str | Path, tick: int,
    *, questions: list[Question] | None = None, ranker_id: str = "v1",
    bundle_hash: str = "", top: int = 10,
) -> tuple[list[Proposal], Path]:
    """Emit the ranked question items into the queue and render
    research_agenda.md from it. Returns (emitted proposals, md path).

    Replaces the old parallel agenda_<tick>.json artifact: the queue is now
    the single source of truth (the dashboard reads it too)."""
    emitted = emit_agenda(items, questions or [], run_dir,
                          ranker_id=ranker_id, bundle_hash=bundle_hash, tick=tick)
    md_path = render_agenda_md(run_dir, tick=tick, top=top)
    return emitted, md_path
