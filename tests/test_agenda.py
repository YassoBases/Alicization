"""Agenda v1: determinism on a frozen store, the noisy-TV guard, proposal
candidates in the same ranking, and rendered output."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.competence import REPORT_SCHEMA_VERSION, CompetenceReport, RegionCompetence
from proposals.schema import Proposal
from researcher.agenda import TRACTABILITY_FLOOR, rank_v1, write_agenda
from researcher.questions import Question


def q(qid: str, uncertainty: float, region=None) -> Question:
    return Question(
        id=qid, type="world_uncertainty",
        statement=f"what are the dynamics of {qid}?",
        evidence_refs=["viz_state:epistemic_map:cell=(1,1)"],
        candidate_experiments=[{"name": "directed_visit", "cost": 2.0}],
        uncertainty=uncertainty, region=region,
    )


def region(r, c, progress: float) -> RegionCompetence:
    return RegionCompetence(
        region=(r, c), task="all", n_samples=500, wm_loss_ema=1.0,
        wm_loss_ratio=1.0, body_brier_ema=0.1, body_brier_ratio=1.0,
        forecaster_nmse_ema=float("nan"), reward_rate_ema=0.5,
        reward_ratio=1.0, learning_progress=progress,
        adaptation_status="stable", replay_coverage=0.1,
    )


def competence(*regions: RegionCompetence) -> CompetenceReport:
    return CompetenceReport(schema_version=REPORT_SCHEMA_VERSION, tick=0,
                            run_id="fixture", regions=list(regions))


def pending_proposal() -> Proposal:
    return Proposal.new(
        type="hyperparameter", created_tick=5, run_id="fixture",
        source="ledger", rationale="lower the lr because reasons",
        expected_benefit={"metric": "reward/rollout", "direction": "up",
                          "magnitude_estimate": 0.4},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0.5, "gpu_hours": 0.5}, risks=[],
        success_criteria={"metric": "reward/rollout", "threshold": 0,
                          "eval_window_ticks": 100},
    )


def test_agenda_is_deterministic_on_frozen_inputs() -> None:
    questions = [q("q-a", 0.9, region=(0, 0)), q("q-b", 0.5, region=(1, 1))]
    comp = competence(region(0, 0, 0.02), region(1, 1, 0.03))
    props = [pending_proposal()]
    first = rank_v1(questions, props, comp)
    second = rank_v1(list(questions), list(props), comp)
    assert [i.id for i in first] == [i.id for i in second]
    assert [i.score for i in first] == [i.score for i in second]


def test_noisy_tv_guard_floors_zero_progress_regions() -> None:
    """Maximal uncertainty with zero learning progress must NOT hold the top
    slot: the pure-noise region ranks below a moderate, learnable one."""
    noisy_tv = q("q-noise", 1.0, region=(0, 0))       # max uncertainty
    learnable = q("q-learn", 0.5, region=(1, 1))
    comp = competence(region(0, 0, 0.0),              # zero progress: noise
                      region(1, 1, 0.05))             # real learning signal
    ranked = rank_v1([noisy_tv, learnable], [], comp)
    assert ranked[0].ref == "q-learn"
    noise_item = next(i for i in ranked if i.ref == "q-noise")
    assert noise_item.decomposition["tractability"] == TRACTABILITY_FLOOR


def test_proposals_enter_the_same_agenda() -> None:
    ranked = rank_v1([q("q-a", 0.1, region=None)], [pending_proposal()], None)
    kinds = {i.kind for i in ranked}
    assert kinds == {"question", "proposal"}
    prop_item = next(i for i in ranked if i.kind == "proposal")
    assert prop_item.decomposition["value"] == pytest.approx(0.4 * 0.5)


def test_write_agenda_emits_proposals_and_renders_from_queue(tmp_path: Path) -> None:
    from proposals.schema import load_all

    run = tmp_path / "run"
    questions = [q("q-a", 0.9, region=None)]
    items = rank_v1(questions, [], None)
    emitted, md_path = write_agenda(items, run, tick=1234, questions=questions,
                                    bundle_hash="hash0", ranker_id="v1")
    # Emitted as an experiment proposal into the shared queue (stage-C3).
    assert len(emitted) == 1
    p = emitted[0]
    assert p.intervention_class == "experiment" and p.source == "researcher"
    assert p.target == "q-a"
    assert p.provenance["evidence_bundle_hash"] == "hash0"
    assert p.provenance["agenda_decomposition"]["value"] == 0.9
    assert "viz_state:epistemic_map" in p.supporting_observations[0]
    assert [x.id for x in load_all(run)] == [p.id]
    # research_agenda.md rendered FROM the queue.
    md = md_path.read_text()
    assert "score" in md and "tractability" in md and p.rationale[:20] in md
    assert md_path.parent.name == "researcher"


def test_emit_agenda_dedups_across_reruns(tmp_path: Path) -> None:
    from proposals.schema import load_all

    run = tmp_path / "run"
    questions = [q("q-a", 0.9, region=None), q("q-b", 0.5, region=None)]
    items = rank_v1(questions, [], None)
    first, _ = write_agenda(items, run, tick=1, questions=questions)
    assert len(first) == 2
    # A second sleep-phase pass over the same store emits nothing new.
    second, _ = write_agenda(items, run, tick=2, questions=questions)
    assert second == []
    assert len(load_all(run)) == 2
