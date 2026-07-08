"""Proposal-layer behavior: schema validation, generators on crafted
evidence (both source variants), dedup/rate-limit/decision-log mechanics,
confidence recalibration, and the full review lifecycle (generate ->
blind review -> approve -> ticket -> human-run evaluation -> unblind)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ledger.competence import REPORT_SCHEMA_VERSION, CompetenceReport, RegionCompetence
from proposals.evidence import Evidence
from proposals.generator import (
    ComputeBudgetGenerator,
    GeneratorSuite,
    HyperparameterGenerator,
    LoggingChangeGenerator,
    MemoryPolicyGenerator,
    RetrainingGenerator,
    recalibrate_confidence,
)
from proposals.schema import Proposal, load_all, save_proposal
from review.queue import ReviewQueue, blind_view

ROOT = Path(__file__).resolve().parent.parent


def _series(values: list[float]) -> tuple[list[int], list[float]]:
    return list(range(len(values))), values


def make_evidence(source: str = "ledger", **scalars: list[float]) -> Evidence:
    return Evidence(
        source=source, run_id="fixture-run", tick=10_000,
        scalars={tag: _series(vals) for tag, vals in scalars.items()},
        config={"rssm": {"free_nats": 1.0, "sleep_grad_steps": 100},
                "ppo": {"lr": 3e-4}, "checkpoints": {"interval": 50_000}},
    )


def degraded_report() -> CompetenceReport:
    return CompetenceReport(
        schema_version=REPORT_SCHEMA_VERSION, tick=10_000, run_id="fixture-run",
        regions=[RegionCompetence(
            region=(1, 2), task="all", n_samples=500,
            wm_loss_ema=3.0, wm_loss_ratio=2.0,
            body_brier_ema=0.2, body_brier_ratio=1.1,
            forecaster_nmse_ema=float("nan"),
            reward_rate_ema=0.1, reward_ratio=0.8,
            learning_progress=-0.01, adaptation_status="degrading",
            replay_coverage=0.001,
        )],
    )


# ------------------------------------------------------------------ schema


def test_schema_validation_rejects_bad_records() -> None:
    good = dict(
        type="retraining", created_tick=1, run_id="r", source="ledger",
        rationale="text", expected_benefit={"metric": "m", "direction": "up",
                                            "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 1, "gpu_hours": 1}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 10},
    )
    p = Proposal.new(**good)
    restored = Proposal.from_json(p.to_json())
    assert restored.id == p.id and restored.status == "pending"

    for corrupt in ({"type": "nonsense"}, {"source": ""},
                    {"confidence": 1.5}, {"rationale": "  "},
                    {"intervention_class": "sideways"}):
        with pytest.raises(ValueError):
            Proposal.new(**{**good, **corrupt})


def test_schema_v2_fields_roundtrip() -> None:
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id="r",
        source="architect:sonnet",  # open string (v2): not an enum member
        intervention_class="architecture",
        rationale="text", expected_benefit={"metric": "m", "direction": "up",
                                            "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=["code:agent/core_rssm.py@abc#L1-L9"],
        estimated_cost={"human_hours": 1, "gpu_hours": 1}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 10},
        provenance={"evidence_bundle_hash": "deadbeef00000000",
                    "generator_id": "architect", "prompt_hash": "cafe",
                    "model_id": "claude-x"},
        artifacts=["architect/diffs/p.diff"],
    )
    r = Proposal.from_json(p.to_json())
    assert r.schema_version == 2
    assert r.intervention_class == "architecture"
    assert r.source == "architect:sonnet"
    assert r.provenance["model_id"] == "claude-x"
    assert r.artifacts == ["architect/diffs/p.diff"]


def test_v1_record_migrates_to_v2_filling_defaults() -> None:
    v1 = {
        "schema_version": 1, "id": "prop-legacyrecord01", "type": "retraining",
        "created_tick": 5, "run_id": "r", "source": "ledger",
        "rationale": "legacy", "expected_benefit": {"metric": "m",
            "direction": "up", "magnitude_estimate": 0.1},
        "confidence": 0.5, "supporting_observations": [],
        "estimated_cost": {"human_hours": 0, "gpu_hours": 0}, "risks": [],
        "success_criteria": {"metric": "m", "threshold": 0,
                             "eval_window_ticks": 10},
        "status": "pending", "decision": {}, "linked_experiment_id": None,
        "realized_benefit": None, "target": "region-1-1",
        "proposed_change": None,
    }
    p = Proposal.from_json(json.dumps(v1))
    assert p.schema_version == 2
    assert p.intervention_class == "experiment"  # no knob -> experiment
    assert p.provenance == {} and p.artifacts == []

    v1_knob = {**v1, "id": "prop-legacyknob0001",
               "proposed_change": {"config_path": "rssm.free_nats",
                                   "new_value": 0.5}}
    assert Proposal.from_json(json.dumps(v1_knob)).intervention_class == "config"


def test_artifact_paths_must_stay_in_run_dir() -> None:
    base = dict(
        type="hyperparameter", created_tick=1, run_id="r", source="ledger",
        rationale="t", expected_benefit={"metric": "m", "direction": "up",
                                         "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 1},
    )
    for bad in (["../escape.diff"], ["/abs/path.diff"], ["a/../../b"], [""]):
        with pytest.raises(ValueError):
            Proposal.new(**base, artifacts=bad)
    ok = Proposal.new(**base, artifacts=["architect/diffs/ok.diff"])
    assert ok.artifacts == ["architect/diffs/ok.diff"]


# -------------------------------------------------------------- generators


def test_retraining_fires_on_ledger_and_logs_variants() -> None:
    gen = RetrainingGenerator()
    ev = make_evidence("ledger", **{"rssm/recon": [0.5] * 30})
    ev.competence = degraded_report()
    p = gen.generate(ev)
    assert p is not None and p.type == "retraining" and p.source == "ledger"
    assert "region (1, 2)" in p.rationale
    assert any("region-(1, 2)" in ref for ref in p.supporting_observations)

    ev_logs = make_evidence("logs_only",
                            **{"reward/rollout": [2.0] * 20 + [0.5] * 10})
    ev_logs.positions = np.array([[4, 4]] * 990 + [[28, 28]] * 10)
    p_logs = gen.generate(ev_logs)
    assert p_logs is not None and p_logs.source == "logs_only"
    assert "raw logs" in p_logs.rationale


def test_hyperparameter_detects_kl_pinned_at_free_nats() -> None:
    ev = make_evidence("ledger", **{"rssm/kl": [1.0] * 15})
    p = HyperparameterGenerator().generate(ev)
    assert p is not None and p.target == "rssm.free_nats"
    assert "tb:rssm/kl@step=" in p.supporting_observations[0]


def test_generated_proposals_carry_provenance_and_class() -> None:
    ev = make_evidence("ledger", **{"rssm/kl": [1.0] * 15})
    ev.bundle_hash = "abcd1234abcd1234"
    knob = HyperparameterGenerator().generate(ev)
    assert knob is not None
    assert knob.intervention_class == "config"          # carries a knob
    assert knob.provenance["generator_id"] == "propose_hyperparameter"
    assert knob.provenance["evidence_bundle_hash"] == "abcd1234abcd1234"

    ev2 = make_evidence("ledger", **{"rssm/recon": [0.5] * 30})
    ev2.competence = degraded_report()
    exp = RetrainingGenerator().generate(ev2)
    assert exp is not None and exp.intervention_class == "experiment"  # no knob


def test_memory_policy_is_ledger_only() -> None:
    scal = {"memory/stale_trip_rate_per_1k": [120.0] * 5}
    assert MemoryPolicyGenerator().generate(make_evidence("ledger", **scal)) is not None
    # logs_only evidence never carries memory telemetry (stripped upstream),
    # and the generator refuses to fire from that variant regardless.
    assert MemoryPolicyGenerator().generate(make_evidence("logs_only", **scal)) is None


def test_compute_budget_fires_under_budget() -> None:
    ev = make_evidence("ledger", **{"sleep/grad_steps": [40.0, 42.0, 38.0]})
    p = ComputeBudgetGenerator().generate(ev)
    assert p is not None and p.target == "rssm.sleep_grad_steps"
    assert "budget of 100" in p.rationale


def test_logging_change_fires_on_unexplained_variance() -> None:
    rng = np.random.default_rng(0)
    noisy = (rng.normal(0.01, 1.0, 40)).tolist()
    p = LoggingChangeGenerator().generate(
        make_evidence("ledger", **{"reward/rollout": noisy}))
    assert p is not None and p.target.startswith("scalar:")


# ---------------------------------------------------- suite mechanics


def test_suite_dedups_rate_limits_and_logs_decisions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    suite = GeneratorSuite(run_dir, generators=(HyperparameterGenerator,))
    ev_l = make_evidence("ledger", **{"rssm/kl": [1.0] * 15})
    ev_logs = make_evidence("logs_only")  # nothing to fire on

    fired = suite.run(ev_l, ev_logs)
    assert len(fired) == 1
    # Same evidence again: dedup suppresses.
    fired2 = suite.run(ev_l, ev_logs)
    assert fired2 == []

    decisions_file = run_dir / "proposals" / "generator_decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_file.read_text().splitlines()]
    assert any(d["decision"] == "FIRED" for d in decisions)
    # Same tick, same suite: the RATE LIMIT suppresses first.
    assert any(d["decision"] == "SUPPRESSED" and "rate-limited" in d["reason"]
               for d in decisions)
    assert any(d["source"] == "logs_only" and d["decision"] == "SUPPRESSED"
               for d in decisions)

    # A fresh suite over the same dir has no rate-limit memory but DOES
    # remember existing proposals: dedup survives restart.
    suite2 = GeneratorSuite(run_dir, generators=(HyperparameterGenerator,))
    assert suite2.run(ev_l, ev_logs) == []
    decisions = [json.loads(line) for line in decisions_file.read_text().splitlines()]
    assert any(d["decision"] == "SUPPRESSED" and "duplicate" in d["reason"]
               for d in decisions)


def test_recalibrate_confidence_binned_hit_rates(tmp_path: Path) -> None:
    history = []
    for i in range(30):
        p = Proposal.new(
            type="evaluation", created_tick=i, run_id="r", source="ledger",
            rationale="x", expected_benefit={"metric": "m", "direction": "up",
                                             "magnitude_estimate": 0},
            confidence=0.9 if i % 2 else 0.1, supporting_observations=[],
            estimated_cost={"human_hours": 0, "gpu_hours": 0}, risks=[],
            success_criteria={"metric": "m", "threshold": 0,
                              "eval_window_ticks": 1},
        )
        # High-confidence proposals actually miss; low-confidence ones hit.
        p.realized_benefit = {"met_success_criteria": not bool(i % 2)}
        history.append(p)
    mapper = recalibrate_confidence(history)
    assert mapper is not None
    assert mapper(0.9) == pytest.approx(0.0)   # overconfident bin -> 0% hits
    assert mapper(0.1) == pytest.approx(1.0)   # underconfident bin -> 100%
    assert recalibrate_confidence(history[:10]) is None  # < 20 evaluated


# ----------------------------------------------------------- review lifecycle


def _fixture_proposal(run_dir: Path, metric: str = "reward/rollout") -> Proposal:
    p = Proposal.new(
        type="hyperparameter", created_tick=100, run_id=run_dir.name,
        source="ledger", rationale="fixture rationale",
        expected_benefit={"metric": metric, "direction": "up",
                          "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=["tb:reward/rollout@step=1"],
        estimated_cost={"human_hours": 0.1, "gpu_hours": 0.1}, risks=[],
        success_criteria={"metric": metric, "threshold": -10.0,
                          "eval_window_ticks": 512},
    )
    save_proposal(p, run_dir)
    return p


def test_blind_review_hides_source_until_evaluated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    p = _fixture_proposal(run_dir)
    view = blind_view(ReviewQueue(run_dir).get(p.id))
    assert view["source"] == "<blinded until evaluated>"
    p.status = "evaluated"
    save_proposal(p, run_dir)
    view2 = blind_view(ReviewQueue(run_dir).get(p.id))
    assert view2["source"] == "ledger"


def test_modify_two_step_records_human_diff(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    p = _fixture_proposal(run_dir)
    queue = ReviewQueue(run_dir)
    edit_file = queue.modify_start(p.id)
    edited = json.loads(edit_file.read_text())
    edited["risks"] = ["human added a risk"]
    edit_file.write_text(json.dumps(edited, indent=2))
    result = queue.modify_apply(p.id)
    assert result.status == "modified"
    assert result.risks == ["human added a risk"]
    assert "human added a risk" in result.decision["human_diff"]
    assert not edit_file.exists()


def test_full_lifecycle_generate_approve_ticket_evaluate_unblind(tmp_path: Path) -> None:
    """The whole loop on fixture data: pending -> approved (ticket emitted)
    -> human-run evaluation (--ticket path) -> evaluated with
    realized_benefit -> source unblinded. Uses a real (tiny) PPO eval run."""
    import experiments.runner as runner
    from world.config import load_config

    run_dir = tmp_path / "runs" / "src-run"
    run_dir.mkdir(parents=True)
    # The evaluation loads the source run's config: give it a tiny real one.
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["device"] = "cpu"
    cfg["agent"] = {"hidden_size": 16, "gru_layers": 1, "encoder_channels": [4, 8]}
    cfg["ppo"].update(rollout_steps=16, seq_len=8, num_envs=2, episode_length=64,
                      minibatch_transitions=16, epochs=1, anneal_lr=False)
    cfg["run"]["assert_improvement"] = False
    (run_dir / "config.json").write_text(json.dumps(cfg))

    p = _fixture_proposal(run_dir)
    queue = ReviewQueue(run_dir)

    queue.decide(p.id, "approve", note="worth a try")
    ticket = Path("experiments/tickets") / f"{p.id}.md"
    assert ticket.exists()
    assert f"--ticket {p.id}" in ticket.read_text(encoding="utf-8")
    # Decision log is append-only jsonl and captured the approval.
    decisions = (run_dir / "proposals" / "decisions.jsonl").read_text().splitlines()
    assert json.loads(decisions[-1])["action"] == "approve"

    realized = runner.evaluate_ticket(
        p.id, run_dir, eval_ticks=512, eval_run_root=str(tmp_path / "eval_runs")
    )
    assert realized["met_success_criteria"] is True  # threshold -10 is generous

    final = queue.get(p.id)
    assert final.status == "evaluated"
    assert final.realized_benefit["metric"] == "reward/rollout"
    assert final.linked_experiment_id
    assert blind_view(final)["source"] == "ledger"  # unblinded now

    queue.rate(p.id, 4, note="useful")
    assert queue.get(p.id).decision["usefulness_rating"] == 4
    assert len(load_all(run_dir)) == 1
    ticket.unlink()  # keep the repo's tickets dir clean of test artifacts


def test_review_cli_commands(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Drive the actual `python -m review` entry point: list shows a blinded
    row, approve transitions status and emits the ticket, rate records."""
    from review.__main__ import main as review_main

    run_dir = tmp_path / "run"
    p = _fixture_proposal(run_dir)

    review_main(["--run-dir", str(run_dir), "list"])
    out = capsys.readouterr().out
    assert p.id in out and "<blinded until evaluated>" in out and "ledger" not in out

    review_main(["--run-dir", str(run_dir), "show", p.id])
    assert "<blinded until evaluated>" in capsys.readouterr().out

    review_main(["--run-dir", str(run_dir), "approve", p.id, "--note", "go"])
    out = capsys.readouterr().out
    assert "approved" in out and "BY HAND" in out.upper()
    ticket = Path("experiments/tickets") / f"{p.id}.md"
    assert ticket.exists()

    review_main(["--run-dir", str(run_dir), "rate", p.id, "5", "--note", "great"])
    assert "rated 5/5" in capsys.readouterr().out
    assert ReviewQueue(run_dir).get(p.id).decision["usefulness_rating"] == 5
    ticket.unlink()
