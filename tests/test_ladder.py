"""Stage-C4 evaluation ladder: tier-0 auto smoke-A/B over config knobs,
independent-metric scoring, the tautology flag, and tier-binned
recalibration. Training is mocked (module-level _eval_run/_series) so these
stay unit-fast; the real >=10-knob run is the Gate-C acceptance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import experiments.runner as runner
from proposals.generator import recalibrate_confidence
from proposals.schema import Proposal, load_all, save_proposal


def _knob(run_dir: Path, cfg_path: str, metric: str, *, status: str = "pending",
          intervention_class: str = "config", knob: bool = True) -> Proposal:
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id=run_dir.name,
        source="ledger", rationale="knob", intervention_class=intervention_class,
        expected_benefit={"metric": metric, "direction": "down",
                          "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 0, "gpu_hours": 1}, risks=[],
        success_criteria={"metric": metric, "threshold": 0.5,
                          "eval_window_ticks": 1000},
        target=cfg_path,
        proposed_change=({"config_path": cfg_path, "new_value": 0.5}
                         if knob else None),
    )
    p.status = status
    save_proposal(p, run_dir)
    return p


# ------------------------------------------------------------ pure helpers


def test_apply_change_dotted_path() -> None:
    cfg = {"rssm": {"free_nats": 1.0}, "ppo": {"lr": 3e-4}}
    out = runner._apply_change(cfg, {"config_path": "rssm.free_nats",
                                     "new_value": 0.5})
    assert out["rssm"]["free_nats"] == 0.5
    assert cfg["rssm"]["free_nats"] == 1.0  # original untouched (deepcopy)


def test_tautology_flag_flags_free_nats_kl_only() -> None:
    p = _knob(Path("r"), "rssm.free_nats", "rssm/kl")  # not saved (run="r")
    assert runner.tautology_flag(p) is not None
    p2 = _knob(Path("r"), "rssm.free_nats", "rssm/recon")
    assert runner.tautology_flag(p2) is None
    p3 = _knob(Path("r"), "ppo.lr", "rssm/kl")
    assert runner.tautology_flag(p3) is None


# ------------------------------------------------------------ tier-0 ladder


@pytest.fixture()
def mocked_eval(monkeypatch: pytest.MonkeyPatch):
    """Skip training: control ~ constant, treated ~ lower (a real effect on
    the proposal's own metric and on wm_loss; reward flat)."""
    def fake_eval_run(cfg, ticks, run_root, needs_sleep):  # noqa: ANN001
        # tag which arm by the applied knob value living in cfg
        return Path(run_root) / ("treated" if _is_treated(cfg) else "control")

    def _is_treated(cfg) -> bool:  # noqa: ANN001
        return cfg.get("rssm", {}).get("free_nats") == 0.5 or \
            cfg.get("ppo", {}).get("lr") == 0.5 or cfg.get("_treated", False)

    def fake_series(run_dir, metric):  # noqa: ANN001
        treated = run_dir.name == "treated"
        base = {"rssm/kl": 1.0, "rssm/recon": 1.0, "loss/wm": 2.0,
                "sleep/wm_total": 2.0, "reward/rollout": 0.5}.get(metric, 1.0)
        arr = np.full(6, base, dtype=float)
        arr += np.linspace(0, 0.01, 6)  # tiny nonzero variance (control std>0)
        if treated and metric != "reward/rollout":
            arr -= 0.3
        return arr

    monkeypatch.setattr(runner, "_eval_run", fake_eval_run)
    monkeypatch.setattr(runner, "_series", fake_series)


def test_run_ladder_evaluates_only_pending_config_knobs(
        tmp_path: Path, mocked_eval) -> None:
    run_dir = tmp_path / "runs" / "src"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text('{"ppo": {}, "rssm": {}, "run": {}}')

    knob = _knob(run_dir, "rssm.free_nats", "rssm/kl")           # tier-0 target
    _knob(run_dir, "rssm.free_nats", "rssm/recon", status="approved")  # not pending
    _knob(run_dir, "region-1-1", "reward/rollout", knob=False,   # not a knob
          intervention_class="experiment")

    evaluated = runner.run_ladder(run_dir, eval_run_root=str(tmp_path / "ev"))
    assert [p.id for p in evaluated] == [knob.id]

    done = {p.id: p for p in load_all(run_dir)}
    got = done[knob.id]
    assert got.status == "evaluated"
    rb = got.realized_benefit
    assert rb["evaluation"] == "smoke_ab"
    # Independent metrics scored alongside the own criterion (both tiers).
    assert set(rb["independent_metrics"]) == {"wm_loss", "reward"}
    # free_nats knob judged on rssm/kl is flagged near-tautological.
    assert rb["tautological_criterion"] is not None
    # The others were left untouched by tier-0.
    assert done != {} and all(
        p.realized_benefit is None for p in load_all(run_dir) if p.id != knob.id)


def test_ladder_degenerate_control_is_nan(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "runs" / "src"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text('{"ppo": {}, "rssm": {}, "run": {}}')
    knob = _knob(run_dir, "rssm.free_nats", "rssm/recon")
    monkeypatch.setattr(runner, "_eval_run",
                        lambda cfg, t, r, s: Path(r) / "x")
    # Constant control series (std 0): the A/B benefit must be NaN, never
    # an astronomical number (stage-A degenerate-control guard, both tiers).
    monkeypatch.setattr(runner, "_series", lambda d, m: np.ones(1))
    [p] = runner.run_ladder(run_dir, eval_run_root=str(tmp_path / "ev"))
    assert np.isnan(p.realized_benefit["benefit_normalized"])


# ------------------------------------------------------ tier-binned recal


def _evaluated(conf: float, hit: bool, evaluation: str) -> Proposal:
    p = Proposal.new(
        type="evaluation", created_tick=1, run_id="r", source="ledger",
        rationale="x", expected_benefit={"metric": "m", "direction": "up",
                                         "magnitude_estimate": 0},
        confidence=conf, supporting_observations=[],
        estimated_cost={"human_hours": 0, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 1})
    p.realized_benefit = {"met_success_criteria": hit, "evaluation": evaluation}
    return p


def test_recalibrate_bins_tier0_separately_from_tier1() -> None:
    # 20 tier-0 (smoke_ab) where high confidence MISSES; 20 tier-1 (ab) where
    # high confidence HITS. Tier-scoped recalibration must disagree.
    history = ([_evaluated(0.9, i % 2 == 0, "smoke_ab") for i in range(20)]
               + [_evaluated(0.9, True, "ab") for _ in range(20)])
    tier0 = recalibrate_confidence(history, tier="smoke_ab")
    tier1 = recalibrate_confidence(history, tier="ab")
    assert tier0 is not None and tier1 is not None
    assert tier0(0.9) == pytest.approx(0.5)   # half hit under tier-0
    assert tier1(0.9) == pytest.approx(1.0)   # all hit under tier-1
    # Fewer than 20 in a tier -> keep heuristics (None).
    assert recalibrate_confidence(history[:5], tier="smoke_ab") is None
