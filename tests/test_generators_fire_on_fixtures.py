"""Every generator fires exactly on its synthetic trigger trace and stays
SILENT otherwise (Section 18 delta). One trigger + one benign fixture per
generator; the stubs must never fire."""

from __future__ import annotations

import numpy as np
import pytest

from ledger.competence import REPORT_SCHEMA_VERSION, CompetenceReport, RegionCompetence
from proposals.evidence import Evidence
from proposals.generator import (
    CheckpointScheduleGenerator,
    ComputeBudgetGenerator,
    CurriculumGenerator,
    DatasetExtensionGenerator,
    EvaluationGenerator,
    HyperparameterGenerator,
    LoggingChangeGenerator,
    MemoryPolicyGenerator,
    RetrainingGenerator,
    VisualizationGenerator,
)


def _series(values: list[float]) -> tuple[list[int], list[float]]:
    return list(range(len(values))), values


def evidence(source: str = "ledger", **scalars: list[float]) -> Evidence:
    return Evidence(
        source=source, run_id="fixture", tick=10_000,
        scalars={tag: _series(v) for tag, v in scalars.items()},
        config={"rssm": {"free_nats": 1.0, "sleep_grad_steps": 100},
                "ppo": {"lr": 3e-4}, "checkpoints": {"interval": 50_000}},
    )


def _region(status: str, ratio: float, coverage: float,
            progress: float, region=(1, 1)) -> RegionCompetence:
    return RegionCompetence(
        region=region, task="all", n_samples=500, wm_loss_ema=1.0,
        wm_loss_ratio=ratio, body_brier_ema=0.1, body_brier_ratio=1.0,
        forecaster_nmse_ema=float("nan"), reward_rate_ema=0.5,
        reward_ratio=1.0, learning_progress=progress,
        adaptation_status=status, replay_coverage=coverage,
    )


def report(regions: list[RegionCompetence]) -> CompetenceReport:
    return CompetenceReport(schema_version=REPORT_SCHEMA_VERSION, tick=10_000,
                            run_id="fixture", regions=regions)


BENIGN_SCALARS = {
    "reward/rollout": [1.0 + 0.01 * i for i in range(40)],  # healthy climb
    "rssm/kl": [2.5 + 0.1 * (i % 3) for i in range(15)],    # off the floor
    "clip_frac": [0.05] * 15,
    "sleep/grad_steps": [100.0] * 8,                        # budget met
    "memory/stale_trip_rate_per_1k": [10.0] * 8,            # low
    "ledger/reliability_ece": [0.05] * 15,                  # flat, low
    "rssm/participation_ratio": [8.0, 8.5, 8.2, 8.4, 8.1],  # healthy
    "loss/total": [1.0 - 0.001 * i for i in range(40)],     # smooth
}


def benign_evidence() -> Evidence:
    ev = evidence("ledger", **BENIGN_SCALARS)
    ev.competence = report([_region("stable", 1.0, 0.1, 0.0)])
    return ev


ALL_CASES = [
    (RetrainingGenerator, lambda: _with_competence(
        [_region("degrading", 2.0, 0.001, -0.01)])),
    (CurriculumGenerator, lambda: _with_competence(
        [_region("mid-adaptation", 1.6, 0.05, 0.05, region=(0, 0)),
         _region("mid-adaptation", 1.7, 0.05, 0.04, region=(0, 1))])),
    (HyperparameterGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS, "rssm/kl": [1.0] * 15})),
    (MemoryPolicyGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS,
                     "memory/stale_trip_rate_per_1k": [120.0] * 8})),
    (CheckpointScheduleGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS,
                     "rssm/participation_ratio": [8.0, 8.5, 1.0, 8.2, 8.1]})),
    (EvaluationGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS,
                     "ledger/reliability_ece": [0.05 + 0.01 * i
                                                for i in range(15)]})),
    (LoggingChangeGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS,
                     "reward/rollout": list(np.random.default_rng(0)
                                            .normal(0.01, 1.0, 40))})),
    (ComputeBudgetGenerator, lambda: evidence(
        "ledger", **{**BENIGN_SCALARS, "sleep/grad_steps": [30.0] * 8})),
]


def _with_competence(regions: list[RegionCompetence]) -> Evidence:
    ev = evidence("ledger", **BENIGN_SCALARS)
    ev.competence = report(regions)
    return ev


@pytest.mark.parametrize("gen_cls,trigger", ALL_CASES,
                         ids=[g.__name__ for g, _ in ALL_CASES])
def test_generator_fires_on_trigger_and_is_silent_on_benign(gen_cls, trigger) -> None:
    gen = gen_cls()
    fired = gen.generate(trigger())
    assert fired is not None, f"{gen.name} did not fire on its trigger"
    assert fired.type == gen.ptype
    assert fired.supporting_observations, f"{gen.name} fired without evidence refs"

    silent = gen.generate(benign_evidence())
    assert silent is None, f"{gen.name} fired on benign evidence"


@pytest.mark.parametrize("stub_cls", [DatasetExtensionGenerator,
                                      VisualizationGenerator])
def test_stub_generators_never_fire(stub_cls) -> None:
    assert stub_cls().generate(benign_evidence()) is None


def test_knob_generators_carry_machine_readable_change() -> None:
    """Every config-knob proposal must carry proposed_change (the A/B
    evaluation path depends on it — a silent None here degrades the
    flagship battery to not_executed records)."""
    hp = HyperparameterGenerator().generate(evidence(
        "ledger", **{**BENIGN_SCALARS, "rssm/kl": [1.0] * 15}))
    assert hp is not None
    assert hp.proposed_change == {"config_path": "rssm.free_nats",
                                  "new_value": 0.5}
    cb = ComputeBudgetGenerator().generate(evidence(
        "ledger", **{**BENIGN_SCALARS, "sleep/grad_steps": [30.0] * 8}))
    assert cb is not None and cb.proposed_change["config_path"] == "rssm.sleep_grad_steps"
    ck = CheckpointScheduleGenerator().generate(evidence(
        "ledger", **{**BENIGN_SCALARS,
                     "rssm/participation_ratio": [8.0, 8.5, 1.0, 8.2, 8.1]}))
    assert ck is not None and ck.proposed_change == {
        "config_path": "checkpoints.interval", "new_value": 25_000}
