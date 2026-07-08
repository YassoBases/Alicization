"""Rule-based proposal generators, dual-source (ledger vs logs_only).

Every generator is a small class with ``generate(evidence) ->
Proposal | None`` reading ONLY diagnostics (competence reports, Ledger
scalars, raw training logs). The GeneratorSuite runs each generator on BOTH
evidence variants every time — the control condition this stage's science
rests on — dedups by hash(type, target), rate-limits per type, and logs
every FIRED and SUPPRESSED decision to
runs/<id>/proposals/generator_decisions.jsonl.

Confidence starts heuristic (each generator's own estimate). Once >= 20
proposals carry realized_benefit, ``recalibrate_confidence`` maps heuristic
confidence through binned historical hit rates during sleep.

Proposals are DATA, never code: nothing here executes anything, writes are
confined to runs/<id>/proposals/ (schema.save_proposal validates).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from proposals.evidence import Evidence
from proposals.schema import Proposal, proposals_dir, save_proposal


def _slope(values: np.ndarray, tail: int = 20) -> float:
    v = values[-tail:]
    if len(v) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(v), dtype=float), v, 1)[0])


@dataclass
class GeneratorResult:
    generator: str
    source: str
    fired: bool
    reason: str
    proposal_id: str | None = None


class BaseGenerator:
    """One rule. Subclasses set ``name``/``ptype`` and implement
    ``generate``. ``min_interval_ticks`` rate-limits per (type, source)."""

    name = "base"
    ptype = "evaluation"
    min_interval_ticks = 5_000

    def generate(self, ev: Evidence) -> Proposal | None:  # pragma: no cover
        raise NotImplementedError

    def _proposal(self, ev: Evidence, *, target: str, rationale: str,
                  benefit: dict[str, Any], confidence: float,
                  observations: list[str], cost: dict[str, float],
                  risks: list[str], criteria: dict[str, Any]) -> Proposal:
        return Proposal.new(
            type=self.ptype, created_tick=ev.tick, run_id=ev.run_id,
            source=ev.source, rationale=rationale, expected_benefit=benefit,
            confidence=confidence, supporting_observations=observations,
            estimated_cost=cost, risks=risks, success_criteria=criteria,
            target=target,
        )


class RetrainingGenerator(BaseGenerator):
    """Competence drop below trailing best in a region AND thin replay
    coverage there -> targeted retraining/replay top-up."""

    name = "propose_retraining"
    ptype = "retraining"
    drop_ratio = 1.5
    coverage_thresh = 0.02

    def generate(self, ev: Evidence) -> Proposal | None:
        if ev.competence is not None:  # ledger path
            for r in ev.competence.regions:
                if (r.adaptation_status == "degrading"
                        and r.wm_loss_ratio > self.drop_ratio
                        and r.replay_coverage < self.coverage_thresh):
                    target = f"region-{r.region[0]}-{r.region[1]}"
                    recon_tag = ev.first_tag("rssm/recon", "sleep/recon")
                    return self._proposal(
                        ev, target=target,
                        rationale=(
                            f"Competence report @tick {ev.competence.tick}: region "
                            f"{r.region} world-model loss is {r.wm_loss_ratio:.2f}x its "
                            f"trailing best (status={r.adaptation_status}) while replay "
                            f"coverage there is {r.replay_coverage:.3f} < "
                            f"{self.coverage_thresh}. The model degraded where the "
                            f"replay buffer holds almost no data to recover from."
                        ),
                        benefit={"metric": recon_tag, "direction": "down",
                                 "magnitude_estimate": 0.2},
                        confidence=0.6,
                        observations=[
                            f"competence:report-{ev.competence.tick}:region-{r.region}",
                            ev.ref(recon_tag),
                        ],
                        cost={"human_hours": 0.5, "gpu_hours": 2.0},
                        risks=["replay reweighting may slow learning elsewhere"],
                        criteria={"metric": recon_tag, "threshold": -0.1,
                                  "eval_window_ticks": 10_000},
                    )
            return None
        # logs_only path: global reward drop + thin region visitation.
        reward = ev.series("reward/rollout")
        if len(reward) < 20 or ev.positions is None:
            return None
        recent, past = reward[-10:].mean(), reward[:-10].max()
        if past <= 0 or recent > 0.7 * past:
            return None
        counts = np.bincount(
            (ev.positions[:, 0] // 8) * 100 + ev.positions[:, 1] // 8
        )
        thin = counts[counts > 0].min() / max(1, len(ev.positions))
        if thin >= self.coverage_thresh:
            return None
        return self._proposal(
            ev, target="global-reward-drop",
            rationale=(
                f"Reward rolling mean dropped to {recent:.3f} from a peak of "
                f"{past:.3f} (raw logs) while the least-visited region holds only "
                f"{thin:.3%} of recorded positions — retrain/top-up suggested."
            ),
            benefit={"metric": "reward/rollout", "direction": "up",
                     "magnitude_estimate": 0.3 * (past - recent)},
            confidence=0.4,
            observations=[ev.ref("reward/rollout")],
            cost={"human_hours": 0.5, "gpu_hours": 2.0},
            risks=["confounds: reward drop may be a lever, not forgetting"],
            criteria={"metric": "reward/rollout", "threshold": 0.9 * past,
                      "eval_window_ticks": 10_000},
        )


class CurriculumGenerator(BaseGenerator):
    """Regions with sustained high learning progress -> replay weighting."""

    name = "propose_training_schedule"
    ptype = "training_schedule"
    progress_thresh = 0.0

    def generate(self, ev: Evidence) -> Proposal | None:
        if ev.competence is not None:
            hot = [r for r in ev.competence.regions
                   if r.adaptation_status == "mid-adaptation"
                   and r.learning_progress > self.progress_thresh]
            if len(hot) < 2:
                return None
            names = ", ".join(str(r.region) for r in hot[:4])
            return self._proposal(
                ev, target=f"replay-weighting-{len(hot)}-regions",
                rationale=(
                    f"{len(hot)} regions ({names}...) show positive learning "
                    f"progress mid-adaptation — replay weighting toward them "
                    f"should shorten the adaptation window."
                ),
                benefit={"metric": "competence/regions_mid_adaptation",
                         "direction": "down", "magnitude_estimate": len(hot) / 2},
                confidence=0.5,
                observations=[
                    f"competence:report-{ev.competence.tick}:region-{r.region}"
                    for r in hot[:4]
                ],
                cost={"human_hours": 1.0, "gpu_hours": 1.0},
                risks=["overweighting hot regions starves stable ones"],
                criteria={"metric": "competence/regions_mid_adaptation",
                          "threshold": len(hot) / 2, "eval_window_ticks": 20_000},
            )
        # logs_only: rising reward slope with skewed visitation entropy.
        reward = ev.series("reward/rollout")
        if len(reward) < 20 or ev.positions is None or _slope(reward) <= 0:
            return None
        counts = np.bincount(
            (ev.positions[:, 0] // 8) * 100 + ev.positions[:, 1] // 8
        ).astype(float)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        if entropy > 0.75 * np.log(len(counts)):
            return None
        return self._proposal(
            ev, target="visitation-rebalance",
            rationale=(
                f"Reward slope is positive ({_slope(reward):.4f}/rollout) but "
                f"position visitation is skewed (entropy {entropy:.2f} vs uniform "
                f"{np.log(len(counts)):.2f}) — rebalancing exploration may extend "
                f"the gains (raw logs)."
            ),
            benefit={"metric": "reward/rollout", "direction": "up",
                     "magnitude_estimate": 0.1},
            confidence=0.3,
            observations=[ev.ref("reward/rollout")],
            cost={"human_hours": 1.0, "gpu_hours": 1.0},
            risks=["exploration changes can regress a working policy"],
            criteria={"metric": "reward/rollout", "threshold": 0.05,
                      "eval_window_ticks": 20_000},
        )


class HyperparameterGenerator(BaseGenerator):
    """Posterior-collapse indicators -> KL free-nats; plateaued reward with
    high clip_frac -> lr. Conservative: ONE proposal at a time."""

    name = "propose_hyperparameter"
    ptype = "hyperparameter"

    def generate(self, ev: Evidence) -> Proposal | None:
        if ev.competence is not None or ev.source == "ledger":
            # PPO logs the world-model KL as rssm/kl; the circadian trainer
            # as sleep/kl — the collapse indicator must check both.
            kl = ev.series("rssm/kl")
            if len(kl) == 0:
                kl = ev.series("sleep/kl")
            free_nats = ev.config.get("rssm", {}).get("free_nats", 1.0)
            window = min(10, max(4, len(kl)))
            if len(kl) >= 4 and np.allclose(kl[-window:], free_nats, atol=0.05):
                kl_tag = "rssm/kl" if len(ev.series("rssm/kl")) else "sleep/kl"
                return self._proposal(
                    ev, target="rssm.free_nats",
                    rationale=(
                        f"{kl_tag} has sat at the free-nats floor ({free_nats}) "
                        f"for the last {window} updates — the posterior is not "
                        f"using its budget (collapse indicator). Recommend "
                        f"lowering free_nats to {free_nats / 2} and observing "
                        f"kl/recon."
                    ),
                    benefit={"metric": "rssm/recon", "direction": "down",
                             "magnitude_estimate": 0.05},
                    confidence=0.55,
                    observations=[ev.ref(kl_tag)],
                    cost={"human_hours": 0.25, "gpu_hours": 1.0},
                    risks=["KL below floor can destabilize early training"],
                    criteria={"metric": "rssm/kl", "threshold": free_nats + 0.1,
                              "eval_window_ticks": 10_000},
                )
        reward = ev.series("reward/rollout")
        clip = ev.series("clip_frac")
        if (len(reward) >= 20 and len(clip) >= 10
                and abs(_slope(reward)) < 1e-3 and clip[-10:].mean() > 0.2):
            lr = ev.config.get("ppo", {}).get("lr", 3e-4)
            return self._proposal(
                ev, target="ppo.lr",
                rationale=(
                    f"Reward is flat (slope {_slope(reward):.5f}) while clip_frac "
                    f"averages {clip[-10:].mean():.2f} — updates are being clipped "
                    f"hard without progress. Recommend lr {lr} -> {lr / 3:.1e}."
                ),
                benefit={"metric": "reward/rollout", "direction": "up",
                         "magnitude_estimate": 0.1},
                confidence=0.45,
                observations=[ev.ref("reward/rollout"), ev.ref("clip_frac")],
                cost={"human_hours": 0.25, "gpu_hours": 1.0},
                risks=["lower lr slows recovery from genuine shifts"],
                criteria={"metric": "clip_frac", "threshold": 0.15,
                          "eval_window_ticks": 10_000},
            )
        return None


class MemoryPolicyGenerator(BaseGenerator):
    """High stale-trip rate -> reliability-decay / write-gate adjustment."""

    name = "propose_memory_policy"
    ptype = "memory_policy"
    stale_thresh = 50.0

    def generate(self, ev: Evidence) -> Proposal | None:
        stale = ev.series("memory/stale_trip_rate_per_1k")
        if ev.source == "logs_only" or len(stale) == 0:
            return None  # memory scalars are Ledger telemetry: ledger-only rule
        if stale[-1] <= self.stale_thresh:
            return None
        ece = ev.series("ledger/reliability_ece")
        ece_txt = f"; reliability ECE {ece[-1]:.3f}" if len(ece) else ""
        return self._proposal(
            ev, target="memory.write-gate-or-decay",
            rationale=(
                f"Stale-trip rate is {stale[-1]:.1f}/1k ticks (> {self.stale_thresh})"
                f"{ece_txt} — remembered food locations are mostly dead on arrival. "
                f"Recommend faster reliability age-decay (lower age_tau) or a "
                f"stricter write gate so trips chase fresher memories."
            ),
            benefit={"metric": "memory/stale_trip_rate_per_1k",
                     "direction": "down", "magnitude_estimate": 20.0},
            confidence=0.5,
            observations=[ev.ref("memory/stale_trip_rate_per_1k")],
            cost={"human_hours": 0.5, "gpu_hours": 1.0},
            risks=["stricter gating can starve retrieval entirely"],
            criteria={"metric": "memory/stale_trip_rate_per_1k",
                      "threshold": self.stale_thresh, "eval_window_ticks": 20_000},
        )


class CheckpointScheduleGenerator(BaseGenerator):
    """Instability episodes -> shorter checkpoint interval."""

    name = "propose_checkpoint_schedule"
    ptype = "checkpoint_schedule"

    def generate(self, ev: Evidence) -> Proposal | None:
        interval = ev.config.get("checkpoints", {}).get("interval", 50_000)
        if ev.source == "ledger":
            pr = ev.series("rssm/participation_ratio")
            if len(pr) >= 5 and pr.min() < 0.25 * pr.max():
                return self._proposal(
                    ev, target="checkpoints.interval",
                    rationale=(
                        f"Participation ratio dipped to {pr.min():.2f} "
                        f"(max {pr.max():.2f}) — a collapse near-miss. With "
                        f"checkpoints every {interval} steps a collapse could cost "
                        f"the whole interval; recommend {interval // 2}."
                    ),
                    benefit={"metric": "rssm/participation_ratio", "direction": "up",
                             "magnitude_estimate": 0.0},
                    confidence=0.5,
                    observations=[ev.ref("rssm/participation_ratio")],
                    cost={"human_hours": 0.1, "gpu_hours": 0.0},
                    risks=["more checkpoint I/O"],
                    criteria={"metric": "rssm/participation_ratio",
                              "threshold": 0.25 * float(pr.max()),
                              "eval_window_ticks": 20_000},
                )
            return None
        loss = ev.series("loss/total")
        if len(loss) < 20:
            return None  # PPO-only scalar; absent or too short -> no evidence
        has_nan = not np.isfinite(loss).all()
        finite = loss[np.isfinite(loss)]
        jumps = np.abs(np.diff(finite)) if len(finite) > 1 else np.zeros(1)
        if not has_nan and jumps.max() < 6 * (jumps.std() + 1e-9):
            return None
        symptom = ("non-finite loss values (NaN near-miss)" if has_nan else
                   f"spike episodes (max jump {jumps.max():.3f})")
        return self._proposal(
            ev, target="checkpoints.interval",
            rationale=(
                f"loss/total shows {symptom} — instability argues for "
                f"halving checkpoints.interval from {interval} (raw logs)."
            ),
            benefit={"metric": "loss/total", "direction": "down",
                     "magnitude_estimate": 0.0},
            confidence=0.35,
            observations=[ev.ref("loss/total")],
            cost={"human_hours": 0.1, "gpu_hours": 0.0},
            risks=["more checkpoint I/O"],
            criteria={"metric": "loss/total",
                      "threshold": float(np.median(finite)) if len(finite) else 0.0,
                      "eval_window_ticks": 20_000},
        )


class EvaluationGenerator(BaseGenerator):
    """Calibration drift -> recommend a named battery run."""

    name = "propose_evaluation"
    ptype = "evaluation"

    def generate(self, ev: Evidence) -> Proposal | None:
        if ev.source == "ledger":
            ece = ev.series("ledger/reliability_ece")
            if len(ece) >= 10 and _slope(ece, tail=10) > 0.002:
                return self._proposal(
                    ev, target="battery.memory_reliability",
                    rationale=(
                        f"Reliability ECE is drifting up (slope "
                        f"{_slope(ece, tail=10):.4f}/rollout, now {ece[-1]:.3f}) — "
                        f"recommend running the memory_reliability battery to "
                        f"re-baseline calibration."
                    ),
                    benefit={"metric": "ledger/reliability_ece", "direction": "down",
                             "magnitude_estimate": 0.05},
                    confidence=0.5,
                    observations=[ev.ref("ledger/reliability_ece")],
                    cost={"human_hours": 0.25, "gpu_hours": 2.0},
                    risks=[],
                    criteria={"metric": "ledger/reliability_ece", "threshold": 0.1,
                              "eval_window_ticks": 20_000},
                )
            return None
        reward = ev.series("reward/rollout")
        if len(reward) < 30:
            return None
        half = len(reward) // 2
        if reward[half:].std() < 2.0 * reward[:half].std():
            return None
        return self._proposal(
            ev, target="battery.full",
            rationale=(
                f"Reward variance doubled between run halves "
                f"({reward[:half].std():.3f} -> {reward[half:].std():.3f}, raw "
                f"logs) — recommend a full battery pass to locate the source."
            ),
            benefit={"metric": "reward/rollout", "direction": "up",
                     "magnitude_estimate": 0.0},
            confidence=0.3,
            observations=[ev.ref("reward/rollout")],
            cost={"human_hours": 0.5, "gpu_hours": 4.0},
            risks=[],
            criteria={"metric": "reward/rollout", "threshold": 0.0,
                      "eval_window_ticks": 20_000},
        )


class LoggingChangeGenerator(BaseGenerator):
    """High unexplained variance in a headline metric -> a specific scalar."""

    name = "propose_logging_change"
    ptype = "logging_change"

    def generate(self, ev: Evidence) -> Proposal | None:
        reward = ev.series("reward/rollout")
        if len(reward) < 30:
            return None
        cv = reward.std() / (abs(reward.mean()) + 1e-9)
        if cv < 3.0:
            return None
        missing = ("reward/per_region" if ev.source == "ledger"
                   else "reward/per_episode_breakdown")
        return self._proposal(
            ev, target=f"scalar:{missing}",
            rationale=(
                f"reward/rollout coefficient of variation is {cv:.1f} and no "
                f"logged scalar decomposes it — recommend adding `{missing}` so "
                f"the variance source is attributable."
            ),
            benefit={"metric": "reward/rollout", "direction": "up",
                     "magnitude_estimate": 0.0},
            confidence=0.4,
            observations=[ev.ref("reward/rollout")],
            cost={"human_hours": 0.5, "gpu_hours": 0.0},
            risks=["log volume"],
            criteria={"metric": "reward/rollout", "threshold": 0.0,
                      "eval_window_ticks": 10_000},
        )


class ComputeBudgetGenerator(BaseGenerator):
    """Sleep windows ending under budget -> sleep_grad_steps change."""

    name = "propose_compute_budget"
    ptype = "compute_budget"

    def generate(self, ev: Evidence) -> Proposal | None:
        steps = ev.series("sleep/grad_steps")
        budget = ev.config.get("rssm", {}).get("sleep_grad_steps", 200)
        if len(steps) < 3 or steps.mean() >= 0.9 * budget:
            return None
        return self._proposal(
            ev, target="rssm.sleep_grad_steps",
            rationale=(
                f"Sleep windows averaged {steps.mean():.1f} grad steps against a "
                f"budget of {budget} — consolidation is ending early (likely a "
                f"thin replay buffer). Recommend lowering sleep_grad_steps to "
                f"{max(1, int(steps.mean()))} or growing replay before raising it."
            ),
            benefit={"metric": "sleep/wm_total", "direction": "down",
                     "magnitude_estimate": 0.0},
            confidence=0.5,
            observations=[ev.ref("sleep/grad_steps")],
            cost={"human_hours": 0.1, "gpu_hours": 0.0},
            risks=[],
            criteria={"metric": "sleep/grad_steps",
                      "threshold": 0.9 * budget, "eval_window_ticks": 20_000},
        )


class DatasetExtensionGenerator(BaseGenerator):
    """Stub: schema-supported, generator intentionally inert."""

    name = "propose_dataset_extension"
    ptype = "dataset_extension"

    def generate(self, ev: Evidence) -> Proposal | None:
        return None


class VisualizationGenerator(BaseGenerator):
    """Stub: schema-supported, generator intentionally inert."""

    name = "propose_visualization"
    ptype = "visualization"

    def generate(self, ev: Evidence) -> Proposal | None:
        return None


ALL_GENERATORS: tuple[type[BaseGenerator], ...] = (
    RetrainingGenerator, CurriculumGenerator, HyperparameterGenerator,
    MemoryPolicyGenerator, CheckpointScheduleGenerator, EvaluationGenerator,
    LoggingChangeGenerator, ComputeBudgetGenerator, DatasetExtensionGenerator,
    VisualizationGenerator,
)


class GeneratorSuite:
    """Runs every generator on both evidence variants; dedups, rate-limits,
    and logs every decision."""

    def __init__(self, run_dir: str | Path,
                 generators: tuple[type[BaseGenerator], ...] = ALL_GENERATORS) -> None:
        self.run_dir = Path(run_dir)
        self.generators = [g() for g in generators]
        self._seen_hashes: set[str] = set()
        self._last_fired: dict[tuple[str, str], int] = {}
        for existing in self._existing():
            self._seen_hashes.add(existing.dedup_hash())

    def _existing(self) -> list[Proposal]:
        from proposals.schema import load_all

        return load_all(self.run_dir)

    def _log_decision(self, result: GeneratorResult, tick: int) -> None:
        out_dir = proposals_dir(self.run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"tick": tick, "generator": result.generator,
               "source": result.source,
               "decision": "FIRED" if result.fired else "SUPPRESSED",
               "reason": result.reason, "proposal_id": result.proposal_id,
               "timestamp": time.time()}
        with open(out_dir / "generator_decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def run(self, evidence_ledger: Evidence, evidence_logs: Evidence,
            confidence_map: Callable[[float], float] | None = None) -> list[Proposal]:
        fired: list[Proposal] = []
        for ev in (evidence_ledger, evidence_logs):
            for gen in self.generators:
                proposal = gen.generate(ev)
                if proposal is None:
                    self._log_decision(GeneratorResult(
                        gen.name, ev.source, False, "rule not matched"), ev.tick)
                    continue
                key = (proposal.type, ev.source)
                last = self._last_fired.get(key)
                if last is not None and ev.tick - last < gen.min_interval_ticks:
                    self._log_decision(GeneratorResult(
                        gen.name, ev.source, False,
                        f"rate-limited (last fired @{last})"), ev.tick)
                    continue
                if proposal.dedup_hash() in self._seen_hashes:
                    self._log_decision(GeneratorResult(
                        gen.name, ev.source, False,
                        f"duplicate of existing (type={proposal.type}, "
                        f"target={proposal.target})"), ev.tick)
                    continue
                if confidence_map is not None:
                    proposal.confidence = float(
                        np.clip(confidence_map(proposal.confidence), 0.0, 1.0)
                    )
                save_proposal(proposal, self.run_dir)
                self._seen_hashes.add(proposal.dedup_hash())
                self._last_fired[key] = ev.tick
                self._log_decision(GeneratorResult(
                    gen.name, ev.source, True, "rule matched",
                    proposal.id), ev.tick)
                fired.append(proposal)
        return fired


def recalibrate_confidence(history: list[Proposal],
                           bins: int = 5) -> Callable[[float], float] | None:
    """Binned hit-rate mapping from heuristic confidence to realized success.

    Requires >= 20 proposals with realized_benefit; returns None (keep
    heuristics) below that. A proposal 'hit' iff realized_benefit reports
    it met its own success_criteria.
    """
    evaluated = [p for p in history
                 if p.realized_benefit is not None
                 and "met_success_criteria" in p.realized_benefit]
    if len(evaluated) < 20:
        return None
    conf = np.array([p.confidence for p in evaluated])
    hit = np.array([bool(p.realized_benefit["met_success_criteria"])
                    for p in evaluated], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rates: list[tuple[float, float, float]] = []
    for b in range(bins):
        mask = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= 1)
        if mask.any():
            rates.append((edges[b], edges[b + 1], float(hit[mask].mean())))

    def mapper(c: float) -> float:
        for lo, hi, rate in rates:
            if lo <= c < hi or (hi == 1.0 and c == 1.0):
                return rate
        return c

    return mapper
