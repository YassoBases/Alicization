# Assumption Registry

Auto-generated from `lab/assumptions.yaml` by `python -m lab.render` — do not edit by hand. Every subsystem is a scientific hypothesis, not a permanent design; this is the empirical status of each.

| status | count |
|--------|-------|
| supported | 7 |
| contested | 7 |
| unsupported | 2 |
| untested | 7 |

| id | component | status | confidence |
|----|-----------|--------|------------|
| [recurrent-core](#recurrent-core) | `agent/core_rssm.py, agent/core_gru.py` | **supported** | 0.7 |
| [latent-representation](#latent-representation) | `agent/core_rssm.py (deter+stoch split)` | **untested** | 0.4 |
| [episodic-memory](#episodic-memory) | `memory/episodic.py` | **contested** | 0.4 |
| [memory-reliability-weighting](#memory-reliability-weighting) | `ledger/reliability.py` | **unsupported** | 0.85 |
| [consolidation-imagination](#consolidation-imagination) | `training/sleep.py (Dreamer-style imagination in sleep)` | **contested** | 0.45 |
| [wake-sleep-scheduling](#wake-sleep-scheduling) | `training/sleep.py (is_sleep_tick)` | **untested** | 0.6 |
| [drives-intrinsic](#drives-intrinsic) | `agent/drives.py, training/reward.py` | **untested** | 0.5 |
| [planning-arbiter](#planning-arbiter) | `agent/drives.py (macro-plan arbiter), ledger/forecaster.py` | **contested** | 0.4 |
| [ledger-body-model](#ledger-body-model) | `ledger/body_model.py` | **supported** | 0.65 |
| [ledger-attribution](#ledger-attribution) | `ledger/attribution.py` | **unsupported** | 0.7 |
| [ledger-forecaster](#ledger-forecaster) | `ledger/forecaster.py` | **supported** | 0.75 |
| [ledger-mirror](#ledger-mirror) | `ledger/mirror.py` | **supported** | 0.75 |
| [ledger-competence](#ledger-competence) | `ledger/competence.py` | **supported** | 0.65 |
| [selfq-unified](#selfq-unified) | `selfq/model.py, selfq/adapters.py` | **contested** | 0.6 |
| [gradient-isolation](#gradient-isolation) | `tests/test_grad_isolation.py, all of ledger/ and selfq/` | **supported** | 0.95 |
| [researcher-monitors](#researcher-monitors) | `researcher/registry.py (cusum_frozen_baseline, mean_shift)` | **supported** | 0.75 |
| [research-agenda-value](#research-agenda-value) | `researcher/agenda.py, researcher/eig.py` | **contested** | 0.55 |
| [proposal-generators-dual-source](#proposal-generators-dual-source) | `proposals/generator.py` | **contested** | 0.5 |
| [architect-instrument](#architect-instrument) | `architect/` | **untested** | 0.35 |
| [training-procedure](#training-procedure) | `training/ppo.py, training/sleep.py` | **contested** | 0.65 |
| [hydration-degradation](#hydration-degradation) | `world/engine.py, ledger/ (to be built in Stage W)` | **untested** | 0.2 |
| [multi-agent-attribution](#multi-agent-attribution) | `world/engine.py (other-agent causes)` | **untested** | 0.15 |
| [causal-self-theory](#causal-self-theory) | `(to be built in Stage T)` | **untested** | 0.2 |

## recurrent-core

- **component**: `agent/core_rssm.py, agent/core_gru.py`
- **status**: supported  |  **confidence**: 0.7 — The core demonstrably supports downstream heads at scale, but no ablation isolates RSSM vs GRU vs a simpler predictor on THIS world.
- **purpose**: A persistent recurrent hidden state h that carries the agent across ticks and is the substrate every self-model head reads.
- **hypothesis**: A learned recurrent state-space model (deter + stoch latent, dynamics ensemble) captures the gridworld's dynamics well enough to (a) drive a competent policy and (b) support self-prediction.
- **success**: wm loss decreases and stays down; downstream heads (forecaster, pose) beat their baselines at >=113k steps.
- **failure (replace/remove when)**: A strictly simpler core (GRU-only, or a fixed featuriser) matches forecaster NMSE and kidnapped relocalisation within noise at full scale.
- **evidence for**: `docs/acceptance/stage-4c/forecaster_report.md`, `docs/acceptance/stage-6a/kidnapped_report.md`, `docs/architecture.md`
- **evidence against**: `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: GRU-only core (already present as agent.core=gru) (established), deterministic latent (drop the stochastic path) (adaptation), structured-state-space / linear-attention core (underexplored)

## latent-representation

- **component**: `agent/core_rssm.py (deter+stoch split)`
- **status**: untested  |  **confidence**: 0.4 — Untested in isolation — no run compares stoch-on vs stoch-off on any downstream metric. The epistemic map is used (researcher EIG) but its dependence on the stochastic path is unmeasured.
- **purpose**: The factorisation of h into a deterministic path and a sampled stochastic latent.
- **hypothesis**: A stochastic latent improves world-model calibration and the ensemble-disagreement epistemic signal versus a purely deterministic state.
- **success**: stoch-on beats stoch-off on forecaster NMSE or on EIG calibration (docs/acceptance/stage-B/researcher_value_cusum.md style measurement).
- **failure (replace/remove when)**: stoch-off matches within noise -> drop the stochastic path.
- **evidence for**: `docs/architecture.md`
- **evidence against**: _none_
- **replacement candidates**: deterministic-only latent (adaptation), discrete/categorical latent (DreamerV3-style) (established)

## episodic-memory

- **component**: `memory/episodic.py`
- **status**: contested  |  **confidence**: 0.4 — No committed A/B isolates memory-on vs memory-off on reward or stale-trip rate; the reliability sub-experiment suggests the world's ~75-tick food turnover limits what any memory can buy.
- **purpose**: A surprise-gated store of past observations with spatial retrieval, fed to the policy as a detached summary.
- **hypothesis**: Remembering surprising food/water locations improves foraging over a memoryless agent.
- **success**: memory-on beats memory-off on reward or stale-trip rate at full scale.
- **failure (replace/remove when)**: no reward/stale-trip difference -> memory is dead weight in this regime.
- **evidence for**: `docs/architecture.md`
- **evidence against**: `docs/acceptance/stage-5b/reliability_report.md`
- **replacement candidates**: no episodic memory (rely on recurrent state) (established), shorter-horizon / decaying store tuned to food-turnover period (adaptation)

## memory-reliability-weighting

- **component**: `ledger/reliability.py`
- **status**: unsupported  |  **confidence**: 0.85 — Two independent scales agree the weighting does not help and the fitted curves are flat/indistinguishable; the model is a zero-init logistic regression that barely moves. Confidence is in the NULL.
- **purpose**: A learned P(memory still matches world) that reweights retrieval and drives an "inspect" plan.
- **hypothesis**: Down-weighting unreliable memories reduces stale trips to dead food locations.
- **success**: stale-trip rate WITH reliability strictly below the ablation at full scale.
- **failure (replace/remove when)**: ALREADY MET (twice). This is the directive's first named removal candidate: it becomes a proposal with predicted effect ~0, reviewed and executed by the human.
- **evidence for**: _none_
- **evidence against**: `docs/acceptance/stage-5b/reliability_report.md`, `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: REMOVE the reliability head + inspect drive (established), replace with a fixed age/volatility decay (no learned model) (adaptation)

## consolidation-imagination

- **component**: `training/sleep.py (Dreamer-style imagination in sleep)`
- **status**: contested  |  **confidence**: 0.45 — Wake+sleep ends slightly ahead of wake-only but CIs overlap at smoke scale; the benefit is unproven pending the full-scale sleep-ablation.
- **purpose**: Sleep-phase actor-critic learning on imagined RSSM rollouts (lambda-returns) plus world-model replay.
- **hypothesis**: Imagination-based consolidation accelerates policy improvement over wake-only PPO.
- **success**: wake+sleep final reward beats wake-only with non-overlapping CI at full scale.
- **failure (replace/remove when)**: no separation at full scale -> imagination is not earning its compute.
- **evidence for**: `docs/acceptance/stage-4c/forecaster_report.md`
- **evidence against**: `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: wake-only PPO (drop imagination) (established), replay-only consolidation (no imagined rollouts) (adaptation)

## wake-sleep-scheduling

- **component**: `training/sleep.py (is_sleep_tick)`
- **status**: untested  |  **confidence**: 0.6 — The purity/safety of the schedule is established and tested; whether THIS cadence is optimal is untested (no cadence sweep).
- **purpose**: An exogenous, step-counter-only schedule that alternates env interaction and consolidation.
- **hypothesis**: A fixed circadian schedule is a sufficient and safe consolidation trigger.
- **success**: safety signature test stays green; consolidation shows value (ties to consolidation-imagination).
- **failure (replace/remove when)**: a different cadence materially changes outcomes -> the fixed cadence is a hidden hyperparameter.
- **evidence for**: `docs/safety_scope.md`
- **evidence against**: `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: cadence sweep (sleep_every) (adaptation)

## drives-intrinsic

- **component**: `agent/drives.py, training/reward.py`
- **status**: untested  |  **confidence**: 0.5 — The agent clearly acts and survives, but no ablation isolates the intrinsic/epistemic terms' contribution to any downstream capability.
- **purpose**: Homeostatic task terms (energy/fatigue setpoints) plus an epistemic-exploration plan.
- **hypothesis**: Homeostatic drives + epistemic exploration produce competent, self-maintaining behaviour without an extrinsic score.
- **success**: removing the epistemic drive measurably reduces coverage or adaptation speed.
- **failure (replace/remove when)**: removing it changes nothing -> the intrinsic term is inert.
- **evidence for**: `docs/architecture.md`, `experiments/results/20260708-1311/ANALYSIS.md`
- **evidence against**: _none_
- **replacement candidates**: reward-only (drop intrinsic/epistemic) (established), count-based or RND-style novelty bonus (established)

## planning-arbiter

- **component**: `agent/drives.py (macro-plan arbiter), ledger/forecaster.py`
- **status**: contested  |  **confidence**: 0.4 — No head-to-head of arbiter-controller vs actor-controller on reward is committed; the forecaster it depends on only works past ~113k.
- **purpose**: An epsilon-greedy arbiter selecting macro-plans scored by the forecaster.
- **hypothesis**: Forecaster-scored macro-plan selection outperforms the raw reactive actor.
- **success**: arbiter controller beats actor controller on reward/adaptation at full scale.
- **failure (replace/remove when)**: actor-only matches -> the arbiter is unnecessary overhead.
- **evidence for**: `docs/acceptance/stage-4c/forecaster_report.md`
- **evidence against**: `experiments/results/20260708-2141/researcher_value/ANALYSIS.md`
- **replacement candidates**: actor-only controller (already present as agent.controller=actor) (established), short-horizon MPC through the RSSM prior (adaptation)

## ledger-body-model

- **component**: `ledger/body_model.py`
- **status**: supported  |  **confidence**: 0.65 — The head trains and predicts well (Brier ~0.02); whether feeding it to the policy causally helps (architecture A vs B) is not yet decided at scale.
- **purpose**: Per-action success / dpos / Delta-energy predictions fed (detached) to the policy.
- **hypothesis**: An explicit body model gives the policy useful capability estimates and sharp self-prediction.
- **success**: architecture A (ledger features on) beats B (withheld) on detection/recovery at full scale.
- **failure (replace/remove when)**: A == B at full scale -> the body features do not help the policy (keep head for evaluation, drop from policy input).
- **evidence for**: `docs/acceptance/stage-E/parity.md`
- **evidence against**: `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: use_ledger_features=false (already the B control) (established), fold into SelfQ (already available) (adaptation)

## ledger-attribution

- **component**: `ledger/attribution.py`
- **status**: unsupported  |  **confidence**: 0.7 — Confidence is in the finding that attribution, AS BUILT, does not cleanly earn its keep: unstable training and no margin over the trivial baseline. A stability investigation (loss balance / pseudo-label thresholds) precedes any architecture claim.
- **purpose**: A self / world / both classifier over body-model residuals, labelling the cause of each transition.
- **hypothesis**: The agent can learn to attribute state changes to itself vs the world.
- **success**: accuracy beats the always-SELF majority by a stable margin across all seeds.
- **failure (replace/remove when)**: ALREADY MET (seed-fragile, no margin) -> redesign (SelfQ-residual reframing, TODO.md) or remove from the policy path.
- **evidence for**: `docs/acceptance/stage-A/scale_curves_fullscale/README.md`
- **evidence against**: `docs/acceptance/stage-A/scale_curves_fullscale/README.md`, `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: SelfQ-residual attribution (deferred spec (adaptation), class-balanced / focal loss + threshold calibration (established), REMOVE (keep only as an evaluation probe) (established)

## ledger-forecaster

- **component**: `ledger/forecaster.py`
- **status**: supported  |  **confidence**: 0.75 — Solidly beats identity at k=10 once past ~113k, across seeds — one of the strongest positive results. k=1 is a degenerate metric, not a failure.
- **purpose**: K-step interoceptive forecasts per macro-plan, scored against the identity baseline.
- **hypothesis**: The agent can predict its own future interoceptive state better than "nothing changes".
- **success**: NMSE(k=10) < 1 across seeds at full scale. MET.
- **failure (replace/remove when)**: regresses above identity at full scale.
- **evidence for**: `docs/acceptance/stage-4c/forecaster_report.md`, `docs/acceptance/stage-A/scale_curves_fullscale/README.md`
- **evidence against**: `docs/acceptance/stage-E/parity.md`
- **replacement candidates**: fold into SelfQ (contested via ledger.impl) (adaptation)

## ledger-mirror

- **component**: `ledger/mirror.py`
- **status**: supported  |  **confidence**: 0.75 — Clear positive: the spike criterion fires reliably and the mirror responses roughly halve relocalisation time versus the monitor-only ablation. Depends on a well-calibrated pose head (pose_scale, grad-steps).
- **purpose**: Divergence between the decoder-implied and body-model-implied self-position; a monitor + probe trigger, never a loss.
- **hypothesis**: A self-consistency monitor detects localisation failure (kidnapping) and speeds relocalisation.
- **success**: spike < 20 ticks all seeds; mirror relocalisation < ablation. MET at acceptance scale.
- **failure (replace/remove when)**: no spike, or mirror >= ablation relocalisation, at full scale across 5 seeds.
- **evidence for**: `docs/acceptance/stage-6a/kidnapped_report.md`
- **evidence against**: _none_
- **replacement candidates**: keep (best-evidenced self-model component) (established)

## ledger-competence

- **component**: `ledger/competence.py`
- **status**: supported  |  **confidence**: 0.65 — Works as a descriptive instrument (feeds proposals/researcher); its flags are noisy per-region and their downstream value depends on the proposal/researcher layers, which are themselves contested.
- **purpose**: Per-region rolling self-assessment (wm loss, body Brier, forecaster NMSE) with adaptation-status flags.
- **hypothesis**: A per-region competence tracker surfaces where the agent is degrading vs adapting.
- **success**: adaptation flags correlate with injected volatility/shifts.
- **failure (replace/remove when)**: flags are pure noise vs the lever log.
- **evidence for**: `docs/acceptance/stage-7a/competence_report.md`
- **evidence against**: _none_
- **replacement candidates**: simpler global competence (drop per-region) (adaptation)

## selfq-unified

- **component**: `selfq/model.py, selfq/adapters.py`
- **status**: contested  |  **confidence**: 0.6 — Body parity is real; the forecaster gap is a documented shared-base starvation and its full-scale resolution is running now. This is the canonical contested-with-active-comparison entry (the ledger.impl seam).
- **purpose**: One conditional model replacing the body model + forecaster behind adapters (ledger.impl=selfq).
- **hypothesis**: A single conditioned self-model matches or beats the separate heads while unifying the interface.
- **success**: matches heads on body CE/Brier AND forecaster NMSE(k=10) at full scale.
- **failure (replace/remove when)**: forecaster NMSE stays materially worse at full scale -> keep separate heads (or fix update-budget balance).
- **evidence for**: `docs/acceptance/stage-E/parity.md`, `docs/acceptance/stage-E/README.md`
- **evidence against**: `docs/acceptance/stage-E/parity.md`
- **replacement candidates**: separate heads (ledger.impl=heads (established), balance wake/sleep update budgets on the shared base (adaptation), larger shared base / per-task learning rates (adaptation)

## gradient-isolation

- **component**: `tests/test_grad_isolation.py, all of ledger/ and selfq/`
- **status**: supported  |  **confidence**: 0.95 — A HARD RULE and safety invariant, structurally tested. Recorded for completeness; NOT a replaceable hypothesis — failure_criteria never authorise removal (constitutional boundary).
- **purpose**: Every self-model head consumes h.detach(); its losses never reach the core.
- **hypothesis**: Isolating self-model gradients keeps the world model honest and the self-model a genuine (non-self-fulfilling) predictor.
- **success**: structural test stays green; self-model heads never move core params.
- **failure (replace/remove when)**: n/a — constitutional. A leak is a bug to fix, never a design to drop.
- **evidence for**: `CLAUDE.md`, `docs/safety_scope.md`, `docs/acceptance/stage-E/README.md`
- **evidence against**: _none_
- **replacement candidates**: PROTECTED — no replacement candidates (hard rule) (established)

## researcher-monitors

- **component**: `researcher/registry.py (cusum_frozen_baseline, mean_shift)`
- **status**: supported  |  **confidence**: 0.75 — CUSUM demonstrably closes mean_shift's onset-only blind spot; the win is a monitor-quality result, independent of whether the agenda it feeds is valuable.
- **purpose**: Sequential change detectors that drive hypotheses supported->weakening->contradicted.
- **hypothesis**: A frozen-baseline CUSUM detects persistent capability change that the sliding mean_shift misses.
- **success**: detects a blind-spot lever within budget, zero pre-lever false alarms. MET.
- **failure (replace/remove when)**: misses persistent change or false-alarms pre-lever at full scale.
- **evidence for**: `docs/acceptance/stage-B/README.md`, `docs/acceptance/stage-8a/README.md`
- **evidence against**: `docs/acceptance/stage-B/README.md`
- **replacement candidates**: mean_shift ablation (kept (established), Bayesian online change-point detection (underexplored)

## research-agenda-value

- **component**: `researcher/agenda.py, researcher/eig.py`
- **status**: contested  |  **confidence**: 0.55 — The agenda ranks sensibly and the noisy-TV guard holds, but its central value claim is an honest null at smoke scale; EIG predicted_gain does not yet track realised reduction.
- **purpose**: A ranked agenda of experiments (value x tractability x novelty / cost, or EIG v2) the researcher recommends.
- **hypothesis**: Following the ranked agenda reduces the agent's uncertainty faster than random or greedy selection.
- **success**: agenda arm beats random/greedy on drift-corrected uncertainty reduction at full scale; EIG Spearman > 0.
- **failure (replace/remove when)**: no separation at full scale -> the ranking adds no value over cheaper selection.
- **evidence for**: `docs/acceptance/stage-8b/README.md`
- **evidence against**: `experiments/results/20260708-2141/researcher_value/ANALYSIS.md`
- **replacement candidates**: v1 heuristic ranker (kept as ablation) (established), greedy raw-uncertainty (the control arm) (established)

## proposal-generators-dual-source

- **component**: `proposals/generator.py`
- **status**: contested  |  **confidence**: 0.5 — The comparison exists and is honest, but there is no logs_only sample to compare against on healthy runs; the ledger's advantage is sensitivity, not yet demonstrated benefit.
- **purpose**: Rule generators run on ledger evidence AND a logs-only control, feeding the blind-review queue.
- **hypothesis**: Ledger-sourced proposals outperform logs-only-sourced ones on realised benefit and calibration.
- **success**: ledger arm beats logs_only on realised benefit with both arms non-empty at full scale.
- **failure (replace/remove when)**: logs_only stays empty or ledger shows no benefit edge -> the dual-source claim is undecidable/false.
- **evidence for**: `docs/acceptance/stage-C/proposal_quality_v2_summary.md`
- **evidence against**: `experiments/results/20260708-1808/proposal_quality/ANALYSIS.md`
- **replacement candidates**: keep as an honest control (report the null) (established), adversarial/unhealthy-run battery to make logs_only speak (adaptation)

## architect-instrument

- **component**: `architect/`
- **status**: untested  |  **confidence**: 0.35 — An instrument whose capability is UNMEASURED: offline it produces nothing, and the online arm (needs ANTHROPIC_API_KEY, Stage Z) has not run. archbench's control arm shows the bar to beat (symptom-without-cause).
- **purpose**: Experimenter-side analysis + LLM drafting + self-critique that proposes (never applies) repository changes.
- **hypothesis**: An LLM Architect localises injected flaws better than the deterministic rule generators.
- **success**: online architect arm localises >= the rules arm across the flaw battery, with a low clean-control false-positive rate.
- **failure (replace/remove when)**: online architect no better than rules (or worse false-positive rate) at smoke scale.
- **evidence for**: `docs/acceptance/stage-D/README.md`
- **evidence against**: `docs/acceptance/stage-D/archbench.md`
- **replacement candidates**: rule generators only (the control arm) (established), retrieval-augmented drafting over the analysis report (underexplored)

## training-procedure

- **component**: `training/ppo.py, training/sleep.py`
- **status**: contested  |  **confidence**: 0.65 — The procedure works but is compute-hungry relative to the laptop budget; most evidence is smoke-scale machinery validation, not architecture evidence (the MIN_VIABLE_SCALE contract makes this explicit).
- **purpose**: Recurrent PPO (wake) plus the circadian imagination trainer.
- **hypothesis**: This training procedure produces a competent policy at laptop scale.
- **success**: full-scale battery produces evidence-stamped (not machinery-only) rows.
- **failure (replace/remove when)**: cannot reach viable scale within the laptop budget -> procedure/efficiency is the bottleneck.
- **evidence for**: `experiments/results/20260708-1311/ANALYSIS.md`
- **evidence against**: `experiments/results/20260708-1311/ANALYSIS.md`
- **replacement candidates**: plain PPO (drop circadian) where sleep shows no value (established), smaller world / shorter episodes to reach viable scale cheaper (adaptation)

## hydration-degradation

- **component**: `world/engine.py, ledger/ (to be built in Stage W)`
- **status**: untested  |  **confidence**: 0.2 — Untested placeholder. NOTE: the Stage-W specification was NOT included in this engagement's directive (see docs/evolution_report.md, Gate P1).
- **purpose**: PLACEHOLDER for Stage W: fatigue/hydration degradation mechanics that make capability state-dependent (a richer self-model target).
- **hypothesis**: State-dependent capability degradation gives the body model / SelfQ a non-trivial thing to predict and a reason for self-modelling to pay off.
- **success**: (to be defined by the Stage-W spec) degradation is predictable and lever-labelled ground-truth only.
- **failure (replace/remove when)**: degradation is unpredictable noise, or a self-model that ignores it does just as well.
- **evidence for**: _none_
- **evidence against**: _none_
- **replacement candidates**: static capabilities (status quo) (established)

## multi-agent-attribution

- **component**: `world/engine.py (other-agent causes)`
- **status**: untested  |  **confidence**: 0.15 — Untested placeholder; the current engine is single-agent by design. Spec not provided in the directive.
- **purpose**: PLACEHOLDER for a multi-agent extension where another agent is a distinct cause class for attribution.
- **hypothesis**: A second agent gives attribution a genuinely three-way (self/world/other) problem worth solving.
- **success**: attribution distinguishes other-agent causes above chance.
- **failure (replace/remove when)**: the other-agent class is indistinguishable from world.
- **evidence for**: _none_
- **evidence against**: `docs/safety_scope.md`
- **replacement candidates**: single-agent world (status quo) (established)

## causal-self-theory

- **component**: `(to be built in Stage T)`
- **status**: untested  |  **confidence**: 0.2 — Untested placeholder. NOTE: the Stage-T specification was NOT included in this engagement's directive (see docs/evolution_report.md, Gate P1).
- **purpose**: PLACEHOLDER for Stage T — an explicit causal model of the agent's own action->outcome structure.
- **hypothesis**: An explicit, testable causal self-theory (interventional, not just predictive) lets the agent reason about capability change it has not yet observed.
- **success**: (to be defined by the Stage-T spec) the causal theory makes interventional predictions that beat the body model's associative ones.
- **failure (replace/remove when)**: the causal theory adds no predictive/interventional power over the associative body model.
- **evidence for**: _none_
- **evidence against**: _none_
- **replacement candidates**: associative body model / SelfQ only (status quo) (established)
