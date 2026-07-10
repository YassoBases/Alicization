# Architecture Evolution Report (Phase 1)

Written as Principal Research Scientist + Systems Architect, reading the
Assumption Registry (`lab/assumptions.yaml`, rendered to
[docs/assumptions.md](assumptions.md)) against the empirical record under
`docs/acceptance/` and `experiments/results/`. The objective is not to
defend the current architecture but to turn this repository into a platform
that continuously generates, tests, compares, and retires architectural
candidates. Negative results carry the same weight as positive ones.

> **Directive gap flagged up front (affects Gate P1 and everything after).**
> The directive references "Stages W, I, M, T, L, U, Z below" as
> pre-approved and specified, but **their specifications were not included in
> the message.** The stage intents are inferable from the hints (Stage W
> "modifies world/engine.py, sleep.py, drives.py, reliability.py" and adds
> "fatigue degradation / hydration / cache dynamics"; Stage I5 = `compare.py`;
> Stage T = the causal self-theory; Stage L = comparison-harness glue; Stage Z
> reads `ANTHROPIC_API_KEY`), but the numbered tasks and acceptance criteria
> are absent. The stage assessment below is therefore at the level of
> **intent**, and I **cannot implement any stage "as specified" without its
> spec.** Per the standing rule ("do not interpret silence as approval"),
> Gate P1 stops here and requests the W–Z specifications before Stage W.

---

## 1. Reading the registry

### Strongest assumptions (keep, use as anchors)

- **`ledger-forecaster`** (0.75, *supported*). The clearest positive in the
  repo: NMSE(k=10) 0.78 at 50k
  (`docs/acceptance/stage-4c/forecaster_report.md`) and ~0.5 across 3 seeds
  by 200k, crossing below identity between 63k–113k
  (`docs/acceptance/stage-A/scale_curves_fullscale/README.md`). Self-
  prediction of interoceptive state genuinely works once past viable scale.
- **`ledger-mirror`** (0.75, *supported*). The kidnapped-agent test spikes
  within 20 ticks every seed and roughly halves relocalisation (18.3 vs 34.3
  ablation) — `docs/acceptance/stage-6a/kidnapped_report.md`. A self-
  consistency monitor that measurably earns its keep.
- **`researcher-monitors`** (0.75, *supported*). CUSUM closes mean_shift's
  onset-only blind spot: caught in 3/3 seeds where mean_shift never
  escalates, zero pre-lever false alarms
  (`docs/acceptance/stage-B/README.md`). Notably this is an *outside-AI*
  import (statistical process control) that worked — a template for §4.
- **`gradient-isolation`** (0.95, *supported*, **protected**). Structurally
  tested, extends verbatim to SelfQ. Recorded but not a replaceable
  hypothesis (constitutional boundary).

### Weakest assumptions (the actionable core of this report)

- **`memory-reliability-weighting`** (*unsupported*, 0.85 confidence in the
  null). Stale-trip rate WITH reliability is 87/1k versus the ablation's
  28/1k, and the fitted decay curves do not differ by region
  (`docs/acceptance/stage-5b/reliability_report.md`); reconfirmed at quick
  scale (`experiments/results/20260708-1311/ANALYSIS.md`). Two scales agree.
  The learned head is a zero-init logistic regression that barely moves, and
  the world's ~75-tick food turnover means a remembered location is usually
  dead on arrival. **This is the first removal candidate.**
- **`ledger-attribution`** (*unsupported*, 0.7). Bimodal and seed-fragile —
  0.99 or a 0.00 collapse at every budget including 200k, and even when it
  converges it only *matches* the trivial always-SELF baseline (~0.99),
  never beats it (`docs/acceptance/stage-A/scale_curves_fullscale/README.md`).
  As built, it does not earn its keep and its training is unstable.

### Components that appear unnecessary

1. **The reliability head + its "inspect" drive** — confirmed null, twice.
   Remove or replace with a fixed age/volatility decay (no learned model).
2. **Attribution in the policy path** — no margin over the baseline and
   unstable; keep at most as an offline evaluation probe until a redesign
   (SelfQ-residual reframing, `TODO.md`) is shown to stabilise it.
3. **Episodic memory itself is *contested*, not yet condemned** — there is
   no committed memory-on vs memory-off A/B on reward. It may be dead weight
   in this food-turnover regime, but that must be *measured*, not assumed.

### Likely bottlenecks

- **Compute / viable scale.** Six of eight battery tests were below minimum
  viable scale at quick, and convergence needs 50k–200k
  (`experiments/results/20260708-1311/ANALYSIS.md`). The MIN_VIABLE_SCALE
  contract makes this explicit, but it means every architecture claim costs
  hours. Efficiency (smaller world, shorter episodes, or a cheaper core) is
  itself a research lever, not just an annoyance.
- **The central question is undecided.** Does feeding Ledger estimates to the
  policy help — `architecture A vs B` (`agent.use_ledger_features`)? The
  whole thesis rests on this and it has *never* been decided at full scale
  (`ledger-body-model` entry). Until it is, several downstream stages build
  on an unproven foundation.
- **Attribution instability** blocks any multi-agent / causal extension that
  depends on it.

### Components deserving competing implementations (the funded seams)

The anti-abstraction rule caps this at **three** contested-with-active-
comparison slots. My recommended three, all of which already have a
config-selectable seam:

1. **`selfq-unified` vs separate heads** (`ledger.impl`) — comparison already
   running (full-scale forecaster parity). Keep as slot 1.
2. **Reliability removal** (`reliability.enabled`) — a proposal with predicted
   effect ≈ 0, straight to review. Slot 2.
3. **Attribution redesign-or-remove** — stabilise (class-balanced loss +
   threshold calibration) or drop from the policy path. Slot 3.

Everything else that *could* be compared (`agent.core` gru/rssm,
`agent.controller` actor/arbiter, circadian vs plain PPO, stoch-on/off) waits
its turn behind these — the laptop budget is the binding constraint.

---

## 2. Assessment of Stages W–Z (intent-level; specs not provided)

Treating each as a hypothesis batch (or, for I/U/L, an instrument judged on
cost/utility). **Recommendations are provisional on the actual specs.**

| stage | inferred intent | recommendation | reason |
|-------|-----------------|----------------|--------|
| **W** | world mechanics: fatigue/hydration degradation, cache dynamics; touches engine/sleep/drives/reliability | **KEEP, but MODIFY** | Directly addresses the strongest gap: capabilities are static, so the body model/attribution have little non-trivial to predict. State-dependent degradation gives the self-model a real target. **Tension:** W is said to modify `reliability.py`, which §1 flags for *removal* — resolve which before touching it. **Guard:** every new mechanic must (a) leave lever-labelled ground-truth only, and (b) ship a falsifiable test (does a self-model that sees degradation beat one blind to it?), or it is complexity for its own sake. |
| **I** | comparison infrastructure (`compare.py`, competing arms) | **KEEP** | Instrument, high utility. This is the missing piece that makes every contested registry entry testable head-to-head via the ladder. Judge on cost/utility, not evidence — and utility is high. |
| **M** | *unknown* | **CANNOT ASSESS** | Intent not inferable from the directive. Needs its spec. |
| **T** | causal self-theory | **MODIFY / DEFER** | The *associative* self-model (body model) has not yet been shown to help the policy at scale (`ledger-body-model`, undecided A vs B). Building an interventional causal theory on top of an unproven associative layer is novelty before foundation. Recommend gating Stage T on a positive `architecture A vs B` at full scale; otherwise it risks being speculative complexity. Its falsifying experiment (does the causal theory beat the associative body model on interventional prediction?) must be in the spec. |
| **L** | laboratory comparison glue (registry + compare.py + ladder) | **KEEP** | Instrument. The glue that turns §5's redesign into a loop. High utility, low risk. |
| **U** | *unknown* (instrument) | **CANNOT ASSESS** | Intent not inferable. Needs its spec. |
| **Z** | LLM-online (reads `ANTHROPIC_API_KEY`) — Architect/lab online | **KEEP** | The Architect is *untested* (offline it drafts nothing; `docs/acceptance/stage-D/archbench.md`). Stage Z is the only way to actually measure it. Keep, but its value is unknown until it runs against ARCH-bench online — treat the first run as a measurement, not a deliverable. |

No stage looks *actively harmful* at the intent level, so none triggers the
"STOP and ask" clause on that ground. But two (M, U) are unassessable and one
(T) I recommend re-sequencing — all of which require the specs.

---

## 3. New research directions (each with a falsifying experiment + cost)

Justified by observed failures in the record, not invented for novelty.

1. **Decide the thesis: `architecture A vs B` at full scale.** The one
   experiment that most changes what everything else means. Falsify "Ledger
   features help the policy" by running the capability-shift battery A
   (features on) vs B (withheld) at full scale. *Cost:* the battery already
   exists; ~overnight, resumable. **This should run before Stages T and
   beyond.**
2. **Stabilise attribution or drop it.** Hypothesis: the bimodal collapse is
   a loss-balance / pseudo-label-threshold pathology, not a capacity limit.
   Falsify by adding class-balanced (focal) loss + threshold calibration and
   measuring seed-variance of accuracy at 200k. *Cost:* 3 seeds × 200k on the
   ghost lever, resumable, ~2–3h.
3. **Match memory to the world's timescale, or remove it.** Hypothesis: a
   store whose horizon is tuned to the ~75-tick food turnover beats both the
   current store and no store. Falsify with a 3-way memory-on / tuned-decay /
   memory-off A/B on stale-trip rate + reward at full scale. *Cost:* ~3
   variants × 5 seeds, resumable.
4. **Is consolidation worth its compute?** Full-scale sleep-ablation
   (wake+sleep vs wake-only) with ≥100 grad-steps/window — the smoke result
   was null (`ledger-consolidation-imagination`). *Cost:* 2 arms × 5 seeds
   full scale, overnight.

Each direction, if it produces a removal/replacement, becomes a proposal
through the existing review queue — the loop in §6.

## 4. Ideas worth integrating (each labeled by maturity + a kill criterion)

**From modern AI:**
- *Discrete/categorical latent (DreamerV3-style)* — **established**. Could
  stabilise the world model and the epistemic signal. **Kill if** stoch-off
  or the current continuous latent matches forecaster NMSE / EIG calibration.
- *JEPA-style non-generative world model* — **underexplored**. Predict in
  latent space rather than reconstructing pixels/grid; may localise better.
  **Kill if** the recon-based RSSM matches on downstream forecaster + pose.
- *Successor features for the forecaster* — **adaptation**. **Kill if** the
  current per-plan Gaussian forecaster matches NMSE at k=10,100.

**From outside the AI literature** (the CUSUM import is the proof this pays
off — `researcher-monitors`):
- *Survival analysis / hazard functions for memory staleness*
  — **adaptation**. Model a memory's probability of still-valid as a hazard
  rate; principled where the logistic reliability head failed. **Kill if** it
  does not beat a fixed age/volatility decay on stale-trip rate.
- *Active-inference free-energy framing for the drives* — **speculative**.
  Recast homeostatic setpoints + exploration as expected-free-energy
  minimisation. **Kill if** it does not beat the current homeostatic +
  epistemic terms on coverage/adaptation. (High risk; only after §3.1.)
- *Sequential-analysis stopping rules for the evaluation ladder*
  — **established** (SPRT). Stop a smoke A/B as soon as the arms separate
  (or provably won't), saving compute. **Kill if** it changes ladder verdicts
  vs fixed-N.

## 5. Repository redesign — making evolution first-class

The good news: **the seams already exist.** The repo is unusually well-set-up
for architecture evolution because contested assumptions were already made
config-selectable as they arose (`ledger.impl`, `agent.core`,
`agent.controller`, `researcher.monitors.capability_test`,
`reliability.enabled`, `agent.use_ledger_features`). The anti-abstraction
rule is already the house style — SelfQ's seam appeared *when SelfQ existed*.

The minimal additions to close the loop (the `lab/` layer):

1. **The registry** (done) — the machine-readable status of every assumption,
   with a citation test so claims can't rot silently.
2. **`lab/compare.py`** (Stage I/L) — take two config variants that differ on
   one seam, run both through the existing evaluation ladder (smoke screen →
   quick → full), and emit a head-to-head using the ladder's A/B +
   independent-metric + degenerate-control machinery already built in
   `experiments/runner.py`. Most candidates should die at smoke.
3. **A thin registry↔seam map** — each contested entry names its config
   seam and its candidate; `compare.py` reads it. No new interfaces beyond
   what a *funded* candidate requires (≤3 at a time).
4. **Removals are proposals** — a removal (reliability, attribution-from-
   policy) is an intervention with a predicted effect, through `review/`,
   executed by the human. The constitution already rejects proposals that
   touch the safety files.

**Migration path (never breaks `pytest -q -m "not slow"`):** every step is
additive — the registry and `compare.py` are new files; the seams already
exist; removals go through review, not direct deletion. No existing test
changes behaviour.

## 6. Roadmap — the evolution loop

```
 registry (lab/assumptions.yaml)
    │  flags: unsupported / contested entries + funded candidate
    ▼
 lab/compare.py  ──►  evaluation ladder (smoke → quick → full)
    │                    │  A/B + independent metrics + degenerate guard
    │                    ▼
    │             result updates the entry's status + confidence
    ▼
 removal / replacement  ──►  review queue (blind, human gate)
                               │  constitution rejects safety-file targets
                               ▼
                          human executes  ──►  re-run acceptance, commit, tag
```

Properties preserved at every turn:
- **Reproducibility** — provenance (evidence-bundle hashes, git sha, prompt
  hashes) + resumable sweeps + per-stage acceptance evidence.
- **Safety** — gradient isolation, the exogenous-lever/ground-truth-only
  rule, the constitution, and the human gate are constitutional and out of
  the lab's scope to question.
- **Honesty** — the MIN_VIABLE_SCALE contract keeps below-scale numbers
  machinery-only, and nulls are recorded as prominently as positives.

The end state is a repository where the *default activity* is: pick the
lowest-confidence contested assumption with a funded candidate, run the
comparison through the ladder, update the registry, and route any
removal/replacement through the human gate — continuously, cheaply, and
without ever loosening the safety constraints or the reproducibility trail.

## Gate P1 status

- Registry validates: schema + citation test green
  (`tests/test_assumptions_registry.py`, 6 passed), 23 assumptions,
  `docs/assumptions.md` rendered.
- This report is committed.
- **W–Z recommendations recorded (§2).** Two stages (M, U) are unassessable
  and one (T) I recommend re-sequencing — all because **the W–Z specs were
  not in the directive.** Per the standing rules I am stopping at Gate P1 to
  request those specifications before implementing Stage W, rather than
  inventing them.
