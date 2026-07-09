# Stage-E acceptance: SelfQ unified self-model

SelfQ (`selfq/`) replaces the body model + forecaster with one conditional
model over `[h.detach(), intent embedding, learned horizon embedding]`,
behind adapters that preserve the exact BodyModel/Forecaster interfaces.
Selected by `ledger.impl: heads | selfq` (default `heads` — no silent
swap). The mirror and attribution are NOT replaced this stage.

## Gate E evidence

### 1. Parity gate — body parity PASSES (`parity.md`)

`python scripts/verify_selfq.py --seeds 3 --steps 24576` trains the SAME
circadian config under both impls and compares each replaced head's
acceptance metric:

| metric | kind | heads | selfq | selfq/heads | result |
|--------|------|-------|-------|-------------|--------|
| body_ce | gate | 0.269 | 0.288 | 1.07 | PASS |
| body_brier | gate | 0.0198 | 0.0086 | 0.44 | PASS |
| nmse_k1 | descriptive | 1443 | 3046 | 2.11 | n/a (below MVS) |
| nmse_k10 | descriptive | 82.3 | 131 | 1.59 | n/a (below MVS) |

- **Body metrics (gated): SelfQ matches CE (within 7%) and beats Brier
  (2×), stably across 3 seeds.** These feed the policy and the mirror and
  are functional at smoke scale — the primary parity target, and it holds.
- **Forecaster NMSE (descriptive, not gated):** at smoke scale BOTH impls
  sit at NMSE ≫ identity (~80–160×) — the forecaster is non-functional in
  either. Stage A established the forecaster needs ~50k steps / 100
  grad-steps to beat identity, and its `MIN_VIABLE_SCALE` contract makes a
  metric below viable scale machinery-only, not admissible parity evidence.
  So forecaster NMSE is reported (SelfQ ~1.6× the heads' at k=10) but
  **forecaster parity is deferred to the full-scale run** — the exact
  Stage-A precondition Stage E was gated on. (k=1 is additionally
  denominator-degenerate: identity MSE ~6e-5 at one tick.)

### 2. What the first run taught us (and the fix)

The first 3-seed gate FAILED: SelfQ's body-CE was 3× worse and **unstable**
across seeds (0.24 / 0.66 / 1.59 vs the heads' tight 0.18–0.40). Root
cause: a single fully-shared trunk AND one shared Adam optimizer let the
frequent wake body updates and the sleep forecaster updates clobber each
other's representation. Fix (committed): a shared base with **task-specific
branches** (confining interference to the base, giving each head family
private capacity) plus **separate optimizers** over the same params (so the
two tasks' Adam moments stay task-appropriate). Body-CE went 3.09× → 1.07×
and became stable — a real architecture result, not a tuned metric.

The remaining forecaster gap has a clear, documented cause visible in the
per-seed data: an inverse body↔forecaster correlation (the seed with the
best SelfQ body has the worst forecaster), i.e. the shared base is won by
the more-frequent body task, starving the forecaster. Balancing the
wake/sleep update budgets (or a larger base) is the natural full-scale
follow-up; noted in TODO.md.

### 3. Suite green under both impls

- Full not-slow suite (heads default): **410 passed**.
- SelfQ path exercised by `tests/test_selfq.py` (model, adapters, wiring,
  gradient isolation through the REAL trainer) and
  `tests/test_selfq_competence.py` (E3 aggregation); gradient isolation
  extends to SelfQ verbatim (`tests/test_grad_isolation.py`).

## Scale note & the honest caveat

Body parity is confirmed at smoke scale. Forecaster parity is genuinely
undeterminable at smoke scale (the forecaster is pre-convergence in both
impls) and is the full-scale follow-up. Stage E was, per the roadmap, meant
to start only after the human confirmed Stage A's full-scale numbers; it was
started early by explicit request, so this forecaster deferral is expected,
not a surprise — the full-scale battery + scale_curves (docs/acceptance/
stage-A) close it.

Reproduce: `python scripts/verify_selfq.py --seeds 3 --steps 24576`
(full-scale: `--steps 200000`).
