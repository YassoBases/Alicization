# Stage-E full-scale parity — SelfQ FAILS (final)

The full-scale follow-up the smoke-scale stage-E acceptance deferred to.
`python scripts/verify_selfq.py --seeds 3 --steps 200000` (resumable; ran
across sessions). Result: **SelfQ fails body parity at 200k — the
smoke-scale body win did not survive scale.**

| metric | heads (seed-mean) | selfq (seed-mean) | selfq/heads |
|--------|-------------------|-------------------|-------------|
| body_ce | 0.0428 | 0.0855 | **2.00** |
| body_brier | 0.0109 | 0.0283 | **2.59** |
| nmse_k1 (descriptive) | 32.1 | 133.5 | 4.16 |
| nmse_k10 (descriptive) | 1.84 | 7.29 | 3.95 |

Reading (`parity.md` has per-seed raw):

- At 200k the separate heads keep improving (heads seed-1 reaches body CE
  6e-4 / Brier 1e-6 — near-perfect) while SelfQ plateaus ~2x worse on both
  body metrics and ~4x worse on forecaster NMSE. The shared-base
  interference that the smoke-scale branch/optimizer fix mitigated does
  not vanish at scale — it compounds: with 40x more updates, the two tasks'
  competition for the shared base costs more, not less.
- The forecaster rows are labeled descriptive by the script (a smoke-scale
  assumption baked into its gating); at 200k the forecaster IS past viable
  scale (docs/acceptance/stage-A/scale_curves_fullscale), so the ~4x gap is
  in fact meaningful evidence against SelfQ, strengthening the verdict.

**Verdict: the unified-self-model hypothesis is UNSUPPORTED at scale.**
`ledger.impl: heads` (the default) stands; SelfQ remains in the tree as the
losing arm of a completed comparison. This is the parity gate doing its
job — an honest negative for the stage-E hypothesis, recorded with the same
prominence as a win. Registry updated (`lab/assumptions.yaml:selfq-unified`
-> unsupported, citing this file).

Possible follow-ups (Phase-3 recommendation list, NOT implemented): fully
separate towers behind one query API (unification of interface without
unification of representation); per-task LoRA-style adapters on a frozen
shared base.
