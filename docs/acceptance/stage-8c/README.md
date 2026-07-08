# Stage-8c acceptance: EIG ranker v2

**Criteria:** v2 differs plausibly from v1; the noisy-TV region is never
promoted; the world model is injected (researcher/ stays torch-free).

- `verify_eig_run.txt`: on a live 12,288-step smoke run, v2 rescored all
  3 world_uncertainty items via the RSSM adapter and REORDERED the
  agenda (v1's third item promoted to first — imagined-rollout
  learnability differs from raw disagreement). Predicted gains finite
  and non-negative; visit reduction 0.027 in [0, 1] (aleatoric variance
  dominates at this scale — reported, not inflated).
- Noisy-TV: a short organic run has no guaranteed irreducibly-random
  region, so the guard is proven with a synthetic adapter in
  tests/test_eig.py — region (0,0) carries the LARGEST predicted gain
  (0.9) with zero learning progress and never takes the top slot.
- Approximations documented in researcher/eig.py and
  experiments/model_adapter.py docstrings (reducible-fraction visit
  estimate; Beta posterior-variance probe estimate).
- v1 kept as config-selectable ablation: `researcher.ranker: v1|v2`.

Reproduce: `python scripts/verify_eig.py`.
