# Proposal-quality comparative battery

**HYPOTHESIS: ledger-sourced proposals outperform logs-only-sourced proposals on realized benefit and calibration.** Written either way — a null or negative result belongs in this table with the same prominence.

- scale: **quick**, seeds: 3, scripted blind reviewer (approve-all on a fixed schedule; usefulness ratings n/a by construction)

| source | n evaluated | realized benefit (mean +/- CI95) | hit rate | ECE (n bins) | acceptance rate* | time-to-first-useful (ticks) |
|--------|-------------|-----------------------------------|---------|--------------|------------------|------------------------------|
| ledger | 3 | +72.965 +/- 26.586 | 1.00 | 0.450 (1) | 1.00 | 8192 |
| logs_only | 0 | +nan +/- nan | nan | nan (0) | nan | inf |

\* acceptance rate measures reviewer behavior as much as proposal quality — never headline it alone (Section 17). Under this battery's approve-all scripted reviewer it is 1.0 by construction.

## Per-type breakdown

| source | type | n | benefit mean | hit rate |
|--------|------|---|--------------|----------|
| ledger | hyperparameter | 3 | +72.965 | 1.00 |

## Proposal quality vs agent lifetime

- ledger: Pearson r(created_tick, benefit) = +nan over 3 proposals
- logs_only: too few evaluated proposals for a lifetime trend (n=0)
