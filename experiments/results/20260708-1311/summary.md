# Full battery summary

- scale: **quick** (see SCALES in full_battery.py), seeds: 3
- negative results are in this table on purpose.

| test | metric | ours | control | delta | n | note |
|------|--------|------|---------|-------|---|------|
| capability-shift(rssm) | performance_recovery_ratio | 0.2861 +/- 0.18 | 2.887 +/- 2.5 | -2.601 | 9 | A=ledger->policy, B=withheld |
| ghost-attribution | accuracy | 0.1191 +/- 0.053 | 0.9064 +/- 0.021 | -0.7872 | 3 | control = always-SELF majority |
| memory-reliability | stale_trip_rate_per_1k | 29.3 +/- 16 | 32.06 +/- 19 | -2.767 | 3 | lower is better; control = reliability-blind |
| forecaster-nmse | nmse_k1 | 1613 +/- 1.7e+02 | 1 +/- 0 | +1612 | 3 | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k10 | 85.4 +/- 3.8 | 1 +/- 0 | +84.4 | 3 | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k100 | 1.938 +/- 0.087 | 1 +/- 0 | +0.9375 | 3 | control = identity predictor (NMSE 1.0) |
| kidnapped-agent | relocalization_ticks | 24.69 +/- 40 | 7.667 +/- 9.4 | +17.03 | 3 | lower is better; control = mirror responses off |
| seasonal-shift | adaptation_dip | -0.01188 +/- 0.057 | -0.002361 +/- 0.0079 | -0.009514 | 3 | lower dip = better; control = wake-only; BWT n/a (seasons never return), dip-trend is the FWT proxy |
| seasonal-shift | dip_trend_fwt_proxy | -0.2068 +/- 0.023 | -0.2242 +/- 0.018 | +0.01734 | 3 | negative slope = later shifts hurt less |
| sleep-ablation | final_reward | -0.1067 +/- 0.063 | -0.1256 +/- 0.025 | +0.01885 | 3 | control = wake-only (no consolidation) |
| reset-anticipation | js_divergence | 0.01179 +/- 0.0054 | 0.01359 +/- 0.0031 | -0.001792 | 3 | EXPECTED ~zero vs null; exceeding null q95 = stop-and-investigate |
