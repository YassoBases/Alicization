# Full battery summary

- scale: **quick** (see SCALES in full_battery.py), seeds: 1
- negative results are in this table on purpose.
- `stamp` = evidence iff this scale meets the test's MIN_VIABLE_SCALE contract; machinery-only rows validate plumbing and must never be pooled with evidence rows (experiments/metrics.py refuses).

| test | metric | ours | control | delta | n | stamp | note |
|------|--------|------|---------|-------|---|-------|------|
| capability-shift(rssm) | performance_recovery_ratio | 0.3924 +/- 0.32 | 0.3219 +/- 0.32 | +0.07047 | 3 | machinery-only | A=ledger->policy, B=withheld |
| ghost-attribution | accuracy | 0.07682 +/- nan | 0.9204 +/- nan | -0.8436 | 1 | machinery-only | control = always-SELF majority |
| memory-reliability | stale_trip_rate_per_1k | 12.86 +/- nan | 24.01 +/- nan | -11.15 | 1 | evidence | lower is better; control = reliability-blind |
| forecaster-nmse | nmse_k1 | 1780 +/- nan | 1 +/- nan | +1779 | 1 | machinery-only | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k10 | 89.17 +/- nan | 1 +/- nan | +88.17 | 1 | machinery-only | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k100 | 1.848 +/- nan | 1 +/- nan | +0.8483 | 1 | machinery-only | control = identity predictor (NMSE 1.0) |
| kidnapped-agent | relocalization_ticks | 4 +/- nan | 7.75 +/- nan | -3.75 | 1 | machinery-only | lower is better; control = mirror responses off |
| seasonal-shift | adaptation_dip | 0.0225 +/- nan | -0.005391 +/- nan | +0.02789 | 1 | machinery-only | lower dip = better; control = wake-only; BWT n/a (seasons never return), dip-trend is the FWT proxy |
| seasonal-shift | dip_trend_fwt_proxy | -0.2028 +/- nan | -0.2389 +/- nan | +0.03609 | 1 | machinery-only | negative slope = later shifts hurt less |
| sleep-ablation | final_reward | -0.1492 +/- nan | -0.112 +/- nan | -0.03719 | 1 | machinery-only | control = wake-only (no consolidation) |
| reset-anticipation | js_divergence | 0.007503 +/- nan | 0.01346 +/- nan | -0.005952 | 1 | evidence | EXPECTED ~zero vs null; exceeding null q95 = stop-and-investigate |
