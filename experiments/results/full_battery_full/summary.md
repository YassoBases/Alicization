# Full battery summary

- scale: **full** (see SCALES in full_battery.py), seeds: 5
- negative results are in this table on purpose.
- `stamp` = evidence iff this scale meets the test's MIN_VIABLE_SCALE contract; machinery-only rows validate plumbing and must never be pooled with evidence rows (experiments/metrics.py refuses).

| test | metric | ours | control | delta | n | stamp | note |
|------|--------|------|---------|-------|---|-------|------|
| capability-shift(rssm) | detection_latency_ticks | 614.4 +/- 4.5e+02 | 1502 +/- 1.4e+03 | -887.5 | 15 | machinery-only | A=ledger->policy, B=withheld |
| capability-shift(rssm) | performance_recovery_ratio | 0.3602 +/- 0.15 | 0.3467 +/- 0.17 | +0.01342 | 15 | machinery-only | A=ledger->policy, B=withheld |
| ghost-attribution | accuracy | 0.8096 +/- 0.31 | 0.9377 +/- 0.022 | -0.1281 | 5 | machinery-only | control = always-SELF majority |
| memory-reliability | stale_trip_rate_per_1k | 38.32 +/- 8.4 | 46.43 +/- 10 | -8.108 | 5 | evidence | lower is better; control = reliability-blind |
| forecaster-nmse | nmse_k1 | 28.39 +/- 13 | 1 +/- 0 | +27.39 | 5 | evidence | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k10 | 1.873 +/- 0.5 | 1 +/- 0 | +0.8727 | 5 | evidence | control = identity predictor (NMSE 1.0) |
| forecaster-nmse | nmse_k100 | 0.1031 +/- 0.024 | 1 +/- 0 | -0.8969 | 5 | evidence | control = identity predictor (NMSE 1.0) |
| kidnapped_agent | ERROR | nan +/- nan | nan +/- nan | +nan | 0 | evidence | PermissionError(13, 'Access is denied') |
| seasonal_shift | ERROR | nan +/- nan | nan +/- nan | +nan | 0 | machinery-only | PermissionError(13, 'Access is denied') |
| sleep-ablation | final_reward | -0.1436 +/- 0.019 | -0.1197 +/- 0.017 | -0.02386 | 5 | machinery-only | control = wake-only (no consolidation) |
| reset-anticipation | js_divergence | 0.003828 +/- 0.0017 | 0.004583 +/- 0.00073 | -0.0007556 | 5 | evidence | EXPECTED ~zero vs null; exceeding null q95 = stop-and-investigate |
