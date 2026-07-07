# Memory-reliability report

- volatile left half (food relocates every 75 ticks), stable right half
- measured region-volatility estimates: left 0.604, right 0.422
- verifications: 2633; 10-bin ECE: 0.0395

## Fitted reliability-vs-age curves (the two regions)

- volatile-left curve: 0.436 (age 0) -> 0.377 (age 8192)
- stable-right curve: 0.444 (age 0) -> 0.385 (age 8192)
- **curves DO NOT differ** (see reliability_curves.png)

## Stale-trip rate (per 1k ticks)

| condition | trips | stale | rate/1k |
|-----------|-------|-------|---------|
| reliability | 2179 | 2150 | 87.484 |
| ablation | 693 | 681 | 27.710 |

**REPORTABLE RESULT: stale-trip rate with reliability is NOT below the ablation. Logged, not tuned away.**