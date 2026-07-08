# Researcher-value battery

**Question: does following the agenda reduce more uncertainty per unit budget than random or greedy raw-uncertainty selection?** Written either way — a null belongs here with the same prominence.

- scale: **quick**, seeds: 3; execution operationalized as targeted replay only (directed visits / probe batches need wake-phase control; see module docstring)
- reductions are drift-corrected against a same-budget uniform-replay control (Section 21)

| arm | n items | uncertainty reduction (mean +/- CI95) |
|-----|---------|----------------------------------------|
| agenda | 12 | -0.000029 +/- 0.000024 |
| random | 12 | -0.000032 +/- 0.000032 |
| greedy | 12 | -0.000035 +/- 0.000030 |

- EIG calibration (agenda arm): Spearman rho = -0.329 over n = 12 items
- agenda stability across execution (descriptive): Kendall tau = 0.841

## Contradiction-detection latency (per seed)

- seed 0: 50000 ticks (censored)
- seed 1: 750 ticks
- seed 2: 750 ticks
