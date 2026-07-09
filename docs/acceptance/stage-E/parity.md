# Stage-E parity gate: SelfQ vs the heads it replaces

- scale: **24576 steps**, seeds: 3 (lower is better on every metric)
- **GATED** on the body metrics (they feed the policy + mirror and are functional at smoke scale): SelfQ must be within 15% of heads.
- **DESCRIPTIVE** on forecaster NMSE: below its minimum viable scale (both impls NMSE >> identity; stage-A needs ~50k/100 grad-steps), so machinery-only, NOT gated here — forecaster parity is the full-scale stage-A follow-up. k=1 is additionally denominator-degenerate.
- mirror + attribution are NOT replaced this stage.

| metric | kind | heads | selfq | selfq/heads | result |
|--------|------|-------|-------|-------------|--------|
| body_ce | gate | 0.2687 | 0.2881 | 1.072 | PASS |
| body_brier | gate | 0.01978 | 0.008632 | 0.436 | PASS |
| nmse_k1 | descriptive | 1443 | 3046 | 2.110 | n/a (below MVS) |
| nmse_k10 | descriptive | 82.34 | 131.3 | 1.594 | n/a (below MVS) |

**Overall (body parity): PASS**

Per-seed raw metrics:
```json
{
  "heads": [
    {
      "body_ce": 0.3967096358537674,
      "body_brier": 0.02462807334959507,
      "nmse_k1": 1481.66904296875,
      "nmse_k10": 83.24723510742187
    },
    {
      "body_ce": 0.18125061243772506,
      "body_brier": 0.012075916272442555,
      "nmse_k1": 1427.6310791015626,
      "nmse_k10": 81.71634368896484
    },
    {
      "body_ce": 0.22826811373233796,
      "body_brier": 0.02265068721026182,
      "nmse_k1": 1421.00498046875,
      "nmse_k10": 82.06182098388672
    }
  ],
  "selfq": [
    {
      "body_ce": 0.36054699793457984,
      "body_brier": 0.017992213244178858,
      "nmse_k1": 1758.9158447265625,
      "nmse_k10": 95.25342712402343
    },
    {
      "body_ce": 0.40349496267735957,
      "body_brier": 0.007649065366604191,
      "nmse_k1": 3067.5376953125,
      "nmse_k10": 135.17470703125
    },
    {
      "body_ce": 0.10033897099783644,
      "body_brier": 0.00025503500069135043,
      "nmse_k1": 4310.59208984375,
      "nmse_k10": 163.3484649658203
    }
  ]
}
```