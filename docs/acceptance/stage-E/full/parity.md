# Stage-E parity gate: SelfQ vs the heads it replaces

- scale: **200000 steps**, seeds: 3 (lower is better on every metric)
- **GATED** on the body metrics (they feed the policy + mirror and are functional at smoke scale): SelfQ must be within 15% of heads.
- **DESCRIPTIVE** on forecaster NMSE: below its minimum viable scale (both impls NMSE >> identity; stage-A needs ~50k/100 grad-steps), so machinery-only, NOT gated here — forecaster parity is the full-scale stage-A follow-up. k=1 is additionally denominator-degenerate.
- mirror + attribution are NOT replaced this stage.

| metric | kind | heads | selfq | selfq/heads | result |
|--------|------|-------|-------|-------------|--------|
| body_ce | gate | 0.04278 | 0.08553 | 2.000 | FAIL |
| body_brier | gate | 0.01092 | 0.02833 | 2.593 | FAIL |
| nmse_k1 | descriptive | 32.11 | 133.5 | 4.156 | n/a (below MVS) |
| nmse_k10 | descriptive | 1.843 | 7.29 | 3.954 | n/a (below MVS) |

**Overall (body parity): FAIL**

Per-seed raw metrics:
```json
{
  "heads": [
    {
      "body_ce": 0.040613842220045625,
      "body_brier": 0.009684801126604725,
      "nmse_k1": 7.922561550140381,
      "nmse_k10": 0.8778468251228333
    },
    {
      "body_ce": 0.000588229761342518,
      "body_brier": 1.3024714411891125e-06,
      "nmse_k1": 79.88314056396484,
      "nmse_k10": 3.6497641086578367
    },
    {
      "body_ce": 0.08712576869875192,
      "body_brier": 0.023086549062281848,
      "nmse_k1": 8.538524055480957,
      "nmse_k10": 1.0025997400283813
    }
  ],
  "selfq": [
    {
      "body_ce": 0.07517835572361946,
      "body_brier": 0.02291367817670107,
      "nmse_k1": 194.646826171875,
      "nmse_k10": 10.871629333496093
    },
    {
      "body_ce": 0.13432038202881813,
      "body_brier": 0.04630646239966154,
      "nmse_k1": 155.34881896972655,
      "nmse_k10": 8.7892822265625
    },
    {
      "body_ce": 0.04710163150448352,
      "body_brier": 0.015764958265310724,
      "nmse_k1": 50.403649139404294,
      "nmse_k10": 2.2078755378723143
    }
  ]
}
```