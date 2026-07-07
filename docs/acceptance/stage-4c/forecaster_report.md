# Forecaster report

- config: `configs/smoke.yaml`, seed 0, 50048 env steps
- 20000 (h, plan, realized-future) tuples; plan usage: forage_nearest=5326, explore_high_epistemic=650, rest=1892, goto_shelter=12132

| horizon | MSE (forecaster) | MSE (identity baseline) | NMSE |
|---------|------------------|-------------------------|------|
| 1 | 0.000513 | 0.000061 | 8.3656 |
| 10 | 0.000962 | 0.001238 | 0.7768 |

NMSE < 1.0 means the forecaster beats the identity predictor.

**PASS: NMSE=0.7768 < 1.0 at k=10 (beats identity)**