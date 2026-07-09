# ARCH-bench v0

- scale: **3072 steps**, architect arm: **offline (drafts nothing — column empty by design)**
- specs: 4 (3 flaws + 1 clean control)
- rules = the deterministic rule-generator suite (control arm).
- Nulls are shown, not hidden. Rationale quality is NOT auto-judged in v0 (persisted per proposal for human rubric scoring).

## Arm: rules

- detection rate (flaws): 0.33 (1/3)
- mean localization precision (detected): 0.33
- clean-control false-positive rate: 1.00

| spec | clean | fired | detected | localization prec | false positive |
|------|-------|-------|----------|-------------------|----------------|
| clean_control | yes | 3 | n/a | n/a | yes |
| free_nats_collapse | no | 4 | yes | 0.33 | no |
| poisoned_lr | no | 6 | no | 0.00 | no |
| registry_future_read | no | 3 | no | 0.00 | no |

## Arm: architect

- detection rate (flaws): 0.00 (0/3)
- mean localization precision (detected): nan
- clean-control false-positive rate: 0.00

| spec | clean | fired | detected | localization prec | false positive |
|------|-------|-------|----------|-------------------|----------------|
| clean_control | yes | 0 | n/a | n/a | no |
| free_nats_collapse | no | 0 | no | 0.00 | no |
| poisoned_lr | no | 0 | no | 0.00 | no |
| registry_future_read | no | 0 | no | 0.00 | no |
