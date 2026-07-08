# Kidnapped-agent report

- teleported >= half the map during sleep; measured 24576-step-trained agent
- spike criterion: divergence > 4.95 (max of threshold 3.0 and baseline q99)

| condition | baseline div | spike ticks | relocalization (mean) | mirror triggers |
|-----------|--------------|-------------|------------------------|-----------------|
| mirror | 1.40 | [1, 1, 1, 1] | 18.3 | 117 |
| ablation | 1.26 | [0, 1, 1, 1] | 34.3 | 0 |

**PASS: divergence spiked within 20 ticks of waking in every env**