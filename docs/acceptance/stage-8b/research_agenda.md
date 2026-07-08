# Research agenda @ tick 6144

The single top-level research artifact: what the agent does not understand, and which experiment reduces that uncertainty most efficiently. Nothing here executes; the human picks.

## 1. [question] is capability (action 3) degraded or mis-modeled? success rate shifted 7.1 sd between recent windows

- experiment: `{"name": "probe_action_batch", "action": 3, "n": 32, "cost": 1.0}`
- score 0.5000 = value 1.000 x tractability 0.500 x novelty 1.000 / cost 1.00

## 2. [question] standing assumption contradicted: success rate of action 2 is stable — what changed?

- experiment: `{"name": "probe_action_batch", "action": 2, "n": 32, "cost": 1.0}`
- score 0.5000 = value 1.000 x tractability 0.500 x novelty 1.000 / cost 1.00
- a result would move: hyp-capability-success-2

## 3. [question] standing assumption contradicted: success rate of action 3 is stable — what changed?

- experiment: `{"name": "probe_action_batch", "action": 3, "n": 32, "cost": 1.0}`
- score 0.5000 = value 1.000 x tractability 0.500 x novelty 1.000 / cost 1.00
- a result would move: hyp-capability-success-3

## 4. [question] what are the dynamics of region (3, 3) (cell 29,28)? ensemble disagreement is at 100% of the map peak

- experiment: `{"name": "directed_visit", "region": [3, 3], "cost": 2.0}`
- score 0.2500 = value 1.000 x tractability 0.500 x novelty 1.000 / cost 2.00

## 5. [question] what are the dynamics of region (0, 2) (cell 21,7)? ensemble disagreement is at 100% of the map peak

- experiment: `{"name": "directed_visit", "region": [0, 2], "cost": 2.0}`
- score 0.2491 = value 0.996 x tractability 0.500 x novelty 1.000 / cost 2.00

## 6. [question] what are the dynamics of region (2, 1) (cell 9,16)? ensemble disagreement is at 100% of the map peak

- experiment: `{"name": "directed_visit", "region": [2, 1], "cost": 2.0}`
- score 0.2489 = value 0.996 x tractability 0.500 x novelty 1.000 / cost 2.00

## 7. [question] is capability (action 2) degraded or mis-modeled? success rate shifted 1.3 sd between recent windows

- experiment: `{"name": "probe_action_batch", "action": 2, "n": 32, "cost": 1.0}`
- score 0.1290 = value 0.258 x tractability 0.500 x novelty 1.000 / cost 1.00
