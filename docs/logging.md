# Log schema

Two sinks per run, both under `runs/<timestamp>/`:

1. **JSONL event log** — one record per tick, written by
   `training.loggers.JsonlRunLogger`.
2. **TensorBoard scalars** — written by `training.loggers.TBLogger`
   (scalars only for now).

## JSONL event log

Files rotate every 100,000 ticks (`rotate_every`), named
`events-<chunk:09d>.jsonl` with `chunk = tick // rotate_every`.
One JSON object per line:

| field     | type          | description                                              |
|-----------|---------------|----------------------------------------------------------|
| `tick`    | int           | world tick after the step (first record has tick 1)      |
| `pos`     | [int, int]    | agent position [x, y] after the step                     |
| `action`  | int           | action id 0..8 (see action table in world/engine.py)     |
| `success` | bool          | realized success flag from the engine                    |
| `intero`  | [float x 6]   | [energy, fatigue, memory_pressure, sin(tod), cos(tod), 1]|
| `reward`  | float         | per-tick reward (0.0 until drives are implemented)       |
| `events`  | list, optional| ground-truth world events this tick (omitted when empty) |

### Event records (`events[]`)

Every entry has at least `tick`, `type`, `cause`. `cause` is the ground-truth
attribution label (`"self"` or `"world"`); it exists for evaluation only and
must never enter observations or losses.

| type                     | cause | extra fields                              |
|--------------------------|-------|-------------------------------------------|
| `agent_moved`            | self/world | `pos`, `agent`, `dpos`               |
| `food_consumed`          | self/world | `pos`, `agent` (null if world)       |
| `food_regrown`           | world | `pos`                                      |
| `food_relocated`         | world | `pos` (destination), `src`, `had_food`     |
| `seasonal_shift`         | world | `patches_moved`                            |
| `mark_placed`            | self  | `pos`, `agent`                             |
| `mark_erased`            | self  | `pos`, `agent`                             |
| `capability_shift_start` | world | `action`, `fail_prob`, `energy_mult`       |
| `capability_shift_end`   | world | `action`                                   |
| `exogenous_reset`        | world | —                                          |

## TensorBoard scalars

Tag convention: `<group>/<name>`, e.g. `train/loss`, `world/ticks_per_sec`.
Scalars only; histograms/images may be added later behind the same wrapper.
