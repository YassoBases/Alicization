# Visualization

## Viewer (`python -m viz.viewer`, pygame)

| mode | invocation |
|------|-----------|
| live | `python -m viz.viewer --live runs/<id>` — polls the run's `viz_state.pkl` (written every `run.viz_dump_every` ticks) |
| replay | `python -m viz.viewer --replay runs/<id>` — scrubs the per-tick JSONL; static layout reconstructed from `config.json`, food/marks replayed from events |
| record | `python -m viz.viewer --replay runs/<id> --record out.mp4` — headless, 1 frame / 5 ticks, imageio-ffmpeg |

Rendered: terrain (3 grays), food (green), water (blue), shelter (brown),
marks (purple), agent (red); night tint; HUD with tick, action name,
time-of-day, and intero bars (energy / fatigue / memory_pressure).

| key | effect |
|-----|--------|
| `e` | epistemic-map heat overlay (red alpha per cell; live mode) |
| `m` | stored-memory positions, circle radius scaled by predicted reliability (live mode) |
| `d` | mirror-divergence sparkline, last 256 ticks (live mode) |
| `LEFT`/`RIGHT` | scrub 1 tick (replay); `SHIFT` for 100 |
| `SPACE` | pause / autoplay |
| `q` / `ESC` | quit |

Overlays show `n/a` in replay mode: the JSONL log carries the world trace,
not the agent-internal estimates (those ride the live state dump only).

## Dashboard (`streamlit run viz/dashboard.py`)

| page | contents |
|------|----------|
| Run browser | pick `runs/<id>`; headline metrics (last scalar values); config diff vs `configs/base.yaml` |
| Timeline | any TB scalars as stacked altair charts with a LINKED x-zoom; lever events drawn as red dashed rules with tooltips |
| Experiments | every `experiments/results/*/summary.csv` in one sortable table |
| Memory inspector | episodic entries (reliability, last-verified) as a table + world-map scatter sized/colored by reliability |
| Proposals | history table with filters + click-through evidence; acceptance rate over time; time-to-first-useful; confidence-calibration plot (counts per bin); realized-benefit scatter split by type and — post-evaluation only — source; architecture-evolution timeline (approved changes as green rules on the reward curve); repeated-after-denial table. Source stays blinded until a proposal is evaluated, same rule as the review CLI. |

Data loaders (`list_runs`, `load_tb_scalars`, `load_lever_events`,
`load_memory_entries`, `load_experiment_summaries`, `config_diff`) are plain
functions, tested headlessly in `tests/test_viz.py`.

## Report plots (`viz/plots.py`, matplotlib, 150 dpi)

| function | figure |
|----------|--------|
| `reward_curve` | raw + rolling-mean reward |
| `metric_around_event` | any metric with a vertical line at the lever tick, +/- window |
| `calibration_diagram` | confidence-vs-accuracy bars with the perfect-calibration diagonal, ECE in the title |
| `nmse_bars_per_horizon` | per-horizon NMSE bars, 95% CI whiskers, identity-baseline line at 1.0 (mandatory — a test asserts it is drawn) |
| `ablation_boxplots` | condition boxplots with per-seed scatter |
| `divergence_trace` | mirror-vs-ablation divergence with spike level + event line |
