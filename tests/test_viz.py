"""Viz tests: plots render on synthetic data; the viewer replays a real
(tiny) run headlessly and records a playable mp4; the dashboard's data
loaders work on a fixture run dir. All pygame use runs under
SDL_VIDEODRIVER=dummy."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"

from training.ppo import PPOTrainer  # noqa: E402
from viz import plots  # noqa: E402
from viz.viewer import LiveSource, Renderer, ReplaySource, record_mp4  # noqa: E402
from world.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (tiny) run dir: config.json + JSONL events + viz_state.pkl."""
    run_dir = tmp_path_factory.mktemp("run")
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["seed"] = 11
    cfg["device"] = "cpu"
    cfg["agent"] = {"core": "rssm", "hidden_size": 16, "gru_layers": 1,
                    "encoder_channels": [4, 8]}
    cfg["rssm"].update(deter=16, stoch=4, embed=24, ensemble_k=3)
    cfg["memory"]["enabled"] = True
    cfg["ppo"].update(
        rollout_steps=16, seq_len=8, num_envs=2, episode_length=24,
        minibatch_transitions=16, epochs=1, total_steps=10**9, anneal_lr=False,
    )
    cfg["run"]["assert_improvement"] = False
    cfg["run"]["viz_dump_every"] = 32
    trainer = PPOTrainer(cfg, run_dir=run_dir)
    trainer.train(max_updates=3)  # 96 ticks: crosses an episode boundary at 24
    if trainer.jsonl is not None:
        trainer.jsonl.close()
    return run_dir


# -------------------------------------------------------------------- plots


def test_plot_functions_render_to_tmp(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    outs = [
        plots.reward_curve(rng.normal(size=100).cumsum(), tmp_path / "reward.png"),
        plots.metric_around_event(rng.normal(size=200), 100, tmp_path / "event.png",
                                  window=50, metric_name="body_nll",
                                  event_name="capability shift"),
        plots.calibration_diagram([0.1, 0.5, 0.9], [0.15, 0.4, 0.85],
                                  tmp_path / "calib.png",
                                  bin_counts=[10, 30, 20], ece=0.07),
        plots.nmse_bars_per_horizon(
            [1, 10, 100],
            {1: [1.2, 1.1], 10: [0.8, 0.75], 100: [0.6, 0.9]},
            tmp_path / "nmse.png",
        ),
        plots.ablation_boxplots(
            {"ours": rng.normal(1, 0.2, 8), "control": rng.normal(1.4, 0.2, 8)},
            tmp_path / "ablation.png", metric_name="stale-trip rate",
        ),
        plots.divergence_trace(
            {"mirror": rng.random(120) * 5, "ablation": rng.random(120) * 5},
            tmp_path / "div.png", spike_level=3.0,
        ),
    ]
    for out in outs:
        assert out.exists() and out.stat().st_size > 1000, out


def test_nmse_plot_always_has_identity_baseline(tmp_path: Path) -> None:
    """The identity-baseline line is mandatory on forecasting plots — assert
    the axes actually contain a horizontal line at y=1.0."""
    import matplotlib.pyplot as plt

    calls: list[float] = []
    orig = plt.Axes.axhline

    def spy(self, y=0, *a, **k):  # noqa: ANN001
        calls.append(y)
        return orig(self, y, *a, **k)

    plt.Axes.axhline = spy  # type: ignore[method-assign]
    try:
        plots.nmse_bars_per_horizon([1, 10], {1: [1.0], 10: [0.9]}, tmp_path / "n.png")
    finally:
        plt.Axes.axhline = orig  # type: ignore[method-assign]
    assert 1.0 in calls, "identity baseline (y=1) missing from NMSE plot"


# ------------------------------------------------------------------- viewer


def test_run_dir_artifacts_written(fixture_run: Path) -> None:
    assert (fixture_run / "config.json").exists()
    assert list(fixture_run.glob("events-*.jsonl")), "no JSONL written by trainer"
    assert (fixture_run / "viz_state.pkl").exists(), "no viz state dump written"


def test_replay_source_reconstructs_and_scrubs(fixture_run: Path) -> None:
    src = ReplaySource(fixture_run)
    # JSONL records env 0's stream: 3 rollouts x 16 ticks = 48 records,
    # crossing the episode boundary at tick 24.
    assert len(src) == 48
    assert len(src.episode_starts) >= 2, "expected an episode boundary in the log"

    f0 = src.frame()
    assert f0["world_size"] == json.loads(
        (fixture_run / "config.json").read_text())["world"]["size"]
    assert f0["tick"] == src.records[0]["tick"]
    # Scrub forward across the episode boundary and back.
    src.seek(len(src) - 1)
    f_end = src.frame()
    assert f_end["tick"] == src.records[-1]["tick"]
    src.seek(0)
    f0_again = src.frame()
    assert f0_again["tick"] == f0["tick"]
    assert np.array_equal(f0_again["food"], f0["food"]), "backward seek not reproducible"


def test_renderer_headless_draws_frame(fixture_run: Path) -> None:
    import pygame

    pygame.init()
    src = ReplaySource(fixture_run)
    renderer = Renderer(src.frame()["world_size"])
    for name in ("epistemic", "memory", "divergence"):
        renderer.toggle(name)  # overlays on (n/a in replay must not crash)
    surface = pygame.Surface(renderer.size)
    renderer.render(surface, src.frame())
    arr = pygame.surfarray.array3d(surface)
    assert arr.shape[2] == 3
    assert arr.std() > 10, "frame appears blank"
    pygame.quit()


def test_live_source_reads_state_dump(fixture_run: Path) -> None:
    src = LiveSource(fixture_run)
    state = src.frame()
    assert state is not None
    assert "terrain" in state and state["agent_pos"] is not None
    # Live dumps carry the overlay payloads (memory enabled in the fixture).
    assert state.get("memory") is not None


def test_record_produces_playable_mp4(fixture_run: Path, tmp_path: Path) -> None:
    import imageio.v2 as imageio

    out = record_mp4(ReplaySource(fixture_run), tmp_path / "run.mp4", every=5)
    assert out.exists() and out.stat().st_size > 5000
    reader = imageio.get_reader(out)
    frames = [f for _, f in zip(range(3), reader)]
    reader.close()
    assert len(frames) == 3 and frames[0].ndim == 3, "mp4 not readable"


# ----------------------------------------------------------------- dashboard


def test_dashboard_data_loaders(fixture_run: Path, tmp_path: Path) -> None:
    from viz import dashboard

    runs_root = fixture_run.parent
    runs = dashboard.list_runs(runs_root)
    assert fixture_run in runs

    cfg = dashboard.load_run_config(fixture_run)
    assert cfg["world"]["size"] > 0
    diff = dashboard.config_diff(cfg, dashboard.load_base_config(ROOT / "configs" / "base.yaml"))
    assert any("seed" in row["key"] for row in diff)

    scalars = dashboard.load_tb_scalars(fixture_run)
    assert "reward/rollout" in scalars
    steps, values = scalars["reward/rollout"]
    assert len(steps) == len(values) > 0

    mem = dashboard.load_memory_entries(fixture_run)
    assert mem is not None and {"x", "y", "reliability", "last_verified"} <= set(mem.columns)

    # Experiments loader on a synthetic results dir.
    exp = tmp_path / "results" / "20990101-0000"
    exp.mkdir(parents=True)
    (exp / "summary.csv").write_text(
        "test,metric,ours,ours_ci95,control,control_ci95,delta,n,note\n"
        "demo,acc,0.9,0.02,0.5,0.03,0.4,5,note\n"
    )
    table = dashboard.load_experiment_summaries(tmp_path / "results")
    assert len(table) == 1 and table.iloc[0]["test"] == "demo"
