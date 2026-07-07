"""Streamlit dashboard over runs/ and experiments/results/.

Pages:
  1. Run browser    — pick runs/<id>; config diff vs configs/base.yaml;
                      headline metrics (last value of key scalars).
  2. Timeline       — chosen TB scalars as altair charts with a SHARED x
                      zoom (linked) and lever-event vertical rules.
  3. Experiments    — all experiments/results/*/summary.csv in one sortable
                      table (test, metric, ours, control, delta, CI, n).
  4. Memory inspector — episodic entries (predicted reliability,
                      last-verified tick) as a table + world-map scatter.

Data loaders below are plain functions (no streamlit import) so tests can
exercise them on a fixture run dir; the UI renders only under
``streamlit run viz/dashboard.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import read_viz_state  # noqa: E402

HEADLINE_TAGS = (
    "reward/rollout", "ledger/body_nll", "ledger/attribution_accuracy",
    "ledger/reliability_ece", "rssm/recon", "rssm/participation_ratio",
    "mirror/divergence", "memory/pressure", "sps",
)

LEVER_TEXT_TAGS = ("levers/events/text_summary", "levers/events")


# ------------------------------------------------------------------- loaders


def list_runs(runs_root: str | Path) -> list[Path]:
    """Run dirs under ``runs_root`` (anything holding config.json or tb/)."""
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and ((p / "config.json").exists() or (p / "tb").exists())
    )


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))


def load_base_config(base_path: str | Path) -> dict[str, Any]:
    from world.config import load_config

    return load_config(base_path)


def _flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def config_diff(run_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Leaf-level differences: [{key, run, base}] sorted by key."""
    run_flat, base_flat = _flatten(run_cfg), _flatten(base_cfg)
    rows = []
    for key in sorted(set(run_flat) | set(base_flat)):
        rv, bv = run_flat.get(key, "<absent>"), base_flat.get(key, "<absent>")
        if rv != bv:
            rows.append({"key": key, "run": rv, "base": bv})
    return rows


def load_tb_scalars(run_dir: str | Path) -> dict[str, tuple[list[int], list[float]]]:
    """All scalar series from the run's TensorBoard log: tag -> (steps, values)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = Path(run_dir) / "tb"
    if not tb_dir.exists():
        return {}
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def load_lever_events(run_dir: str | Path) -> list[tuple[int, str]]:
    """(step, text) lever annotations from the TB text log; [] if none."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = Path(run_dir) / "tb"
    if not tb_dir.exists():
        return []
    acc = EventAccumulator(str(tb_dir), size_guidance={"tensors": 0})
    acc.Reload()
    for tag in LEVER_TEXT_TAGS:
        if tag in acc.Tags().get("tensors", []):
            out = []
            for ev in acc.Tensors(tag):
                try:
                    text = ev.tensor_proto.string_val[0].decode("utf-8")
                except (IndexError, AttributeError):
                    text = "lever event"
                out.append((ev.step, text))
            return out
    return []


def load_memory_entries(run_dir: str | Path) -> pd.DataFrame | None:
    """Episodic entries from the run's viz state dump; None if unavailable."""
    state = read_viz_state(Path(run_dir) / "viz_state.pkl")
    if not state or state.get("memory") is None:
        return None
    mem = state["memory"]
    return pd.DataFrame({
        "x": mem["positions"][:, 0],
        "y": mem["positions"][:, 1],
        "reliability": np.asarray(mem["reliability"], dtype=float),
        "last_verified": mem["last_verified"],
    })


def load_experiment_summaries(results_root: str | Path) -> pd.DataFrame:
    """Concatenate experiments/results/*/summary.csv with a run column."""
    frames = []
    root = Path(results_root)
    if root.exists():
        for csv_path in sorted(root.glob("*/summary.csv")):
            df = pd.read_csv(csv_path)
            df.insert(0, "results_dir", csv_path.parent.name)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["results_dir", "test", "metric", "ours",
                                     "control", "delta", "n", "note"])
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ UI pages


def _render() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="reflective cartographer", layout="wide")
    page = st.sidebar.radio(
        "Page", ("Run browser", "Timeline", "Experiments", "Memory inspector")
    )
    runs_root = st.sidebar.text_input("runs root", "runs")
    runs = list_runs(runs_root)

    def pick_run() -> Path | None:
        if not runs:
            st.warning(f"no runs under {runs_root}/")
            return None
        return st.selectbox("run", runs, format_func=lambda p: p.name)

    if page == "Run browser":
        run = pick_run()
        if run is None:
            return
        st.subheader("Headline metrics")
        scalars = load_tb_scalars(run)
        cols = st.columns(4)
        shown = 0
        for tag in HEADLINE_TAGS:
            if tag in scalars:
                _, values = scalars[tag]
                cols[shown % 4].metric(tag, f"{values[-1]:.4g}")
                shown += 1
        if not shown:
            st.info("no TB scalars found")
        st.subheader("Config diff vs configs/base.yaml")
        try:
            diff = config_diff(load_run_config(run), load_base_config("configs/base.yaml"))
            st.dataframe(pd.DataFrame(diff), width="stretch")
        except FileNotFoundError as exc:
            st.warning(str(exc))

    elif page == "Timeline":
        run = pick_run()
        if run is None:
            return
        scalars = load_tb_scalars(run)
        if not scalars:
            st.info("no TB scalars found")
            return
        default = [t for t in ("reward/rollout", "ledger/body_nll") if t in scalars]
        tags = st.multiselect("metrics", sorted(scalars), default=default or None)
        levers = load_lever_events(run)
        zoom = alt.selection_interval(bind="scales", encodings=["x"])
        charts = []
        for tag in tags:
            steps, values = scalars[tag]
            df = pd.DataFrame({"step": steps, "value": values})
            chart = (
                alt.Chart(df, height=180, title=tag)
                .mark_line()
                .encode(x=alt.X("step:Q", title="env step"), y=alt.Y("value:Q", title=None))
                .add_params(zoom)  # same param across charts -> linked x zoom
            )
            if levers:
                rules = alt.Chart(
                    pd.DataFrame({"step": [s for s, _ in levers],
                                  "event": [t for _, t in levers]})
                ).mark_rule(color="red", strokeDash=[4, 3]).encode(
                    x="step:Q", tooltip=["event:N", "step:Q"]
                )
                chart = chart + rules
            charts.append(chart)
        if charts:
            st.altair_chart(alt.vconcat(*charts), use_container_width=True)

    elif page == "Experiments":
        results_root = st.text_input("results root", "experiments/results")
        table = load_experiment_summaries(results_root)
        if table.empty:
            st.info(f"no summary.csv files under {results_root}/*/")
            return
        st.dataframe(table, width="stretch")  # natively sortable
        st.caption("negative results are in this table on purpose.")

    else:  # Memory inspector
        run = pick_run()
        if run is None:
            return
        mem = load_memory_entries(run)
        if mem is None or mem.empty:
            st.info("no episodic-memory state in this run's viz_state.pkl "
                    "(memory.enabled=false, or dump not written yet)")
            return
        try:
            size = load_run_config(run)["world"]["size"]
        except FileNotFoundError:
            size = int(max(mem["x"].max(), mem["y"].max()) + 1)
        st.subheader(f"{len(mem)} episodic entries")
        st.dataframe(mem.sort_values("reliability"), width="stretch")
        scatter = (
            alt.Chart(mem)
            .mark_circle()
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, size])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[size, 0])),  # world y down
                size=alt.Size("reliability:Q", scale=alt.Scale(range=[20, 300])),
                color=alt.Color("reliability:Q", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["x", "y", "reliability", "last_verified"],
            )
            .properties(width=480, height=480, title="entries on the world map")
        )
        st.altair_chart(scatter)


def _streamlit_running() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _streamlit_running():  # pragma: no cover - exercised via `streamlit run`
    _render()
elif __name__ == "__main__":
    print("run me with: streamlit run viz/dashboard.py")
