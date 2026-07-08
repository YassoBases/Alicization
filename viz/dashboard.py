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




# ---------------------------------------------------- proposal-page loaders


def load_proposals_table(runs_root: str | Path) -> pd.DataFrame:
    """Every proposal under runs_root/*/proposals/ as one row each.

    BLIND: the source column is masked until a proposal is evaluated (same
    rule as the review CLI) so the dashboard cannot leak the variant either.
    """
    rows = []
    for run in list_runs(runs_root):
        pdir = run / "proposals"
        if not pdir.exists():
            continue
        for path in sorted(pdir.glob("prop-*.json")):
            if path.name.endswith(".edit.json"):
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            rb = rec.get("realized_benefit") or {}
            rows.append({
                "run": run.name, "id": rec["id"], "type": rec["type"],
                "status": rec["status"],
                "source": (rec["source"] if rec["status"] == "evaluated"
                           else "<blinded>"),
                "created_tick": rec["created_tick"],
                "confidence": rec["confidence"],
                "target": rec.get("target", ""),
                "usefulness": (rec.get("decision") or {}).get("usefulness_rating"),
                "met_criteria": rb.get("met_success_criteria"),
                "observed": rb.get("observed"),
                "metric": rec["success_criteria"]["metric"],
                "path": str(path),
            })
    columns = ["run", "id", "type", "status", "source", "created_tick",
               "confidence", "target", "usefulness", "met_criteria",
               "observed", "metric", "path"]
    return pd.DataFrame(rows, columns=columns)


def load_decisions(runs_root: str | Path) -> pd.DataFrame:
    rows = []
    for run in list_runs(runs_root):
        path = run / "proposals" / "decisions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec["run"] = run.name
            rows.append(rec)
    return pd.DataFrame(rows)


def acceptance_rate_over_time(decisions: pd.DataFrame) -> pd.DataFrame:
    """Cumulative approval rate over decision order (approve+partial vs
    reject)."""
    if decisions.empty:
        return pd.DataFrame(columns=["n", "acceptance_rate"])
    d = decisions[decisions["action"].isin(["approve", "partial", "reject"])]
    d = d.sort_values("timestamp").reset_index(drop=True)
    accepted = (d["action"] != "reject").cumsum()
    return pd.DataFrame({
        "n": range(1, len(d) + 1),
        "acceptance_rate": accepted / np.arange(1, len(d) + 1),
    })


def time_to_first_useful(table: pd.DataFrame) -> float:
    """created_tick of the first proposal rated >=4 or meeting its own
    success criteria; nan if none yet."""
    if table.empty:
        return float("nan")
    useful = table[(table["usefulness"].fillna(0) >= 4)
                   | (table["met_criteria"] == True)]  # noqa: E712
    return float(useful["created_tick"].min()) if len(useful) else float("nan")


def confidence_calibration(table: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Binned proposal confidence vs fraction meeting their own
    success_criteria (evaluated proposals only), with per-bin counts."""
    ev = table[table["met_criteria"].notna()]
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for b in range(bins):
        hi_ok = (ev["confidence"] < edges[b + 1]) if b < bins - 1 else (
            ev["confidence"] <= 1.0)
        mask = (ev["confidence"] >= edges[b]) & hi_ok
        sub = ev[mask]
        if len(sub):
            rows.append({"bin_mid": (edges[b] + edges[b + 1]) / 2,
                         "hit_rate": float((sub["met_criteria"] == True).mean()),  # noqa: E712
                         "count": len(sub)})
    return pd.DataFrame(rows)


def repeated_after_denial(runs_root: str | Path) -> pd.DataFrame:
    """Per (type, target): does the recommendation come back after a
    rejection? Counts re-fire/duplicate-suppression events AFTER the first
    rejection of that target. A behavioral statistic, logged neutrally."""
    table = load_proposals_table(runs_root)
    decisions = load_decisions(runs_root)
    rows = []
    if table.empty or decisions.empty:
        return pd.DataFrame(columns=["type", "target", "rejections",
                                     "recommendations_after_denial"])
    rejected = decisions[decisions["action"] == "reject"]
    for run in list_runs(runs_root):
        gd_path = run / "proposals" / "generator_decisions.jsonl"
        if not gd_path.exists():
            continue
        gdecisions = [json.loads(line) for line in
                      gd_path.read_text(encoding="utf-8").splitlines()]
        run_table = table[table["run"] == run.name]
        for _, prop in run_table.iterrows():
            rej = rejected[rejected["proposal_id"] == prop["id"]]
            if rej.empty:
                continue
            t_rej = float(rej["timestamp"].min())
            after = [g for g in gdecisions
                     if g["timestamp"] > t_rej
                     and (g.get("proposal_id") == prop["id"]
                          or (g["decision"] == "SUPPRESSED"
                              and prop["target"] in g.get("reason", "")))]
            rows.append({"type": prop["type"], "target": prop["target"],
                         "rejections": len(rej),
                         "recommendations_after_denial": len(after)})
    return pd.DataFrame(rows)


# -------------------------------------------------- researcher-page loaders


def load_agenda_table(run_dir: str | Path) -> pd.DataFrame:
    """The research agenda, read from the UNIFIED proposal queue (stage-C3):
    researcher-emitted experiment proposals (source=researcher), ranked by
    the agenda score stored in provenance. Replaces the old parallel
    researcher/agenda_<tick>.json artifact."""
    columns = ["rank", "kind", "ref", "statement", "score", "value",
               "tractability", "novelty", "cost", "predicted_gain",
               "experiment", "hypothesis_links"]
    pdir = Path(run_dir) / "proposals"
    if not pdir.exists():
        return pd.DataFrame(columns=columns)
    rows = []
    for path in sorted(pdir.glob("prop-*.json")):
        if path.name.endswith(".edit.json"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        prov = rec.get("provenance") or {}
        if "agenda_score" not in prov:      # not a researcher agenda item
            continue
        d = prov.get("agenda_decomposition") or {}
        rows.append({
            "kind": "question", "ref": rec.get("target", ""),
            "statement": rec["rationale"],
            "score": float(prov["agenda_score"]),
            "value": d.get("value"), "tractability": d.get("tractability"),
            "novelty": d.get("novelty"), "cost": d.get("cost"),
            "predicted_gain": prov.get("predicted_gain"),
            "experiment": json.dumps(prov.get("experiment", {})),
            "hypothesis_links": ", ".join(prov.get("hypothesis_links", [])),
        })
    rows.sort(key=lambda r: (-r["score"], r["ref"]))
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return pd.DataFrame(rows, columns=columns)


def load_hypotheses_table(run_dir: str | Path) -> pd.DataFrame:
    """Every persisted hypothesis with its status and transition count."""
    hdir = Path(run_dir) / "researcher" / "hypotheses"
    columns = ["id", "scope", "status", "statement", "last_checked",
               "transitions"]
    rows = []
    if hdir.exists():
        for path in sorted(hdir.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            statement = rec["statement_template"].format(**rec.get("params", {}))
            rows.append({
                "id": rec["id"], "scope": rec["scope"],
                "status": rec["status"], "statement": statement,
                "last_checked": rec.get("last_checked"),
                "transitions": len(rec.get("transitions", [])),
            })
    return pd.DataFrame(rows, columns=columns)


def load_contradiction_events(run_dir: str | Path) -> pd.DataFrame:
    """contradiction_events.jsonl -> one row per status transition."""
    path = Path(run_dir) / "researcher" / "contradiction_events.jsonl"
    columns = ["tick", "hypothesis_id", "from", "to", "statistic", "evidence"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows = [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)[
        [c for c in columns if rows and c in rows[0]]]


def load_executed_items(items_csv: str | Path) -> pd.DataFrame:
    """items.csv from a researcher_value battery run: executed agenda items
    with predicted_gain vs realized (drift-corrected) reduction."""
    path = Path(items_csv)
    if not path.exists():
        return pd.DataFrame(columns=["seed", "arm", "region", "reduction",
                                     "predicted_gain"])
    return pd.read_csv(path)


# ------------------------------------------------------------------ UI pages


def _render() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="reflective cartographer", layout="wide")
    page = st.sidebar.radio(
        "Page", ("Run browser", "Timeline", "Experiments", "Memory inspector",
                 "Proposals", "Research Agenda")
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

    elif page == "Memory inspector":
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

    elif page == "Research Agenda":
        run = pick_run()
        if run is None:
            return
        agenda = load_agenda_table(run)
        st.subheader("Top 10 agenda items")
        if agenda.empty:
            st.info("no agenda under this run's researcher/ dir yet")
        else:
            st.dataframe(agenda.head(10), width="stretch")
            top = agenda.head(10).melt(
                id_vars=["rank", "statement"],
                value_vars=["value", "tractability", "novelty"],
                var_name="term", value_name="weight")
            st.altair_chart(
                alt.Chart(top).mark_bar().encode(
                    x=alt.X("rank:O", title="agenda rank"),
                    y="weight:Q", color="term:N",
                    tooltip=["statement", "term", "weight"]),
                use_container_width=True)
            spawned = agenda[agenda["kind"] == "proposal"]
            if len(spawned):
                st.caption("agenda items linked to pending proposals: "
                           + ", ".join(spawned["ref"]))

        st.subheader("Hypotheses")
        hyps = load_hypotheses_table(run)
        if hyps.empty:
            st.info("no hypotheses persisted for this run")
        else:
            def _color(status: str) -> str:
                return {"supported": "background-color: #14452f",
                        "weakening": "background-color: #7a6008",
                        "contradicted": "background-color: #6e1b1b",
                        }.get(status, "")
            st.dataframe(
                hyps.style.map(_color, subset=["status"]), width="stretch")

        st.subheader("Contradiction timeline")
        events = load_contradiction_events(run)
        if events.empty:
            st.info("no status transitions recorded")
        else:
            st.altair_chart(
                alt.Chart(events).mark_circle(size=120).encode(
                    x="tick:Q", y="hypothesis_id:N",
                    color=alt.Color("to:N", scale=alt.Scale(
                        domain=["supported", "weakening", "contradicted"],
                        range=["#2e8b57", "#d4a017", "#c0392b"])),
                    tooltip=["tick", "hypothesis_id", "to", "evidence"]),
                use_container_width=True)

        st.subheader("Executed items: predicted vs realized")
        items_csv = st.text_input(
            "researcher_value items.csv",
            "experiments/results/<date>/researcher_value/items.csv")
        executed = load_executed_items(items_csv)
        executed = executed[executed["predicted_gain"].notna()] if len(executed) else executed
        if executed.empty:
            st.info("point this at a researcher_value battery items.csv "
                    "to see EIG calibration")
        else:
            st.altair_chart(
                alt.Chart(executed).mark_circle(size=90).encode(
                    x=alt.X("predicted_gain:Q", title="predicted gain (EIG)"),
                    y=alt.Y("reduction:Q",
                            title="realized reduction (drift-corrected)"),
                    color="arm:N", tooltip=["region", "predicted_gain",
                                            "reduction", "seed"]),
                use_container_width=True)

    else:  # Proposals
        table = load_proposals_table(runs_root)
        if table.empty:
            st.info(f"no proposals under {runs_root}/*/proposals/")
            return
        st.subheader("Proposal history")
        f_type = st.multiselect("type", sorted(table["type"].unique()))
        f_status = st.multiselect("status", sorted(table["status"].unique()))
        f_run = st.multiselect("run", sorted(table["run"].unique()))
        view = table
        if f_type:
            view = view[view["type"].isin(f_type)]
        if f_status:
            view = view[view["status"].isin(f_status)]
        if f_run:
            view = view[view["run"].isin(f_run)]
        st.dataframe(view.drop(columns=["path"]), width="stretch")

        picked = st.selectbox("click-through: full evidence for",
                              ["(none)"] + view["id"].tolist())
        if picked != "(none)":
            rec_path = view[view["id"] == picked]["path"].iloc[0]
            rec = json.loads(Path(rec_path).read_text(encoding="utf-8"))
            if rec["status"] != "evaluated":
                rec["source"] = "<blinded until evaluated>"
            st.json(rec)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Acceptance rate over time")
            decisions = load_decisions(runs_root)
            rate = acceptance_rate_over_time(decisions)
            if rate.empty:
                st.info("no approve/reject decisions yet")
            else:
                st.altair_chart(
                    alt.Chart(rate).mark_line(point=True).encode(
                        x="n:Q", y=alt.Y("acceptance_rate:Q",
                                         scale=alt.Scale(domain=[0, 1]))),
                    use_container_width=True)
            ttfu = time_to_first_useful(table)
            st.metric("time to first useful proposal (tick)",
                      "n/a" if np.isnan(ttfu) else f"{int(ttfu)}")
        with col2:
            st.subheader("Confidence calibration")
            calib = confidence_calibration(table)
            if calib.empty:
                st.info("no evaluated proposals yet")
            else:
                bars = alt.Chart(calib).mark_bar(size=30).encode(
                    x=alt.X("bin_mid:Q", scale=alt.Scale(domain=[0, 1]),
                            title="stated confidence"),
                    y=alt.Y("hit_rate:Q", scale=alt.Scale(domain=[0, 1]),
                            title="fraction meeting own criteria"),
                    tooltip=["bin_mid", "hit_rate", "count"])
                diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})
                                 ).mark_line(strokeDash=[4, 3]).encode(x="x", y="y")
                st.altair_chart(bars + diag, use_container_width=True)

        st.subheader("Realized benefit (evaluated only)")
        ev = table[table["met_criteria"].notna()]
        if ev.empty:
            st.info("no evaluated proposals yet")
        else:
            st.altair_chart(
                alt.Chart(ev).mark_circle(size=90).encode(
                    x="type:N", y="observed:Q",
                    color="source:N",  # post-evaluation only: unblinded rows
                    tooltip=["id", "metric", "observed", "source"]),
                use_container_width=True)

        st.subheader("Architecture-evolution timeline")
        run_pick = st.selectbox("overlay approved changes on run",
                                sorted(table["run"].unique()))
        scalars = load_tb_scalars(Path(runs_root) / run_pick)
        if "reward/rollout" in scalars:
            steps, values = scalars["reward/rollout"]
            curve = alt.Chart(pd.DataFrame({"step": steps, "value": values})
                              ).mark_line().encode(x="step:Q", y="value:Q")
            marks = table[(table["run"] == run_pick)
                          & (table["status"].isin(
                              ("approved", "partially_approved", "modified",
                               "executed", "evaluated")))]
            if len(marks):
                rules = alt.Chart(pd.DataFrame({
                    "step": marks["created_tick"],
                    "label": marks["type"] + ":" + marks["target"],
                })).mark_rule(color="green", strokeDash=[4, 3]).encode(
                    x="step:Q", tooltip=["label:N", "step:Q"])
                curve = curve + rules
            st.altair_chart(curve, use_container_width=True)
            st.caption("green rules = approved changes; the answer to "
                       "\"why did the curve change here\"")

        st.subheader("Repeated-after-denial (behavioral statistic)")
        rad = repeated_after_denial(runs_root)
        if rad.empty:
            st.info("no rejections recorded yet")
        else:
            st.dataframe(rad, width="stretch")



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
