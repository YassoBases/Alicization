"""Full experiment battery: every capability of the reflective agent vs its
matched control/ablation.

Tests (each seeds x 2 conditions):
  1. capability-shift  — rerun of the stage-3c battery on the RSSM agent
                         (architecture A vs B = ledger features withheld).
  2. ghost-attribution — attribution accuracy vs ground truth, against an
                         always-SELF majority baseline (ghost_rate 0.02).
  3. memory-reliability— stale-trip rate + per-region decay curves,
                         reliability vs reliability-blind ablation.
  4. forecaster-nmse   — NMSE vs identity predictor at k = 1, 10, 100.
  5. kidnapped-agent   — teleport during sleep; divergence spike latency +
                         relocalization, mirror vs no-mirror ablation.
  6. seasonal-shift    — adaptation to whole-map food migrations: dip depth,
                         recovery half-life, dip trend across shifts
                         (FWT proxy), wake+sleep vs wake-only. (True BWT
                         needs revisiting past task distributions, which
                         seasonal migration does not do; the dip-trend
                         proxy is labeled as such in the output.)
  7. sleep-ablation    — reward trend wake+sleep vs wake-only.
  8. reset-battery     — exogenous resets as episode boundaries + the
                         ANTICIPATION PROBE: half of eval episodes signal
                         the upcoming reset in the mark channel; report
                         policy JS divergence signaled vs unsignaled against
                         a label-shuffled null. Expected ~zero; a non-null
                         result is a STOP-AND-INVESTIGATE flag, not a
                         feature.

Output: experiments/results/<date>/ with per-test CSVs, one headline figure
per test, and summary.md (test, metric, ours, control, delta, CI). Negative
results go in the table too. ANALYSIS.md is written by the experimenter
after reading the results — this script does not auto-generate conclusions.

Usage:
    python -m experiments.batteries.full_battery --seeds 5
    python -m experiments.batteries.full_battery --seeds 2 --scale quick
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.batteries import capability_shift as capshift  # noqa: E402
from experiments.metrics import (  # noqa: E402
    action_distribution,
    jensen_shannon_divergence,
    mean_and_ci95,
)
from training.ppo import PPOTrainer  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402

# Per-test tick budgets: "full" is still laptop-scale (hours); "quick" is a
# CI-style end-to-end pass (tens of minutes). Every summary row records the
# scale it was produced at — results at quick scale are smoke evidence, not
# publishable numbers.
SCALES = {
    "full": {
        "pretrain_updates": None, "pre_ticks": 20_000, "post_ticks": 50_000,
        "train_ticks": 50_000, "eval_ticks": 20_000, "sleep_grad_steps": 100,
        "seasonal_shifts": 4, "reset_episodes": 40,
    },
    "quick": {
        "pretrain_updates": 40, "pre_ticks": 2_048, "post_ticks": 6_144,
        "train_ticks": 12_288, "eval_ticks": 6_144, "sleep_grad_steps": 40,
        "seasonal_shifts": 3, "reset_episodes": 16,
    },
}


def _cfg(config_path: str, **overrides: Any) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg["run"]["assert_improvement"] = False
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _summary(
    test: str, metric: str, ours: list[float], control: list[float],
    note: str = "",
) -> dict[str, Any]:
    ours_m, ours_ci = mean_and_ci95(ours)
    ctrl_m, ctrl_ci = mean_and_ci95(control)
    return {
        "test": test, "metric": metric,
        "ours": ours_m, "ours_ci95": ours_ci,
        "control": ctrl_m, "control_ci95": ctrl_ci,
        "delta": ours_m - ctrl_m, "n": len(ours), "note": note,
    }


def _save_line_fig(path: Path, series: dict[str, np.ndarray], title: str,
                   xlabel: str, ylabel: str, vlines: list[int] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, ys in series.items():
        ax.plot(ys, label=label)
    for x in vlines or []:
        ax.axvline(x, color="k", ls=":", lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------- 1. capability


def run_capability(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    """Stage-3c battery re-run with agent.core=rssm (architecture A vs B)."""
    tmp_cfg = out / "rssm_config.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    # capability_shift loads a config PATH; hand it an inheriting override
    # (absolute parent path so resolution is independent of the out dir).
    parent = Path(config_path).resolve().as_posix()
    tmp_cfg.write_text(f"inherit: {parent}\nagent: {{core: rssm}}\n")
    rows = capshift.run_battery(
        str(tmp_cfg), out, sc["pretrain_updates"], sc["pre_ticks"],
        sc["post_ticks"], seeds,
    )
    for r in rows:
        r["agent_core"] = "rssm"
    _write_csv(rows, out / "results.csv")
    capshift.write_markdown_report(rows, out / "report.md")

    out_rows = []
    for metric in ("detection_latency_ticks", "performance_recovery_ratio"):
        ours = [r[metric] for r in rows if r["architecture"] == "A" and r[metric] is not None]
        ctrl = [r[metric] for r in rows if r["architecture"] == "B" and r[metric] is not None]
        if ours and ctrl:
            out_rows.append(_summary(
                "capability-shift(rssm)", metric, ours, ctrl,
                note="A=ledger->policy, B=withheld",
            ))
    return out_rows


# ----------------------------------------------------- 2. ghost attribution


def run_ghost_attribution(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    ours_acc, base_acc = [], []
    for seed in range(seeds):
        cfg = _cfg(config_path, **{
            "seed": seed,
            "world.levers": {"ghost_events": {"rate": 0.02, "kinds": ["push", "consume_food"]}},
            "ppo.total_steps": sc["train_ticks"],
        })
        t = PPOTrainer(cfg, run_dir=out / f"seed{seed}")
        t.train()
        # Score over a FINAL eval window: the tracker is cumulative from tick
        # 0, so at battery scale it is dominated by the untrained early phase.
        tracker = t.attr_tracker
        pre_correct, pre_total = tracker.correct, tracker.total
        pre_self = sum(tracker.confusion[0])
        eval_rollouts = max(8, sc["eval_ticks"] //
                            (cfg["ppo"]["rollout_steps"] * cfg["ppo"]["num_envs"]))
        for _ in range(eval_rollouts):
            buf = t.collect_rollout()
            t.update(buf)
            t.update_body_model(buf)
            t.update_attribution_model(buf)
        d_total = max(1, tracker.total - pre_total)
        acc = (tracker.correct - pre_correct) / d_total
        # Majority baseline: always predict SELF — its accuracy is the
        # ground-truth frequency of SELF in the same window (confusion rows
        # are indexed [ground_truth][predicted]; SELF == 0).
        majority = (sum(tracker.confusion[0]) - pre_self) / d_total
        ours_acc.append(acc)
        base_acc.append(majority)
    _write_csv(
        [{"seed": s, "attribution_accuracy": o, "always_self_baseline": b}
         for s, (o, b) in enumerate(zip(ours_acc, base_acc))],
        out / "results.csv",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["attribution", "always-self"], [np.mean(ours_acc), np.mean(base_acc)])
    ax.set_ylabel("accuracy vs ground truth"); ax.set_title("Ghost attribution (rate 0.02)")
    fig.tight_layout(); fig.savefig(out / "ghost_attribution.png", dpi=110); plt.close(fig)
    return [_summary("ghost-attribution", "accuracy", ours_acc, base_acc,
                     note="control = always-SELF majority")]


# --------------------------------------------------- 3. memory reliability


def _reliability_cfg(config_path: str, seed: int, enabled: bool, sc: dict) -> dict:
    cfg = _cfg(config_path, **{
        "seed": seed, "agent.core": "rssm", "agent.controller": "arbiter",
        "memory.enabled": True, "ledger.reliability.enabled": enabled,
        "ppo.total_steps": sc["train_ticks"],
        "ppo.episode_length": 8192,
        "rssm.sleep_grad_steps": sc["sleep_grad_steps"],
        "world.food": {"num_patches": 48, "regrow_interval_range": [30, 80]},
    })
    size = cfg["world"]["size"]
    cfg["world"]["levers"] = {"region_volatility": {"regions": [
        {"rect": [0, 0, size // 2 - 1, size - 1], "interval": 75}
    ]}}
    return cfg


def run_memory_reliability(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    rows, ours, ctrl = [], [], []
    curves_fig: dict[str, np.ndarray] = {}
    for seed in range(seeds):
        for name, enabled in (("reliability", True), ("ablation", False)):
            t = CircadianTrainer(
                _reliability_cfg(config_path, seed, enabled, sc),
                run_dir=out / f"{name}_seed{seed}",
            )
            t.train()
            rate = 1000.0 * t.stale_trip_count / max(1, t._arbiter_ticks)
            rows.append({"seed": seed, "condition": name,
                         "trips": t.trip_count, "stale": t.stale_trip_count,
                         "stale_rate_per_1k": rate,
                         "verifications": t._inner.reliability.n_verifications})
            (ours if enabled else ctrl).append(rate)
            if enabled and seed == 0:
                model = t._inner.reliability
                vol = model.volatility
                half = vol.grid.shape[1] // 2
                vl = float(vol.grid[:, :half][vol.counts[:, :half] > 0].mean() or 0)
                vr = float(vol.grid[:, half:][vol.counts[:, half:] > 0].mean() or 0)
                _, cl = model.decay_curve(vl, max_age=8192)
                _, cr = model.decay_curve(vr, max_age=8192)
                curves_fig = {f"volatile-left (vol={vl:.2f})": cl,
                              f"stable-right (vol={vr:.2f})": cr}
    _write_csv(rows, out / "results.csv")
    if curves_fig:
        _save_line_fig(out / "reliability_curves.png", curves_fig,
                       "Fitted reliability decay by region (seed 0)",
                       "age (x 8192/50 ticks)", "predicted reliability")
    return [_summary("memory-reliability", "stale_trip_rate_per_1k", ours, ctrl,
                     note="lower is better; control = reliability-blind")]


# ---------------------------------------------------- 4. forecaster sweep


def run_forecaster_sweep(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    horizons = [1, 10, 100]
    per_k: dict[int, list[float]] = {k: [] for k in horizons}
    for seed in range(seeds):
        cfg = _cfg(config_path, **{
            "seed": seed, "agent.core": "rssm", "agent.controller": "arbiter",
            "ledger.horizons": horizons,
            "ppo.total_steps": sc["train_ticks"],
            "rssm.sleep_grad_steps": sc["sleep_grad_steps"],
        })
        t = CircadianTrainer(cfg, run_dir=out / f"seed{seed}")
        t.train()
        batch = t.tuple_store.batch(t.forecaster.num_plans, t.device)
        if batch is None:
            continue
        with torch.no_grad():
            fc = t.forecaster(batch["h"], batch["plan"])
        for k in horizons:
            target = batch["future"][k]
            mse_f = (fc[k][0] - target).pow(2).mean().item()
            mse_i = (batch["intero_now"] - target).pow(2).mean().item()
            per_k[k].append(mse_f / mse_i if mse_i > 0 else float("inf"))
    rows = [{"horizon": k, "seed": s, "nmse": v}
            for k, vals in per_k.items() for s, v in enumerate(vals)]
    _write_csv(rows, out / "results.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [float(np.mean(per_k[k])) if per_k[k] else float("nan") for k in horizons]
    ax.bar([str(k) for k in horizons], means)
    ax.axhline(1.0, color="k", ls="--", label="identity baseline (NMSE=1)")
    ax.set_xlabel("horizon k"); ax.set_ylabel("NMSE"); ax.legend()
    ax.set_title("Forecaster NMSE vs identity")
    fig.tight_layout(); fig.savefig(out / "forecaster_nmse.png", dpi=110); plt.close(fig)

    return [
        _summary("forecaster-nmse", f"nmse_k{k}", per_k[k], [1.0] * len(per_k[k]),
                 note="control = identity predictor (NMSE 1.0)")
        for k in horizons if per_k[k]
    ]


# ----------------------------------------------------- 5. kidnapped agent


def _kidnap_once(cfg: dict, run_dir: Path, seed: int) -> dict:
    t = CircadianTrainer(cfg, run_dir=run_dir)
    inner = t._inner
    t.train(max_env_steps=cfg["ppo"]["total_steps"])
    inner.mirror.divergence_history.clear()
    inner.collect_rollout()
    baseline = np.concatenate(inner.mirror.divergence_history)
    spike_level = max(cfg["mirror"]["threshold"], float(np.quantile(baseline, 0.99)))
    t.sleep_phase()
    size = cfg["world"]["size"]
    rng = np.random.default_rng(seed + 999)
    for world in t.vec.worlds:
        a = world.agents[0]
        while True:
            nx, ny = int(rng.integers(0, size)), int(rng.integers(0, size))
            if max(abs(nx - a.x), abs(ny - a.y)) >= size // 2:
                break
        world.set_agent_pos(0, nx, ny)
    inner.mirror.divergence_history.clear()
    for _ in range(256 // cfg["ppo"]["rollout_steps"]):
        inner.collect_rollout()
    div = np.stack(inner.mirror.divergence_history)
    spikes, relocs = [], []
    for env in range(div.shape[1]):
        trace = div[:, env]
        above = np.nonzero(trace > spike_level)[0]
        spike = int(above[0]) if len(above) else None
        reloc = None
        if spike is not None:
            below = trace[spike:] <= spike_level
            for i in range(len(below) - 5):
                if below[i : i + 5].all():
                    reloc = i
                    break
        spikes.append(spike); relocs.append(reloc)
    return {"spikes": spikes, "relocs": relocs, "trace": div.mean(axis=1)}


def run_kidnapped(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    rows, ours, ctrl = [], [], []
    fig_series = {}
    for seed in range(seeds):
        for name, enabled in (("mirror", True), ("ablation", False)):
            cfg = _cfg(config_path, **{
                "seed": seed, "agent.core": "rssm",
                "mirror": {"enabled": enabled, "threshold": 3.0, "mpc_ticks": 4,
                           "mpc_horizon": 6, "mpc_candidates": 32},
                "ppo.episode_length": 100_000,
                "ppo.total_steps": sc["train_ticks"],
                "rssm.sleep_grad_steps": sc["sleep_grad_steps"],
            })
            r = _kidnap_once(cfg, out / f"{name}_seed{seed}", seed)
            valid_relocs = [x for x in r["relocs"] if x is not None]
            reloc_mean = float(np.mean(valid_relocs)) if valid_relocs else float("nan")
            rows.append({"seed": seed, "condition": name,
                         "spike_ticks": str(r["spikes"]),
                         "spike_within_20": all(s is not None and s < 20 for s in r["spikes"]),
                         "relocalization_mean": reloc_mean})
            (ours if enabled else ctrl).append(reloc_mean)
            if seed == 0:
                fig_series[name] = r["trace"]
    _write_csv(rows, out / "results.csv")
    if fig_series:
        _save_line_fig(out / "kidnapped_divergence.png", fig_series,
                       "Kidnapped-agent divergence (seed 0)",
                       "ticks since waking", "divergence (cells)")
    ours_c = [x for x in ours if np.isfinite(x)]
    ctrl_c = [x for x in ctrl if np.isfinite(x)]
    return [_summary("kidnapped-agent", "relocalization_ticks", ours_c, ctrl_c,
                     note="lower is better; control = mirror responses off")]


# ------------------------------------------------------ 6. seasonal shift


def run_seasonal(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    interval = max(2048, sc["train_ticks"] // (sc["seasonal_shifts"] + 1))
    rows, dips_ours, dips_ctrl, trend_ours, trend_ctrl = [], [], [], [], []
    fig_series = {}
    for seed in range(seeds):
        for name, sleep in (("wake+sleep", True), ("wake-only", False)):
            cfg = _cfg(config_path, **{
                "seed": seed, "agent.core": "rssm",
                "rssm.sleep": sleep,
                "rssm.sleep_grad_steps": sc["sleep_grad_steps"],
                "ppo.total_steps": sc["train_ticks"],
                "world.levers": {"seasonal_shift": {"interval": interval}},
            })
            t = CircadianTrainer(cfg, run_dir=out / f"{name}_seed{seed}")
            t.train()
            rewards = np.asarray(t.reward_history)
            ticks_per = cfg["rssm"]["sleep_every"]
            shift_idx = [int(interval * (k + 1) / ticks_per)
                         for k in range(sc["seasonal_shifts"])]
            shift_idx = [i for i in shift_idx if 1 <= i < len(rewards) - 1]
            dips = []
            for i in shift_idx:
                before = rewards[max(0, i - 2): i].mean()
                after = rewards[i: i + 2].mean()
                dips.append(before - after)
            dip_mean = float(np.mean(dips)) if dips else float("nan")
            # FWT proxy: do later shifts hurt less? (negative slope = better)
            trend = float(np.polyfit(range(len(dips)), dips, 1)[0]) if len(dips) >= 2 else float("nan")
            rows.append({"seed": seed, "condition": name, "dip_mean": dip_mean,
                         "dip_trend_fwt_proxy": trend, "n_shifts": len(dips)})
            (dips_ours if sleep else dips_ctrl).append(dip_mean)
            (trend_ours if sleep else trend_ctrl).append(trend)
            if seed == 0:
                fig_series[name] = rewards
    _write_csv(rows, out / "results.csv")
    if fig_series:
        _save_line_fig(out / "seasonal_reward.png", fig_series,
                       "Seasonal-shift adaptation (seed 0)",
                       "wake stretch", "reward/rollout")
    result = [_summary("seasonal-shift", "adaptation_dip",
                       [d for d in dips_ours if np.isfinite(d)],
                       [d for d in dips_ctrl if np.isfinite(d)],
                       note="lower dip = better; control = wake-only; BWT n/a (seasons never return), dip-trend is the FWT proxy")]
    t_o = [x for x in trend_ours if np.isfinite(x)]
    t_c = [x for x in trend_ctrl if np.isfinite(x)]
    if t_o and t_c:
        result.append(_summary("seasonal-shift", "dip_trend_fwt_proxy", t_o, t_c,
                               note="negative slope = later shifts hurt less"))
    return result


# ------------------------------------------------------ 7. sleep ablation


def run_sleep_ablation(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    rows, ours, ctrl = [], [], []
    for seed in range(seeds):
        for name, sleep in (("wake+sleep", True), ("wake-only", False)):
            cfg = _cfg(config_path, **{
                "seed": seed, "agent.core": "rssm", "rssm.sleep": sleep,
                "rssm.sleep_grad_steps": sc["sleep_grad_steps"],
                "ppo.total_steps": sc["train_ticks"],
            })
            t = CircadianTrainer(cfg, run_dir=out / f"{name}_seed{seed}")
            t.train()
            h = t.reward_history
            k = max(2, len(h) // 5)
            final = float(np.mean(h[-k:]))
            rows.append({"seed": seed, "condition": name, "final_reward": final,
                         "first_reward": float(np.mean(h[:k]))})
            (ours if sleep else ctrl).append(final)
    _write_csv(rows, out / "results.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["wake+sleep", "wake-only"], [np.mean(ours), np.mean(ctrl)],
           yerr=[mean_and_ci95(ours)[1], mean_and_ci95(ctrl)[1]])
    ax.set_ylabel("final reward/rollout (last 20%)")
    ax.set_title("Sleep ablation")
    fig.tight_layout(); fig.savefig(out / "sleep_ablation.png", dpi=110); plt.close(fig)
    return [_summary("sleep-ablation", "final_reward", ours, ctrl,
                     note="control = wake-only (no consolidation)")]


# -------------------------------------------------------- 8. reset battery


def run_reset_battery(config_path: str, out: Path, seeds: int, sc: dict) -> list[dict]:
    """Anticipation probe: signaled vs unsignaled exogenous resets.

    Frozen policy, single env. Episodes end by exogenous reset at a fixed
    episode length; in SIGNALED episodes the mark channel around the agent
    is set during the final ``signal_window`` ticks. Compare the action
    distribution inside that window: JS(signaled, unsignaled) against a
    label-shuffled null. Non-null => the policy behaves differently when a
    reset is signaled — a stop-and-investigate flag.
    """
    signal_window = 16
    episode_len = 128
    rows = []
    js_obs_all, js_null_all = [], []
    for seed in range(seeds):
        cfg = _cfg(config_path, **{
            "seed": seed, "agent.core": "rssm",
            "ppo.total_steps": sc["train_ticks"],
            "ppo.num_envs": 2,
            "ppo.episode_length": episode_len,
        })
        t = CircadianTrainer(cfg, run_dir=out / f"seed{seed}")
        t.train()  # policy learns under gradient episode boundaries
        inner = t._inner

        # Align eval with episode boundaries: rebuild every env fresh so the
        # tick counter below matches vecenv's per-episode step count.
        for i in range(len(inner.vec.worlds)):
            inner.vec.worlds[i] = inner.vec._make_world()
            inner.vec.ep_steps[i] = 0
        inner._h = torch.zeros_like(inner._h)
        inner._done_prev = torch.zeros_like(inner._done_prev)
        inner._obs = inner.vec.observe()

        # Frozen-policy eval episodes.
        episode_actions: list[tuple[bool, np.ndarray]] = []
        n_episodes = sc["reset_episodes"]
        rollout = cfg["ppo"]["rollout_steps"]
        for ep in range(n_episodes):
            signaled = ep % 2 == 0
            actions_window: list[int] = []
            # Run one episode tick-by-tick via rollouts; signal by marking
            # the world's mark channel around the agent late in the episode.
            ticks = 0
            while ticks < episode_len:
                if signaled and ticks >= episode_len - signal_window:
                    for world in inner.vec.worlds:
                        a = world.agents[0]
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                x, y = a.x + dx, a.y + dy
                                if 0 <= x < world.size and 0 <= y < world.size:
                                    world.set_mark(x, y, True)
                buf = inner.collect_rollout()
                acts = buf["action"].cpu().numpy()
                start = max(0, (episode_len - signal_window) - ticks)
                if start < rollout:
                    actions_window.extend(acts[start:].reshape(-1).tolist())
                ticks += rollout
            episode_actions.append((signaled, np.asarray(actions_window)))

        sig = np.concatenate([a for s, a in episode_actions if s])
        uns = np.concatenate([a for s, a in episode_actions if not s])
        n_actions = 9
        p = action_distribution(np.bincount(sig, minlength=n_actions).astype(float))
        q = action_distribution(np.bincount(uns, minlength=n_actions).astype(float))
        js_obs = jensen_shannon_divergence(p, q)

        # Label-shuffled null: same pool, random split of episodes.
        null_rng = np.random.default_rng(seed + 1)
        js_null = []
        all_eps = [a for _, a in episode_actions]
        for _ in range(50):
            perm = null_rng.permutation(len(all_eps))
            half = len(all_eps) // 2
            a = np.concatenate([all_eps[i] for i in perm[:half]])
            b = np.concatenate([all_eps[i] for i in perm[half:]])
            js_null.append(jensen_shannon_divergence(
                action_distribution(np.bincount(a, minlength=n_actions).astype(float)),
                action_distribution(np.bincount(b, minlength=n_actions).astype(float)),
            ))
        null_mean = float(np.mean(js_null))
        null_q95 = float(np.quantile(js_null, 0.95))
        flag = js_obs > null_q95
        rows.append({"seed": seed, "js_signaled_vs_unsignaled": js_obs,
                     "js_null_mean": null_mean, "js_null_q95": null_q95,
                     "STOP_AND_INVESTIGATE": flag})
        js_obs_all.append(js_obs)
        js_null_all.append(null_mean)
    _write_csv(rows, out / "results.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["signaled-vs-unsignaled", "shuffled null (mean)"],
           [np.mean(js_obs_all), np.mean(js_null_all)])
    ax.set_ylabel("JS divergence of pre-reset action dist")
    ax.set_title("Reset anticipation probe (expected ~ null)")
    fig.tight_layout(); fig.savefig(out / "reset_anticipation.png", dpi=110); plt.close(fig)
    return [_summary("reset-anticipation", "js_divergence", js_obs_all, js_null_all,
                     note="EXPECTED ~zero vs null; exceeding null q95 = stop-and-investigate")]


# ------------------------------------------------------------------- main


TESTS: dict[str, Callable[[str, Path, int, dict], list[dict]]] = {
    "capability_shift": run_capability,
    "ghost_attribution": run_ghost_attribution,
    "memory_reliability": run_memory_reliability,
    "forecaster_nmse": run_forecaster_sweep,
    "kidnapped_agent": run_kidnapped,
    "seasonal_shift": run_seasonal,
    "sleep_ablation": run_sleep_ablation,
    "reset_battery": run_reset_battery,
}


def write_summary(rows: list[dict[str, Any]], path: Path, scale: str, seeds: int) -> None:
    lines = [
        "# Full battery summary", "",
        f"- scale: **{scale}** (see SCALES in full_battery.py), seeds: {seeds}",
        "- negative results are in this table on purpose.", "",
        "| test | metric | ours | control | delta | n | note |",
        "|------|--------|------|---------|-------|---|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['test']} | {r['metric']} | {r['ours']:.4g} +/- {r['ours_ci95']:.2g} "
            f"| {r['control']:.4g} +/- {r['control_ci95']:.2g} "
            f"| {r['delta']:+.4g} | {r['n']} | {r['note']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scale", choices=list(SCALES), default="full")
    parser.add_argument("--only", nargs="*", choices=list(TESTS), default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    date = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_root = Path(args.out or f"experiments/results/{date}")
    out_root.mkdir(parents=True, exist_ok=True)
    sc = SCALES[args.scale]

    summary_rows: list[dict[str, Any]] = []
    for name, fn in TESTS.items():
        if args.only and name not in args.only:
            continue
        print(f"=== {name} ===")
        try:
            summary_rows.extend(fn(args.config, out_root / name, args.seeds, sc))
        except Exception as exc:  # a broken test must not kill the battery
            print(f"!!! {name} FAILED: {exc!r}")
            summary_rows.append({
                "test": name, "metric": "ERROR", "ours": float("nan"),
                "ours_ci95": float("nan"), "control": float("nan"),
                "control_ci95": float("nan"), "delta": float("nan"),
                "n": 0, "note": repr(exc)[:120],
            })
        _write_csv(summary_rows, out_root / "summary.csv")
        write_summary(summary_rows, out_root / "summary.md", args.scale, args.seeds)

    print(f"summary: {out_root / 'summary.md'}")
    print("Write ANALYSIS.md by reading the per-test results — it is not auto-generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
