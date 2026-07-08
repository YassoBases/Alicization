"""Researcher-value battery (P9.4): does following the agenda buy anything?

Three arms select K target regions under an IDENTICAL compute budget and
"execute" one targeted-replay item per region (extra world-model grad
steps on replay sequences drawn from that region — the one menu item
executable offline with existing machinery; directed visits and probe
batches need wake-phase control and are out of scope for this harness,
which is stated in the report):

  1. agenda    — the configured ranker's top-K region items (v2 EIG by
                 default; researcher.ranker selects the v1 ablation)
  2. random    — K uniformly random visited regions
  3. greedy    — K regions with the highest RAW measured disagreement,
                 no tractability term (the noisy-TV-vulnerable control)

Every arm starts from the same trained CircadianTrainer state (model +
optimizer snapshots restored between arms — the trainer holds open log
handles and cannot be deepcopied), so arms differ ONLY in target
selection. A drift-control pass receives the same number of grad steps
on uniformly-sampled replay; Section-21 uncertainty reduction is
drift-corrected against it per region.

Metrics per Section 21 (experiments/metrics.py): drift-corrected
uncertainty reduction per item, EIG calibration (predicted vs realized,
Spearman + scatter) for the agenda arm, agenda stability (Kendall tau
before/after execution), contradiction-detection latency for the life
run's lever (a registry property, reported once per seed). Region
disagreement is measured directly: mean ensemble epistemic variance over
encoded replay contexts inside the region, averaged over all actions —
the live epistemic map only updates during rollouts, which offline grad
steps do not produce.

Output: experiments/results/<date>/researcher_value/ with CSVs,
summary.md, and the calibration scatter. ANALYSIS.md is written by a
human from the results — never auto-generated.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.metrics import (  # noqa: E402
    agenda_stability_kendall_tau,
    contradiction_detection_latency,
    eig_calibration,
    mean_and_ci95,
    uncertainty_reduction_per_item,
)
from experiments.model_adapter import RSSMAdapter  # noqa: E402
from researcher.agenda import rank_v1  # noqa: E402
from researcher.eig import rank_v2  # noqa: E402
from researcher.questions import generate_questions  # noqa: E402
from researcher.registry import (  # noqa: E402
    HypothesisRegistry,
    QueryEngine,
    build_default_hypotheses,
)
from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from world.config import load_config  # noqa: E402

MOVE_E = 2

SCALES = {
    "full": {"life_ticks": 200_000, "lever_tick": 25_000, "k_items": 8,
             "steps_per_item": 100},
    # lever_tick vs monitor geometry (window 1000, cadence 250, arm_after
    # 500 -> first armed check at 2500): mean_shift is an ONSET detector —
    # it only violates while the now-window straddles the change and the
    # prev-window is still clean, so two consecutive violating checks exist
    # only if the lever falls in (first_check - window + cadence,
    # first_check). At 1500 the arm gate ate that window and every seed
    # censored; 2000 puts checks 2500/2750 inside it (latency ~750).
    "quick": {"life_ticks": 12_288, "lever_tick": 2_000, "k_items": 4,
              "steps_per_item": 20},
}


def _life_cfg(config_path: str, seed: int, sc: dict[str, int]) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg["seed"] = seed
    cfg["agent"]["core"] = "rssm"
    cfg["ppo"]["total_steps"] = sc["life_ticks"]
    cfg["ppo"]["episode_length"] = 100_000
    cfg["world"]["levers"] = {"capability_shift": [
        {"action": MOVE_E, "start": sc["lever_tick"], "end": None,
         "fail_prob": 0.9}
    ]}
    cfg["run"]["assert_improvement"] = False
    return cfg


# --------------------------------------------------------------- measurement


def measure_disagreement(trainer: CircadianTrainer, region: tuple[int, int],
                         seed: int = 0) -> float:
    """Mean ensemble epistemic variance over encoded replay contexts inside
    ``region``, averaged over all actions. NaN if the region is unvisited."""
    adapter = RSSMAdapter(trainer, seed=seed)
    feats = adapter._region_start_features(region)
    if feats is None:
        return float("nan")
    core = trainer.model.core
    num_actions = core.num_actions
    total = 0.0
    with torch.no_grad():
        for a in range(num_actions):
            onehot = F.one_hot(
                torch.full((feats.shape[0],), a, dtype=torch.long,
                           device=feats.device),
                num_actions).float()
            _, epistemic, _ = core.ensemble_stats(feats, onehot)
            total += float(epistemic.mean())
    return total / num_actions


def visited_regions(trainer: CircadianTrainer, region_size: int = 8) -> list[tuple[int, int]]:
    replay = trainer.replay
    if not replay._filled:
        return []
    size = trainer._inner.epistemic_map.shape[0]
    pos = replay.position[:, :replay._filled].reshape(-1, 2)
    cells = np.clip((pos * size).astype(int), 0, size - 1)
    regions = {(int(y // region_size), int(x // region_size))
               for x, y in cells}
    return sorted(regions)


# ----------------------------------------------------------------- execution


def targeted_replay_steps(trainer: CircadianTrainer,
                          region: tuple[int, int] | None,
                          grad_steps: int, region_size: int = 8) -> None:
    """Extra world-model grad steps on replay sequences biased toward
    ``region`` (None = uniform: the drift-control arm). Bias is applied by
    temporarily boosting priorities of in-region transitions — the same
    proportional sampler the sleep phase uses, so arms differ only in the
    sampling distribution."""
    replay = trainer.replay
    saved = replay.priority.copy()
    if region is not None:
        size = trainer._inner.epistemic_map.shape[0]
        pos = replay.position[:, :replay._filled]
        cells = np.clip((pos * size).astype(int), 0, size - 1)
        in_region = ((cells[..., 1] // region_size == region[0])
                     & (cells[..., 0] // region_size == region[1]))
        boost = np.ones_like(replay.priority)
        boost[:, :replay._filled][in_region] = 100.0
        replay.priority *= boost
    r = trainer.rcfg
    try:
        for _ in range(grad_steps):
            batch = replay.sample(r["batch_seqs"], r["seq_len"], trainer.device)
            if batch is None:
                return
            core = trainer.model.core
            horizon, b = batch["grid"].shape[0], batch["grid"].shape[1]
            flat_g = batch["grid"].reshape(horizon * b, *trainer.vec.grid_shape)
            flat_i = batch["intero"].reshape(horizon * b, -1)
            embeds = trainer.model.encoder(flat_g, flat_i).reshape(horizon, b, -1)
            wm = core.world_model_loss(
                embeds, core.initial_state(b, trainer.device),
                batch["done"], batch["action"], batch["grid"],
                batch["intero"], rewards=batch["reward"],
                positions=batch["position"],
            )
            trainer.world_opt.zero_grad()
            wm["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in trainer.world_opt.param_groups for p in g["params"]],
                trainer.pcfg["max_grad_norm"],
            )
            trainer.world_opt.step()
    finally:
        replay.priority = saved


# ------------------------------------------------------------------- arms


def select_targets(arm: str, trainer: CircadianTrainer, run_dir: Path,
                   registry: HypothesisRegistry, cfg: dict[str, Any],
                   k: int, seed: int) -> tuple[list[tuple[int, int]], dict[str, float], list[str]]:
    """-> (regions, predicted_gain per region key, agenda id order)."""
    rng = np.random.default_rng(seed)
    known = visited_regions(trainer)
    if arm == "random":
        idx = rng.choice(len(known), min(k, len(known)), replace=False)
        return [known[i] for i in idx], {}, []
    if arm == "greedy":
        scored = [(measure_disagreement(trainer, reg, seed), reg)
                  for reg in known]
        scored = [(d, reg) for d, reg in scored if not np.isnan(d)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [reg for _, reg in scored[:k]], {}, []
    # agenda arm: configured ranker over real questions. top_k_cells is
    # widened so the arm can fill its K-region budget (the default 3 world-
    # uncertainty questions left the arm executing fewer items than the
    # other arms — an unequal total budget).
    questions = generate_questions(run_dir, registry, top_k_cells=2 * k)
    competence = trainer.last_competence_report
    rcfg = cfg.get("researcher", {})
    if rcfg.get("ranker", "v2") == "v2":
        items = rank_v2(questions, [], competence,
                        adapter=RSSMAdapter(trainer, seed=seed),
                        visit_steps=int(rcfg.get("visit_steps", 15)))
    else:
        items = rank_v1(questions, [], competence)
    order = [i.id for i in items]
    regions, gains = [], {}
    for item in items:
        q = next((q for q in questions if q.id == item.ref), None)
        if q is None or q.region is None or tuple(q.region) in regions:
            continue
        regions.append(tuple(q.region))
        if item.predicted_gain is not None:
            gains[str(tuple(q.region))] = float(item.predicted_gain)
        if len(regions) == k:
            break
    return regions, gains, order


# ------------------------------------------------------------------- battery


def run_seed(seed: int, sc: dict[str, int], config: str,
             run_root: str) -> dict[str, Any]:
    cfg = _life_cfg(config, seed, sc)
    run_dir = Path(create_run_dir(run_root))
    trainer = CircadianTrainer(cfg, run_dir=run_dir)
    trainer.train()

    # Registry pass (contradiction latency; also feeds agenda questions).
    registry = HypothesisRegistry(run_dir)
    for h in build_default_hypotheses(
            world_size=cfg["world"]["size"], num_actions=9):
        h.monitor["window"] = 1000
        if "arm_after" in h.monitor:
            h.monitor["arm_after"] = 500
        registry.add(h)
    engine = QueryEngine(run_dir)
    max_tick = sc["life_ticks"] // cfg["ppo"]["num_envs"]
    for now_tick in range(250, max_tick + 1, 250):
        registry.check_all(engine, now_tick)
    target_h = registry.hypotheses[f"hyp-capability-success-{MOVE_E}"]
    contra = next((t["tick"] for t in target_h.transitions
                   if t["to"] == "contradicted"), None)
    latency = contradiction_detection_latency(sc["lever_tick"], contra)

    out: dict[str, Any] = {"seed": seed, "run": run_dir.name,
                           "latency": latency, "arms": {}}
    k, steps = sc["k_items"], sc["steps_per_item"]

    # Base-state snapshot: arms and the drift control all start here.
    snap = {"model": copy.deepcopy(trainer.model.state_dict()),
            "opt": copy.deepcopy(trainer.world_opt.state_dict())}

    def restore() -> None:
        trainer.model.load_state_dict(snap["model"])
        trainer.world_opt.load_state_dict(snap["opt"])

    # All selections happen on the base state, before anything trains.
    selections: dict[str, tuple[list[tuple[int, int]], dict[str, float], list[str]]] = {}
    for arm in ("agenda", "random", "greedy"):
        selections[arm] = select_targets(
            arm, trainer, run_dir, registry, cfg, k, seed)
    all_regions = sorted({r for regs, _, _ in selections.values() for r in regs})

    # Drift control: same budget, uniform sampling.
    ctrl_before = {r: measure_disagreement(trainer, r, seed) for r in all_regions}
    targeted_replay_steps(trainer, None, k * steps)
    ctrl_after = {r: measure_disagreement(trainer, r, seed) for r in all_regions}
    restore()

    for arm, (regions, gains, order) in selections.items():
        before = {r: measure_disagreement(trainer, r, seed) for r in regions}
        items = []
        for region in regions:
            targeted_replay_steps(trainer, region, steps)
            after = measure_disagreement(trainer, region, seed)
            red = uncertainty_reduction_per_item(
                before[region], after,
                ctrl_before[region], ctrl_after[region])
            items.append({"region": list(region),
                          "before": before[region], "after": after,
                          "reduction": red,
                          "predicted_gain": gains.get(str(region))})
        entry: dict[str, Any] = {"items": items}
        if arm == "agenda" and order:
            # Stability: re-rank on the post-execution state.
            _, _, order2 = select_targets(
                "agenda", trainer, run_dir, registry, cfg, k, seed)
            entry["kendall_tau"] = agenda_stability_kendall_tau(order, order2)
        out["arms"][arm] = entry
        restore()
    return out


def summarize(results: list[dict[str, Any]], out_dir: Path,
              sc_name: str, seeds: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in results:
        for arm, entry in res["arms"].items():
            for it in entry["items"]:
                rows.append({"seed": res["seed"], "arm": arm,
                             "region": tuple(it["region"]),
                             "reduction": it["reduction"],
                             "predicted_gain": it["predicted_gain"]})
    import csv
    with open(out_dir / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Researcher-value battery",
        "",
        "**Question: does following the agenda reduce more uncertainty per "
        "unit budget than random or greedy raw-uncertainty selection?** "
        "Written either way — a null belongs here with the same prominence.",
        "",
        f"- scale: **{sc_name}**, seeds: {seeds}; execution operationalized "
        "as targeted replay only (directed visits / probe batches need "
        "wake-phase control; see module docstring)",
        "- reductions are drift-corrected against a same-budget uniform-"
        "replay control (Section 21)",
        "",
        "| arm | n items | uncertainty reduction (mean +/- CI95) |",
        "|-----|---------|----------------------------------------|",
    ]
    for arm in ("agenda", "random", "greedy"):
        vals = [r["reduction"] for r in rows
                if r["arm"] == arm and not np.isnan(r["reduction"])]
        mean, ci = mean_and_ci95(vals)
        lines.append(f"| {arm} | {len(vals)} | {mean:+.6f} +/- {ci:.6f} |")

    pred = [r["predicted_gain"] for r in rows
            if r["arm"] == "agenda" and r["predicted_gain"] is not None]
    real = [r["reduction"] for r in rows
            if r["arm"] == "agenda" and r["predicted_gain"] is not None]
    cal = eig_calibration(pred, real)
    lines += ["", f"- EIG calibration (agenda arm): Spearman rho = "
              f"{cal['spearman']:.3f} over n = {int(cal['n'])} items"]
    taus = [res["arms"]["agenda"].get("kendall_tau") for res in results]
    taus = [t for t in taus if t is not None and not np.isnan(t)]
    if taus:
        lines.append(f"- agenda stability across execution (descriptive): "
                     f"Kendall tau = {np.mean(taus):.3f}")
    lats = [res["latency"] for res in results]
    lines += ["", "## Contradiction-detection latency (per seed)", ""]
    for res, lat in zip(results, lats):
        cens = " (censored)" if lat["censored"] else ""
        lines.append(f"- seed {res['seed']}: {lat['latency']:.0f} ticks{cens}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    if len(pred) >= 3:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(pred, real)
            ax.set_xlabel("predicted gain (EIG)")
            ax.set_ylabel("realized reduction (drift-corrected)")
            ax.set_title("EIG calibration — agenda arm")
            fig.savefig(out_dir / "eig_calibration.png", dpi=120,
                        bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # matplotlib backend issues: report, not fail
            print(f"calibration figure skipped: {exc}")
    print(f"summary: {out_dir / 'summary.md'}")
    print("Write ANALYSIS.md from the results — not auto-generated.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scale", choices=list(SCALES), default="full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sc = SCALES[args.scale]
    date = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out or f"experiments/results/{date}/researcher_value")
    run_root = str(out_dir / "runs")

    results = []
    for seed in range(args.seeds):
        print(f"=== seed {seed} ===")
        res = run_seed(seed, sc, args.config, run_root)
        results.append(res)
        (out_dir / f"seed{seed}.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / f"seed{seed}.json").write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8")
    summarize(results, out_dir, args.scale, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
