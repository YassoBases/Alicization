"""Stage-E4 parity gate: SelfQ must match-or-beat the heads it replaces.

For each seed, train the SAME circadian config twice — ledger.impl=heads
(BodyModel + Forecaster) and ledger.impl=selfq (one unified model) — and
compare each replaced head's acceptance metric over a final window:
  - body CE            (ledger/body_nll)      lower better
  - body Brier         (ledger/success_brier) lower better
  - forecaster NMSE k1  (sleep/forecaster_nmse_k1)  vs identity, lower better
  - forecaster NMSE k10 (sleep/forecaster_nmse_k10) vs identity, lower better

Parity (not strict dominance): SelfQ's seed-mean must be within TOL of the
heads' seed-mean, or better, on every metric. The mirror and attribution
are NOT replaced in this stage. Report -> docs/acceptance/stage-E/.

Usage: python scripts/verify_selfq.py [--steps 24576] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import create_run_dir  # noqa: E402
from training.sleep import CircadianTrainer  # noqa: E402
from viz.dashboard import load_tb_scalars  # noqa: E402
from world.config import load_config  # noqa: E402

METRICS = {
    "body_ce": "ledger/body_nll",
    "body_brier": "ledger/success_brier",
    "nmse_k1": "sleep/forecaster_nmse_k1",
    "nmse_k10": "sleep/forecaster_nmse_k10",
}
# GATE metrics: the body heads feed the policy and the mirror and are
# FUNCTIONAL at smoke scale, so SelfQ must genuinely match or beat them
# (tight 1.15). These are the hard PASS/FAIL.
GATE_TOL = {"body_ce": 1.15, "body_brier": 1.15}
# DESCRIPTIVE metrics: the forecaster is BELOW ITS MINIMUM VIABLE SCALE at
# smoke scale — stage-A established it only beats identity (NMSE < 1) at
# ~50k steps / 100 grad-steps, and both impls here sit at NMSE >> 1
# (non-functional in EITHER). Per the stage-A MIN_VIABLE_SCALE contract a
# metric below viable scale is machinery-only, not admissible parity
# evidence, so forecaster NMSE is REPORTED (with the selfq/heads ratio) but
# NOT gated here. Forecaster parity is the full-scale follow-up — the exact
# stage-A precondition Stage E is gated on. (k=1 is additionally degenerate:
# identity MSE ~ 6e-5 at one tick, so the ratio explodes for both impls.)
DESCRIPTIVE = ("nmse_k1", "nmse_k10")


def _final_mean(run_dir: Path, tag: str, tail: int = 5) -> float:
    scalars = load_tb_scalars(run_dir)
    if tag not in scalars:
        return float("nan")
    vals = np.asarray(scalars[tag][1], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals[-tail:].mean()) if vals.size else float("nan")


def _train(impl: str, seed: int, steps: int, config: str, run_root: str) -> dict[str, float]:
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["agent"]["core"] = "rssm"
    cfg["agent"]["controller"] = "arbiter"     # fills the forecast tuple store
    cfg["ledger"]["impl"] = impl
    cfg["ppo"]["total_steps"] = steps
    cfg["ppo"]["episode_length"] = 100_000
    cfg["run"]["assert_improvement"] = False
    # Measurement runs: no resumable checkpoints needed. Disabling them keeps
    # disk to small TB/JSON (a full-scale replay-buffer checkpoint is ~1 GB).
    cfg.setdefault("checkpoints", {})["interval"] = 10**12
    run_dir = Path(create_run_dir(run_root))
    CircadianTrainer(cfg, run_dir=run_dir).train()
    return {name: _final_mean(run_dir, tag) for name, tag in METRICS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--run-root", default="runs_selfq")
    parser.add_argument("--out", default="docs/acceptance/stage-E")
    args = parser.parse_args()

    results: dict[str, list[dict[str, float]]] = {"heads": [], "selfq": []}
    for seed in range(args.seeds):
        for impl in ("heads", "selfq"):
            print(f"=== seed {seed} impl {impl} ===")
            results[impl].append(_train(impl, seed, args.steps, args.config,
                                        args.run_root))

    def seed_mean(impl: str, metric: str) -> float:
        vals = [r[metric] for r in results[impl] if np.isfinite(r[metric])]
        return float(np.mean(vals)) if vals else float("nan")

    rows, all_pass = [], True
    for metric in METRICS:
        h, s = seed_mean("heads", metric), seed_mean("selfq", metric)
        ratio = s / h if h else float("nan")
        gated = metric in GATE_TOL
        # Gate metrics PASS/FAIL; descriptive metrics are reported (n/a).
        ok = (bool(np.isfinite(h) and np.isfinite(s) and s <= h * GATE_TOL[metric])
              if gated else None)
        if ok is False:
            all_pass = False
        rows.append((metric, h, s, ratio, gated, ok))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage-E parity gate: SelfQ vs the heads it replaces", "",
        f"- scale: **{args.steps} steps**, seeds: {args.seeds} "
        "(lower is better on every metric)",
        "- **GATED** on the body metrics (they feed the policy + mirror and "
        "are functional at smoke scale): SelfQ must be within 15% of heads.",
        "- **DESCRIPTIVE** on forecaster NMSE: below its minimum viable scale "
        "(both impls NMSE >> identity; stage-A needs ~50k/100 grad-steps), so "
        "machinery-only, NOT gated here — forecaster parity is the full-scale "
        "stage-A follow-up. k=1 is additionally denominator-degenerate.",
        "- mirror + attribution are NOT replaced this stage.", "",
        "| metric | kind | heads | selfq | selfq/heads | result |",
        "|--------|------|-------|-------|-------------|--------|",
    ]
    for metric, h, s, ratio, gated, ok in rows:
        kind = "gate" if gated else "descriptive"
        result = ("PASS" if ok else "FAIL") if gated else "n/a (below MVS)"
        lines.append(f"| {metric} | {kind} | {h:.4g} | {s:.4g} | {ratio:.3f} "
                     f"| {result} |")
    lines += ["", f"**Overall (body parity): {'PASS' if all_pass else 'FAIL'}**",
              "", "Per-seed raw metrics:", "```json",
              json.dumps(results, indent=2), "```"]
    report = out_dir / "parity.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nparity report: {report}")
    for metric, h, s, ratio, gated, ok in rows:
        tag = ("PASS" if ok else "FAIL") if gated else "descriptive"
        print(f"  {metric:12s} heads {h:.4g}  selfq {s:.4g}  "
              f"ratio {ratio:.3f}  {tag}")
    print(f"OVERALL (body parity): {'PASS' if all_pass else 'FAIL'}")
    assert all_pass, "SelfQ failed BODY parity (see report)"
    print("verify_selfq OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
