"""Enforcement for the simulated-continuity metric (CLAUDE.md Hard rules):
a DEPENDENT variable — decoupled from process/OS state, present in no loss,
reward, or policy-input construction. Plus behavioral tests on synthetic
run data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.continuity import (
    COMPONENTS,
    compare_runs,
    compute_run,
)

ROOT = Path(__file__).resolve().parent.parent

# Modules where any coupling to the continuity metric would put it inside a
# loss, reward, or policy-input path.
AGENT_SIDE_DIRS = ("training", "agent", "ledger", "memory", "world")

# Imports that would couple the metric to process lifetime / system state.
BANNED_IN_CONTINUITY = {"os", "psutil", "subprocess", "time", "datetime",
                        "signal", "socket", "sys", "platform",
                        "world", "training", "agent", "memory", "torch"}


def test_continuity_module_has_no_process_or_training_imports() -> None:
    tree = ast.parse((ROOT / "experiments" / "continuity.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            assert root not in BANNED_IN_CONTINUITY, (
                f"continuity.py imports {name} — the metric must read run "
                f"logs only (no process/system/world/training access)"
            )


def test_continuity_symbols_appear_in_no_agent_side_code() -> None:
    """No loss, reward, or policy-input construction may reference the
    metric: nothing under training/, agent/, ledger/, memory/, or world/
    imports experiments.continuity or names its symbols."""
    for pkg in AGENT_SIDE_DIRS:
        for path in (ROOT / pkg).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not any("continuity" in n for n in names), (
                    f"{path} imports the continuity metric — it is a "
                    f"dependent variable, not a training signal"
                )
                if isinstance(node, ast.Name):
                    assert "continuity" not in node.id.lower(), (
                        f"{path}: symbol {node.id} references continuity"
                    )


# ------------------------------------------------------- behavioral checks


def _write_fixture_run(run_dir: Path, n_ticks: int, action_period: int,
                       reward_dip: bool) -> None:
    """Synthetic JSONL: a walker with controllable action periodicity and an
    optional mid-run reward dip."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    x = y = 16
    with open(run_dir / "events-000000000.jsonl", "w", encoding="utf-8") as f:
        for t in range(n_ticks):
            action = int((t // action_period) % 4)
            dx, dy = [(0, -1), (0, 1), (1, 0), (-1, 0)][action]
            x = int(np.clip(x + dx, 0, 31))
            y = int(np.clip(y + dy, 0, 31))
            reward = 1.0
            if reward_dip and n_ticks // 2 < t < n_ticks // 2 + 400:
                reward = 0.0
            f.write(json.dumps({
                "tick": t + 1, "pos": [x, y], "action": action,
                "success": True, "reward": reward,
                "intero": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            }) + "\n")
    del rng


def test_components_and_composite_on_synthetic_runs(tmp_path: Path) -> None:
    stable = tmp_path / "stable-run"
    flappy = tmp_path / "flappy-run"
    _write_fixture_run(stable, 8192, action_period=1024, reward_dip=False)
    _write_fixture_run(flappy, 8192, action_period=7, reward_dip=True)

    results = compare_runs([stable, flappy], window_ticks=2048)
    for name, r in results.items():
        assert len(r.composite) == 4, name
        # Components reported individually alongside the composite — never
        # only the composite.
        for comp in COMPONENTS:
            assert comp in r.components and len(r.components[comp]) == 4

    # The stable walker repeats its action distribution window over window ->
    # higher preference persistence than the fast-cycling one is NOT
    # guaranteed (both settle), but the dip run must show a worse (more
    # negative) adaptation half-life in its dip window.
    dip_hl = np.nanmin(results["flappy-run"].components["adaptation_half_life"])
    stable_hl = np.nanmin(results["stable-run"].components["adaptation_half_life"])
    assert dip_hl < stable_hl

    mean, ci = results["stable-run"].mean_ci()
    assert np.isfinite(mean) and np.isfinite(ci)


def test_zscores_are_pooled_across_runs(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_fixture_run(a, 4096, action_period=64, reward_dip=False)
    _write_fixture_run(b, 4096, action_period=64, reward_dip=False)
    results = compare_runs([a, b], window_ticks=2048)
    pooled = [v for r in results.values()
              for v in r.components["z_revisit_efficiency"] if np.isfinite(v)]
    # Pooled z-scores have ~zero mean by construction.
    assert abs(np.mean(pooled)) < 1e-9
