"""STRUCTURAL GUARANTEES for the proposal layer (written before the code,
per spec): proposals are data, never code.

1. No execution machinery anywhere under proposals/ or review/: subprocess,
   exec, eval, importlib, os.system are absent (AST scan, not grep — string
   mentions in docstrings are fine, calls/imports are not).
2. Import graph: neither package imports world (levers or otherwise),
   training (mutation/checkpoint-writing code lives there), agent, or torch.
   They read logs and emit JSON — numpy/stdlib/tensorboard-reader only.
3. Filesystem: the packages' write API is confined to runs/<id>/proposals/
   (and the review CLI's tickets to experiments/tickets/); writing anywhere
   else raises.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

BANNED_IMPORTS = {"subprocess", "importlib", "world", "training", "agent",
                  "memory", "torch"}
BANNED_CALLS = {"exec", "eval", "compile", "__import__"}


def _package_files() -> list[Path]:
    files = []
    for pkg in ("proposals", "review"):
        pkg_dir = ROOT / pkg
        assert pkg_dir.exists(), f"{pkg}/ package missing"
        files.extend(sorted(pkg_dir.glob("*.py")))
    assert files, "no python files found under proposals/ or review/"
    return files


def test_no_execution_calls_or_banned_imports() -> None:
    for path in _package_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in BANNED_IMPORTS, f"{path}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in BANNED_IMPORTS, f"{path}: from {node.module} import ..."
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    assert fn.id not in BANNED_CALLS, f"{path}: call to {fn.id}()"
                elif isinstance(fn, ast.Attribute):
                    # os.system, os.exec*, os.spawn*, os.popen
                    if isinstance(fn.value, ast.Name) and fn.value.id == "os":
                        assert not (
                            fn.attr == "system" or fn.attr.startswith(("exec", "spawn", "popen"))
                        ), f"{path}: call to os.{fn.attr}()"


def test_ledger_competence_is_the_only_project_import() -> None:
    """The one project package the proposal layer may read is
    ledger.competence (a numpy-only, read-only report module) and its own
    sibling modules."""
    allowed_roots = {"proposals", "review"}
    for path in _package_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root == "ledger":
                    assert name.startswith("ledger.competence"), (
                        f"{path}: ledger import beyond competence: {name}"
                    )
                else:
                    assert root in allowed_roots or root not in {
                        "viz", "experiments", "scripts", "tests"
                    }, f"{path}: unexpected project import {name}"


def test_writes_confined_to_run_proposals_dir(tmp_path: Path) -> None:
    from proposals.schema import Proposal, proposals_dir, save_proposal

    run_dir = tmp_path / "runs" / "20990101-000000"
    run_dir.mkdir(parents=True)
    p = Proposal.new(
        type="retraining", created_tick=100, run_id=run_dir.name,
        source="ledger", rationale="fixture",
        expected_benefit={"metric": "reward/rollout", "direction": "up",
                          "magnitude_estimate": 0.1},
        confidence=0.5, supporting_observations=["obs:1"],
        estimated_cost={"human_hours": 0.5, "gpu_hours": 1.0},
        risks=["none"],
        success_criteria={"metric": "reward/rollout", "threshold": 0.1,
                          "eval_window_ticks": 1000},
    )
    out = save_proposal(p, run_dir)
    assert out.parent == proposals_dir(run_dir)
    assert out.parent.name == "proposals" and out.parent.parent == run_dir

    # Path traversal / absolute redirection must raise, not write.
    evil = Proposal.new(
        type="retraining", created_tick=1, run_id="../../evil", source="ledger",
        rationale="r", expected_benefit={"metric": "m", "direction": "up",
                                         "magnitude_estimate": 0.0},
        confidence=0.1, supporting_observations=[],
        estimated_cost={"human_hours": 0.0, "gpu_hours": 0.0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0.0,
                          "eval_window_ticks": 1},
    )
    evil.id = "../escape"
    with pytest.raises(ValueError):
        save_proposal(evil, run_dir)
