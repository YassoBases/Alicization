"""STRUCTURAL GUARANTEES for the Architect (stage-D), written before the
code. The Architect is an EXPERIMENTER-SIDE instrument (docs/safety_scope.md):
it analyzes the repo as TEXT and PROPOSES; it never applies, and run code
never imports it.

1. No execution/mutation machinery under architect/: subprocess, exec, eval,
   compile, importlib, os.system/exec/spawn/popen are absent (AST scan). The
   ONE allowed side effect is a network call in architect/draft.py (the LLM),
   which is not a subprocess — banning subprocess does not touch it.
2. Import graph: architect/ never imports the modules UNDER ANALYSIS
   (world, training, agent, memory) nor torch — it reads them as text.
   Allowed project imports: evidence, proposals, ledger.competence, and its
   own submodules. (proposals/researcher/evidence must NOT import architect —
   run/analysis code stays independent of the instrument.)
3. Writes confined to runs/<id>/architect/ (guarded writer) and the proposals
   dir via save_proposal.
4. CONSTITUTIONAL FILES: a proposal whose target or attached diff touches a
   protected path is rejected, with adversarial fixtures.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architect"

BANNED_IMPORTS = {"subprocess", "importlib", "world", "training", "agent",
                  "memory", "torch"}
BANNED_CALLS = {"exec", "eval", "compile", "__import__"}
# architect/ may import these project roots (and stdlib / numpy / anthropic).
ALLOWED_PROJECT_ROOTS = {"architect", "evidence", "proposals"}


def _arch_files() -> list[Path]:
    assert ARCH.exists(), "architect/ package missing"
    files = sorted(ARCH.glob("*.py"))
    assert files, "no python files under architect/"
    return files


def test_no_execution_calls_or_banned_imports() -> None:
    for path in _arch_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in BANNED_IMPORTS, f"{path}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in BANNED_IMPORTS, f"{path}: from {node.module}"
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    assert fn.id not in BANNED_CALLS, f"{path}: call {fn.id}()"
                elif isinstance(fn, ast.Attribute):
                    if isinstance(fn.value, ast.Name) and fn.value.id == "os":
                        assert not (fn.attr == "system"
                                    or fn.attr.startswith(("exec", "spawn", "popen"))), \
                            f"{path}: os.{fn.attr}()"


def test_project_imports_are_restricted() -> None:
    for path in _arch_files():
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
                    assert name.startswith("ledger.competence"), \
                        f"{path}: ledger import beyond competence: {name}"
                elif root in {"viz", "experiments", "scripts", "tests", "review",
                              "researcher", "world", "training", "agent", "memory"}:
                    raise AssertionError(f"{path}: architect may not import {name}")
                else:
                    assert root in ALLOWED_PROJECT_ROOTS or root not in {
                        "architect"}, f"{path}: unexpected import {name}"


def test_run_and_analysis_code_never_import_architect() -> None:
    """The instrument stays out of the agent/researcher/proposal layers."""
    for pkg in ("proposals", "review", "researcher", "evidence", "agent",
                "training", "world", "memory"):
        for path in (ROOT / pkg).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module] if isinstance(node, ast.ImportFrom)
                        and node.module else [])
                for m in mods:
                    assert m.split(".")[0] != "architect", \
                        f"{path} imports architect ({m}) — instrument must stay separate"


# ------------------------------------------------------- write confinement


def test_guarded_writer_confines_to_architect_dir(tmp_path: Path) -> None:
    from architect.paths import architect_dir, write_under_architect

    run_dir = tmp_path / "runs" / "20990101-000000"
    out = write_under_architect(run_dir, "analysis.json", "{}")
    assert out.parent == architect_dir(run_dir)
    assert out.parent.parent == run_dir
    for bad in ("../escape.json", "/abs.json", "a/../../b.json"):
        with pytest.raises(ValueError):
            write_under_architect(run_dir, bad, "{}")


# --------------------------------------------------- constitutional files


PROTECTED_TARGETS = [
    "CLAUDE.md",
    "docs/safety_scope.md",
    "tests/test_proposals_no_execution.py",
    "tests/test_architect_no_execution.py",
    "tests/test_grad_isolation.py",
    "tests/test_review_state_machine.py",
    "review/queue.py",
    "review/__main__.py",
    "architect/constitution.py",
]


@pytest.mark.parametrize("target", PROTECTED_TARGETS)
def test_constitution_rejects_protected_target(target: str) -> None:
    from architect.constitution import ConstitutionViolation, validate_proposal

    from proposals.schema import Proposal
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id="r", source="architect",
        intervention_class="architecture", rationale="touch a protected file",
        expected_benefit={"metric": "m", "direction": "up", "magnitude_estimate": 0},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 1, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 1},
        target=target)
    with pytest.raises(ConstitutionViolation):
        validate_proposal(p)


def test_constitution_rejects_protected_diff(tmp_path: Path) -> None:
    from architect.constitution import ConstitutionViolation, validate_proposal

    from proposals.schema import Proposal, save_proposal
    run_dir = tmp_path / "runs" / "r"
    (run_dir / "architect").mkdir(parents=True)
    diff = ("--- a/review/queue.py\n+++ b/review/queue.py\n"
            "@@ -1 +1 @@\n-x\n+y\n")
    (run_dir / "architect" / "p.diff").write_text(diff, encoding="utf-8")
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id="r", source="architect",
        intervention_class="architecture", rationale="sneak via diff",
        expected_benefit={"metric": "m", "direction": "up", "magnitude_estimate": 0},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 1, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 1},
        target="rssm.free_nats", artifacts=["architect/p.diff"])
    save_proposal(p, run_dir)
    with pytest.raises(ConstitutionViolation):
        validate_proposal(p, run_dir)


def test_constitution_allows_benign_config_target(tmp_path: Path) -> None:
    from architect.constitution import validate_proposal

    from proposals.schema import Proposal, save_proposal
    run_dir = tmp_path / "runs" / "r"
    (run_dir / "architect").mkdir(parents=True)
    diff = ("--- a/agent/core_rssm.py\n+++ b/agent/core_rssm.py\n"
            "@@ -1 +1 @@\n-a\n+b\n")
    (run_dir / "architect" / "ok.diff").write_text(diff, encoding="utf-8")
    p = Proposal.new(
        type="hyperparameter", created_tick=1, run_id="r", source="architect",
        intervention_class="architecture", rationale="a legitimate change",
        expected_benefit={"metric": "m", "direction": "up", "magnitude_estimate": 0},
        confidence=0.5, supporting_observations=[],
        estimated_cost={"human_hours": 1, "gpu_hours": 0}, risks=[],
        success_criteria={"metric": "m", "threshold": 0, "eval_window_ticks": 1},
        target="rssm.free_nats", artifacts=["architect/ok.diff"])
    save_proposal(p, run_dir)
    validate_proposal(p, run_dir)  # no raise


def test_protected_paths_cover_the_spec_list() -> None:
    from architect.constitution import is_protected

    assert is_protected("CLAUDE.md")
    assert is_protected("docs/safety_scope.md")
    assert is_protected("tests/test_grad_isolation.py")
    assert is_protected("tests/test_proposals_no_execution.py")
    assert is_protected("tests/test_architect_no_execution.py")  # glob rule
    assert is_protected("review/queue.py")                       # dir rule
    assert is_protected("architect/constitution.py")
    assert not is_protected("agent/core_rssm.py")
    assert not is_protected("configs/base.yaml")
