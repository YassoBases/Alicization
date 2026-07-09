"""Deterministic, LLM-free analysis of the repository (stage-D2).

Three passes, all pure text/AST (the Architect reads the modules under
analysis, never imports them):

  (a) module map      — every source file's line count and its project
                        imports, plus the package-level import graph.
  (b) invariants      — CLAUDE.md hard-rule bullets and the structural
                        tests' banned-import sets, parsed into a table.
  (c) symptom linkage — for each anomalous scalar tag (the generator
                        trigger vocabulary by default, or a bundle's tags),
                        the source modules that EMIT it (grep the literal).

Output: one JSON-serializable AnalysisReport. Unit-tested on this repo,
asserting stable structural facts rather than a brittle full-file golden
(line counts drift as code changes).
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Source packages the Architect maps. It reads them as text; the import-ban
# test guarantees it never imports the agent/training/world/memory ones.
PROJECT_PACKAGES: tuple[str, ...] = (
    "agent", "training", "world", "ledger", "memory", "proposals", "review",
    "researcher", "evidence", "experiments", "viz", "architect",
)

# The generator trigger vocabulary (proposals/generator.py): the scalar
# tags whose anomalies drive recommendations. Symptom linkage joins these
# to the modules that log them.
GENERATOR_TRIGGER_TAGS: tuple[str, ...] = (
    "reward/rollout", "rssm/kl", "sleep/kl", "rssm/recon",
    "ledger/reliability_ece", "memory/stale_trip_rate_per_1k", "loss/total",
    "sleep/grad_steps", "rssm/participation_ratio", "clip_frac",
)

# Structural test files whose BANNED_IMPORTS sets define the import bans.
_BAN_TEST_FILES = {
    "proposals/review/researcher/evidence": "tests/test_proposals_no_execution.py",
    "architect": "tests/test_architect_no_execution.py",
}


@dataclass
class ModuleInfo:
    path: str                     # repo-relative
    package: str
    loc: int
    project_imports: list[str]    # project packages this module imports


@dataclass
class Invariants:
    hard_rules: list[str]                 # CLAUDE.md bullets
    banned_imports: dict[str, list[str]]  # scope -> banned import roots


@dataclass
class AnalysisReport:
    repo_root: str
    modules: list[ModuleInfo]
    import_graph: dict[str, list[str]]    # package -> imported packages
    invariants: Invariants
    tag_emitters: dict[str, list[str]]    # scalar tag -> emitting modules
    evidence_bundle_hash: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ------------------------------------------------------------- module map


def _module_project_imports(tree: ast.AST) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in PROJECT_PACKAGES:
                found.add(root)
    return sorted(found)


def build_module_map(repo_root: Path) -> tuple[list[ModuleInfo], dict[str, list[str]]]:
    modules: list[ModuleInfo] = []
    graph: dict[str, set[str]] = {}
    for pkg in PROJECT_PACKAGES:
        pkg_dir = repo_root / pkg
        if not pkg_dir.exists():
            continue
        graph.setdefault(pkg, set())
        for path in sorted(pkg_dir.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            imports = _module_project_imports(tree)
            modules.append(ModuleInfo(
                path=path.relative_to(repo_root).as_posix(), package=pkg,
                loc=text.count("\n") + 1, project_imports=imports))
            graph[pkg].update(i for i in imports if i != pkg)
    return modules, {k: sorted(v) for k, v in sorted(graph.items())}


# ------------------------------------------------------------- invariants


def parse_hard_rules(claude_md: Path) -> list[str]:
    """The '- ' bullets under CLAUDE.md's '## Hard rules' section, each
    flattened to a single line."""
    if not claude_md.exists():
        return []
    lines = claude_md.read_text(encoding="utf-8").splitlines()
    rules: list[str] = []
    in_section = False
    current: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current:
                rules.append(" ".join(current).strip())
                current = []
            in_section = line.strip().lower() == "## hard rules"
            continue
        if not in_section:
            continue
        if line.startswith("- "):
            if current:
                rules.append(" ".join(current).strip())
            current = [line[2:].strip()]
        elif line.strip() and current:
            current.append(line.strip())
    if current:
        rules.append(" ".join(current).strip())
    return rules


def _banned_imports_from_test(path: Path) -> list[str]:
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "BANNED_IMPORTS" in targets and isinstance(node.value, ast.Set):
                return sorted(e.value for e in node.value.elts
                              if isinstance(e, ast.Constant)
                              and isinstance(e.value, str))
    return []


def extract_invariants(repo_root: Path) -> Invariants:
    banned = {scope: _banned_imports_from_test(repo_root / rel)
              for scope, rel in _BAN_TEST_FILES.items()}
    return Invariants(hard_rules=parse_hard_rules(repo_root / "CLAUDE.md"),
                      banned_imports={k: v for k, v in banned.items() if v})


# ------------------------------------------------------------ symptom link


def link_symptoms(repo_root: Path, tags: tuple[str, ...]) -> dict[str, list[str]]:
    """For each tag, the source modules that contain its string literal —
    i.e. that log it. Pure grep over the project packages."""
    # Precompute each source file's text once.
    sources: list[tuple[str, str]] = []
    for pkg in PROJECT_PACKAGES:
        for path in sorted((repo_root / pkg).rglob("*.py")) if (repo_root / pkg).exists() else []:
            sources.append((path.relative_to(repo_root).as_posix(),
                            path.read_text(encoding="utf-8")))
    out: dict[str, list[str]] = {}
    for tag in tags:
        needle = f'"{tag}"'
        alt = f"'{tag}'"
        emitters = [rel for rel, text in sources
                    if needle in text or alt in text]
        if emitters:
            out[tag] = emitters
    return out


# ---------------------------------------------------------------- analyze


def analyze(repo_root: str | Path,
            anomalous_tags: tuple[str, ...] | None = None,
            evidence_bundle_hash: str | None = None) -> AnalysisReport:
    """Run all three passes. ``anomalous_tags`` scopes symptom linkage (the
    generator trigger vocabulary by default)."""
    repo_root = Path(repo_root)
    modules, graph = build_module_map(repo_root)
    invariants = extract_invariants(repo_root)
    tags = anomalous_tags if anomalous_tags is not None else GENERATOR_TRIGGER_TAGS
    tag_emitters = link_symptoms(repo_root, tuple(tags))
    return AnalysisReport(
        repo_root=str(repo_root), modules=modules, import_graph=graph,
        invariants=invariants, tag_emitters=tag_emitters,
        evidence_bundle_hash=evidence_bundle_hash)
