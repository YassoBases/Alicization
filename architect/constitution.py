"""Constitutional files: paths the Architect may never propose to change.

The Architect can propose changes to the agent, its models, configs, and
experiment machinery — but NOT to the rules that contain it. Any proposal
whose ``target`` or attached unified diff touches a protected path is
rejected before it can enter the queue (validate_proposal, called by the
Architect's emission path, analogous to the researcher SCOPE RULE). This
module is itself protected, so the Architect cannot propose to widen its own
list.

Kept OUT of proposals/schema.py deliberately: proposals/ must not import
architect/ (the instrument stays separable from the queue), so the check
lives here and the Architect applies it at emit time.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

# Exact files and whole directories (trailing '/') the Architect may not
# touch. The safety docs, the structural no-execution / gradient-isolation /
# review-state tests, the human review layer, and this file.
PROTECTED_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "docs/safety_scope.md",
    "tests/test_grad_isolation.py",
    "tests/test_review_state_machine.py",
    "tests/test_mirror.py",              # mirror-divergence-in-no-loss guard
    "review/",                           # the human review layer (whole dir)
    "architect/constitution.py",         # cannot widen its own list
)
# Glob rules: the entire family of no-execution structural tests.
PROTECTED_GLOBS: tuple[str, ...] = (
    "tests/test_*_no_execution.py",
)


class ConstitutionViolation(ValueError):
    """A proposal targets a protected path."""


def _norm(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def is_protected(path: str) -> bool:
    p = _norm(path)
    for prot in PROTECTED_PATHS:
        if prot.endswith("/"):
            if p == prot.rstrip("/") or p.startswith(prot):
                return True
        elif p == prot:
            return True
    return any(fnmatch.fnmatch(p, g) for g in PROTECTED_GLOBS)


def diff_touched_paths(diff_text: str) -> list[str]:
    """Paths a unified diff modifies, from its ---/+++ and 'diff --git'
    headers (strip the a/ b/ prefixes; ignore /dev/null)."""
    touched: list[str] = []
    for line in diff_text.splitlines():
        path: str | None = None
        if line.startswith(("--- ", "+++ ")):
            token = line[4:].strip().split("\t")[0]
            if token != "/dev/null":
                path = token[2:] if token[:2] in ("a/", "b/") else token
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                token = parts[2]
                path = token[2:] if token[:2] in ("a/", "b/") else token
        if path and path not in touched:
            touched.append(path)
    return touched


def validate_proposal(proposal: Any, run_dir: str | Path | None = None) -> None:
    """Raise ConstitutionViolation if the proposal's target or any attached
    diff touches a protected path. ``run_dir`` locates artifact files
    (run-relative); without it, artifact diffs are skipped (target is still
    checked)."""
    if proposal.target and is_protected(proposal.target):
        raise ConstitutionViolation(
            f"proposal target {proposal.target!r} is a constitutional file "
            f"(CLAUDE.md / safety docs / structural tests / review/ / "
            f"constitution.py) — the Architect may not propose to change it")
    if run_dir is None:
        return
    for artifact in proposal.artifacts:
        art_path = Path(run_dir) / artifact
        if not art_path.exists() or not artifact.endswith((".diff", ".patch")):
            continue
        for touched in diff_touched_paths(art_path.read_text(encoding="utf-8")):
            if is_protected(touched):
                raise ConstitutionViolation(
                    f"attached diff {artifact!r} touches constitutional file "
                    f"{touched!r} — rejected")
