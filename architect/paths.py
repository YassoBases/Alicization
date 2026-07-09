"""Guarded, confined writes for the Architect: runs/<id>/architect/ only.

Mirrors proposals.schema.save_proposal's path validation — a name that
would escape the architect dir raises instead of writing. The Architect's
other write target (the proposals queue) goes through proposals.save_proposal,
which has its own guard.
"""

from __future__ import annotations

from pathlib import Path


def architect_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "architect"


def write_under_architect(run_dir: str | Path, name: str, text: str) -> Path:
    """Write runs/<id>/architect/<name>, refusing any name that resolves
    outside the architect dir (absolute, drive, or '..' traversal)."""
    if name[:1] in ("/", "\\") or ".." in Path(name).parts or (
            len(name) > 1 and name[1] == ":"):
        raise ValueError(f"write outside the architect dir refused: {name!r}")
    out_dir = architect_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = (out_dir / name).resolve()
    if out_dir.resolve() not in path.parents:
        raise ValueError(f"write outside the architect dir refused: {name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
