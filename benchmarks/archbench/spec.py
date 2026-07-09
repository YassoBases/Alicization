"""Flaw specs: declarative descriptions of an injected repository flaw.

Each YAML under flaws/ carries: the base ref it patches, the flaw diff, the
symptomatic scalar pattern it should produce, the ground-truth localization
(files / config paths / subsystem) an analysis SHOULD point at, and
optionally the ground-truth fix diff. ``clean: true`` marks the control
(no flaw — false positives are scored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FlawSpec:
    name: str
    description: str
    base: str                              # git ref to branch the worktree from
    ground_truth: dict[str, Any]           # {files, config_paths, subsystem}
    # A flaw is expressed EITHER as a unified diff (flaw_diff) OR as
    # find/replace ops (flaw_subs: [{file, find, replace}]). v0 prefers subs
    # for robustness (line-numbered diffs break under edits); the runner
    # synthesizes a unified diff from the applied subs so the recorded
    # artifact is still a diff. fix_* is the ground-truth repair (optional).
    flaw_diff: str = ""
    flaw_subs: list[dict[str, str]] = field(default_factory=list)
    fix_diff: str = ""
    fix_subs: list[dict[str, str]] = field(default_factory=list)
    symptom: dict[str, Any] = field(default_factory=dict)   # {tag, pattern}
    clean: bool = False                    # the no-flaw control

    def gt_paths(self) -> list[str]:
        return list(self.ground_truth.get("files", [])) + \
            list(self.ground_truth.get("config_paths", []))


def load_spec(path: str | Path) -> FlawSpec:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return FlawSpec(
        name=raw["name"], description=raw.get("description", ""),
        base=raw.get("base", "HEAD"), ground_truth=raw.get("ground_truth", {}),
        flaw_diff=raw.get("flaw_diff", ""),
        flaw_subs=list(raw.get("flaw_subs", [])),
        fix_diff=raw.get("fix_diff", ""),
        fix_subs=list(raw.get("fix_subs", [])),
        symptom=raw.get("symptom", {}), clean=bool(raw.get("clean", False)))


def load_specs(directory: str | Path) -> list[FlawSpec]:
    directory = Path(directory)
    return [load_spec(p) for p in sorted(directory.glob("*.yaml"))]


def default_specs_dir() -> Path:
    return Path(__file__).resolve().parent / "flaws"
