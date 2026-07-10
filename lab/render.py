"""Render the Assumption Registry (lab/assumptions.yaml) to a human-readable
docs/assumptions.md, and expose the citation/loader helpers the registry
test reuses.

The registry is the machine-readable source of truth; docs/assumptions.md is
its rendering (regenerate with `python -m lab.render`). Every citation in
the registry must resolve to a real repo file (optionally a line anchor) —
enforced by tests/test_assumptions_registry.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "lab" / "assumptions.yaml"
OUT = ROOT / "docs" / "assumptions.md"

REQUIRED_FIELDS = (
    "id", "component", "purpose", "hypothesis", "evidence_for",
    "evidence_against", "confidence", "confidence_note", "success_criteria",
    "failure_criteria", "replacement_candidates", "status",
)
STATUSES = ("supported", "contested", "unsupported", "untested")
MATURITIES = ("established", "adaptation", "underexplored", "speculative")

_CITE_RE = re.compile(r"^(?P<path>[^#]+?)(?:#L(?P<a>\d+)(?:-L(?P<b>\d+))?)?$")


@dataclass(frozen=True)
class CitationError:
    entry_id: str
    field: str
    citation: str
    reason: str


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def iter_citations(registry: dict[str, Any]):
    """Yield (entry_id, field, citation) for every evidence citation."""
    for entry in registry.get("assumptions", []):
        for field in ("evidence_for", "evidence_against"):
            for cite in entry.get(field) or []:
                yield entry["id"], field, str(cite)


def resolve_citation(citation: str, root: Path = ROOT) -> str | None:
    """None if the citation resolves; otherwise a human reason string.

    A citation is 'path' or 'path#Lx' or 'path#Lx-Ly'. The path must exist;
    a line anchor must be within the file's length; a range must be ordered."""
    m = _CITE_RE.match(citation.strip())
    if not m:
        return f"unparseable citation {citation!r}"
    target = root / m.group("path").strip()
    if not target.exists():
        return f"file not found: {m.group('path').strip()}"
    if m.group("a"):
        a = int(m.group("a"))
        b = int(m.group("b")) if m.group("b") else a
        if b < a:
            return f"reversed line range in {citation!r}"
        loc = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        if a < 1 or b > loc:
            return f"line anchor {a}-{b} outside 1..{loc} in {citation!r}"
    return None


def validate(registry: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Return a list of human-readable problems (empty == valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for entry in registry.get("assumptions", []):
        eid = entry.get("id", "<no-id>")
        for f in REQUIRED_FIELDS:
            if f not in entry:
                problems.append(f"{eid}: missing field {f!r}")
        if eid in seen:
            problems.append(f"duplicate id {eid!r}")
        seen.add(eid)
        if entry.get("status") not in STATUSES:
            problems.append(f"{eid}: bad status {entry.get('status')!r}")
        conf = entry.get("confidence")
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            problems.append(f"{eid}: confidence not in [0,1]: {conf!r}")
        for cand in entry.get("replacement_candidates") or []:
            if cand.get("maturity") not in MATURITIES:
                problems.append(f"{eid}: bad maturity {cand.get('maturity')!r}")
    for eid, field, cite in iter_citations(registry):
        reason = resolve_citation(cite, root)
        if reason:
            problems.append(f"{eid}.{field}: {reason}")
    return problems


def render(registry: dict[str, Any]) -> str:
    entries = registry.get("assumptions", [])
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    lines = [
        "# Assumption Registry",
        "",
        "Auto-generated from `lab/assumptions.yaml` by `python -m lab.render` "
        "— do not edit by hand. Every subsystem is a scientific hypothesis, "
        "not a permanent design; this is the empirical status of each.",
        "",
        "| status | count |",
        "|--------|-------|",
    ]
    for st in STATUSES:
        lines.append(f"| {st} | {by_status.get(st, 0)} |")
    lines += ["", "| id | component | status | confidence |",
              "|----|-----------|--------|------------|"]
    for e in entries:
        lines.append(f"| [{e['id']}](#{e['id']}) | `{e['component']}` "
                     f"| **{e['status']}** | {e['confidence']} |")
    lines.append("")
    for e in entries:
        lines += [
            f"## {e['id']}", "",
            f"- **component**: `{e['component']}`",
            f"- **status**: {e['status']}  |  **confidence**: {e['confidence']} "
            f"— {e['confidence_note'].strip()}",
            f"- **purpose**: {e['purpose'].strip()}",
            f"- **hypothesis**: {e['hypothesis'].strip()}",
            f"- **success**: {e['success_criteria'].strip()}",
            f"- **failure (replace/remove when)**: {e['failure_criteria'].strip()}",
        ]
        for label, field in (("evidence for", "evidence_for"),
                             ("evidence against", "evidence_against")):
            cites = e.get(field) or []
            rendered = ", ".join(f"`{c}`" for c in cites) if cites else "_none_"
            lines.append(f"- **{label}**: {rendered}")
        cands = ", ".join(f"{c['name']} ({c['maturity']})"
                          for c in e.get("replacement_candidates") or [])
        lines.append(f"- **replacement candidates**: {cands or '_none_'}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    registry = load_registry()
    problems = validate(registry)
    if problems:
        print("REGISTRY INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1
    OUT.write_text(render(registry), encoding="utf-8")
    print(f"wrote {OUT} ({len(registry['assumptions'])} assumptions)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
