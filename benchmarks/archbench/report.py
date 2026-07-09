"""Render the ARCH-bench report: per-flaw table, both arms, nulls in full,
scale stamped. ANALYSIS is written by a human — this only tabulates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.archbench.scorer import summarize


def _arm_rows(results: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        a = r.get("arms", {}).get(arm)
        if a is None:
            rows.append({"spec": r["spec"], "spec_clean": r["spec_clean"],
                         "fired": 0, "detection": None,
                         "localization_precision": None, "false_positive": None})
            continue
        rows.append({"spec": r["spec"], "spec_clean": r["spec_clean"], **a})
    return rows


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.2f}"
    return str(x)


def render(results: list[dict[str, Any]], out_dir: str | Path,
           scale: str, online: bool) -> Path:
    out_dir = Path(out_dir)
    lines = [
        "# ARCH-bench v0", "",
        f"- scale: **{scale}**, architect arm: "
        f"**{'online' if online else 'offline (drafts nothing — column empty by design)'}**",
        f"- specs: {len(results)} "
        f"({sum(not r['spec_clean'] for r in results)} flaws + "
        f"{sum(r['spec_clean'] for r in results)} clean control)",
        "- rules = the deterministic rule-generator suite (control arm).",
        "- Nulls are shown, not hidden. Rationale quality is NOT auto-judged "
        "in v0 (persisted per proposal for human rubric scoring).", "",
    ]
    for arm in ("rules", "architect"):
        rows = _arm_rows(results, arm)
        s = summarize(rows)
        lines += [
            f"## Arm: {arm}", "",
            f"- detection rate (flaws): {_fmt(s['detection_rate'])} "
            f"({s['n_detected']}/{s['n_flaws']})",
            f"- mean localization precision (detected): "
            f"{_fmt(s['mean_localization_precision'])}",
            f"- clean-control false-positive rate: "
            f"{_fmt(s['clean_false_positive_rate'])}", "",
            "| spec | clean | fired | detected | localization prec | false positive |",
            "|------|-------|-------|----------|-------------------|----------------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['spec']} | {_fmt(r['spec_clean'])} | {r['fired']} | "
                f"{_fmt(r.get('detection'))} | "
                f"{_fmt(r.get('localization_precision'))} | "
                f"{_fmt(r.get('false_positive'))} |")
        lines.append("")
    path = out_dir / "archbench.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
