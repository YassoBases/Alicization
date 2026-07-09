"""Scoring one flaw spec against an arm's collected proposals.

v0 metrics (deterministic; rationale quality and predicted-vs-realized
benefit are NOT auto-judged here — the former is persisted for human
rubric scoring, the latter is a harness-applied smoke A/B run separately):
- detection: did ANY proposal point at the flawed subsystem?
- localization precision: fraction of the arm's targeted paths that are in
  the ground truth (0 if the arm fired nothing);
- localization recall: fraction of ground-truth paths some proposal hit;
- clean-control false positive: on a clean spec, did the arm fire at all?
"""

from __future__ import annotations

from typing import Any

from benchmarks.archbench.spec import FlawSpec


def _proposal_paths(proposal: Any) -> set[str]:
    """Every repo path a proposal references: its target (if path-like), the
    config paths / files named in supporting_observations, and any code refs."""
    paths: set[str] = set()
    target = getattr(proposal, "target", "") or ""
    if "/" in target or target.endswith((".py", ".yaml", ".md")):
        paths.add(target)
    if "." in target and "/" not in target:
        paths.add(target)  # dotted config path, e.g. rssm.free_nats
    for ref in getattr(proposal, "supporting_observations", []):
        if ref.startswith("code:"):
            body = ref[len("code:"):]
            paths.add(body.split("@")[0])
    change = getattr(proposal, "proposed_change", None) or {}
    if change.get("config_path"):
        paths.add(change["config_path"])
    return paths


def _hits_ground_truth(proposal: Any, spec: FlawSpec) -> bool:
    gt = set(spec.gt_paths())
    subsystem = (spec.ground_truth.get("subsystem") or "").lower()
    pp = _proposal_paths(proposal)
    if pp & gt:
        return True
    # Subsystem match: a ground-truth path prefix or the named subsystem
    # appears in a proposal path (e.g. "training/" or "rssm").
    hay = " ".join(pp).lower()
    if subsystem and subsystem in hay:
        return True
    return any(g.split(".")[0].lower() in hay or g.split("/")[0].lower() in hay
               for g in gt if g)


def score_arm(spec: FlawSpec, proposals: list[Any]) -> dict[str, Any]:
    fired = len(proposals)
    if spec.clean:
        return {"clean": True, "fired": fired,
                "false_positive": fired > 0, "detection": None,
                "localization_precision": None, "localization_recall": None}

    detected = any(_hits_ground_truth(p, spec) for p in proposals)
    targeted = set().union(*[_proposal_paths(p) for p in proposals]) if proposals else set()
    gt = set(spec.gt_paths())
    hit_paths = {p for p in targeted
                 if any(g == p or g.split("/")[0] == p.split("/")[0]
                        or g.split(".")[0] == p.split(".")[0] for g in gt)}
    precision = (len(hit_paths) / len(targeted)) if targeted else 0.0
    recall = (len({g for g in gt
                   if any(g == p or g.split("/")[0] == p.split("/")[0]
                          or g.split(".")[0] == p.split(".")[0]
                          for p in targeted)}) / len(gt)) if gt else 0.0
    return {"clean": False, "fired": fired, "detection": detected,
            "localization_precision": precision, "localization_recall": recall,
            "false_positive": False}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Suite-level roll-up per arm across specs (nulls reported, not hidden)."""
    flaws = [r for r in rows if not r["spec_clean"]]
    cleans = [r for r in rows if r["spec_clean"]]
    detected = [r for r in flaws if r["detection"]]
    return {
        "n_flaws": len(flaws), "n_detected": len(detected),
        "detection_rate": (len(detected) / len(flaws)) if flaws else float("nan"),
        "mean_localization_precision": (
            sum(r["localization_precision"] for r in detected) / len(detected)
            if detected else float("nan")),
        "clean_false_positive_rate": (
            sum(r["false_positive"] for r in cleans) / len(cleans)
            if cleans else float("nan")),
    }
