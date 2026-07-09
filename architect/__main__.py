"""`python -m architect --run runs/<id>` (stage-D5).

Analyzes the repo + the run's evidence, drafts and self-critiques proposals,
and emits the survivors into the standard proposal queue — writing
runs/<id>/architect/analysis.json and appending every draft/critique/emit
decision to runs/<id>/architect/decisions.jsonl (mirroring
generator_decisions.jsonl). Offline by default (no network); the whole path
runs and is tested without an API key.

Config comes from the run's ALREADY-RESOLVED config.json (architect may not
import world.config to resolve YAML). Absent or missing an architect
section -> offline defaults on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from architect.analysis import analyze
from architect.constitution import validate_proposal
from architect.critique import critique_proposals
from architect.draft import LLMClient, draft_proposals
from architect.paths import architect_dir, write_under_architect
from evidence import EvidenceStore, RepoSnapshot
from proposals.schema import Proposal, save_proposal


def _append_decisions(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = architect_dir(run_dir) / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({**rec, "timestamp": now}) + "\n")
    return path


def run_architect(run_dir: str | Path, repo_root: str | Path,
                  cfg: dict[str, Any], source: str = "ledger",
                  repo_sha: str | None = None,
                  client: LLMClient | None = None) -> list[Proposal]:
    """The whole pipeline: analyze -> draft -> critique -> emit. Returns the
    emitted proposals."""
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    snapshot = RepoSnapshot(repo_sha, None) if repo_sha else None
    store = EvidenceStore(run_dir, repo_snapshot=snapshot)
    view = store.view(source)

    report = analyze(repo_root, evidence_bundle_hash=view.bundle_hash)
    write_under_architect(run_dir, "analysis.json", report.to_json())

    draft = draft_proposals(report, view, run_dir, cfg, client=client)
    kept, critique_decisions = critique_proposals(
        draft.proposals, view, repo_root, cfg, client=client)

    emit_decisions: list[dict[str, Any]] = []
    emitted: list[Proposal] = []
    for proposal in kept:
        try:
            validate_proposal(proposal, run_dir)   # constitution, at emit time
        except Exception as exc:  # noqa: BLE001 - log and skip, never raise out
            emit_decisions.append({"action": "discard", "proposal_id": proposal.id,
                                   "reason": f"constitution at emit: {exc}"})
            continue
        save_proposal(proposal, run_dir)
        emitted.append(proposal)
        emit_decisions.append({"action": "emit", "proposal_id": proposal.id,
                               "target": proposal.target,
                               "intervention_class": proposal.intervention_class})

    _append_decisions(run_dir, draft.decisions + critique_decisions + emit_decisions)
    return emitted


def _load_cfg(run_dir: Path, offline: bool) -> dict[str, Any]:
    cfg_file = run_dir / "config.json"
    cfg: dict[str, Any] = (json.loads(cfg_file.read_text(encoding="utf-8"))
                           if cfg_file.exists() else {})
    if offline:
        cfg.setdefault("architect", {})["offline"] = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Architect: analyze a run + the repo, draft proposals for "
                    "human review (experimenter-side; never applies changes).")
    parser.add_argument("--run", required=True, help="runs/<id>")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo-sha", default=None)
    parser.add_argument("--source", default="ledger",
                        choices=["ledger", "logs_only"])
    parser.add_argument("--offline", action="store_true",
                        help="force offline (no network); the default anyway")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    cfg = _load_cfg(run_dir, args.offline)
    client: LLMClient | None = None
    if not cfg.get("architect", {}).get("offline", True):
        from architect.draft import build_anthropic_client
        client = build_anthropic_client()

    emitted = run_architect(run_dir, args.repo_root, cfg, source=args.source,
                            repo_sha=args.repo_sha, client=client)
    offline = cfg.get("architect", {}).get("offline", True)
    print(f"architect: {'offline (no drafting)' if offline else 'online'}; "
          f"emitted {len(emitted)} proposal(s) into {run_dir / 'proposals'}")
    for p in emitted:
        print(f"  {p.id} [{p.intervention_class}] {p.target}")
    print(f"analysis + decisions under {architect_dir(run_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
