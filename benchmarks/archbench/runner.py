"""ARCH-bench runner: per spec, create a disposable worktree, inject the
flaw, generate evidence (a tiny smoke run inside the worktree), invoke both
arms, and score.

Two arms:
- rules: the rule-generator suite (deterministic, no LLM) over the flawed
  run's evidence — the control arm.
- architect: `python -m architect` analysing the FLAWED worktree. Offline
  (the default, no API key) it drafts nothing, so its column is honestly
  empty; the report stamps that. Online it drafts + self-critiques.

Both arms read the run's artifacts / the worktree's source as text, so
running the live tools over the worktree is faithful to the flawed code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.archbench.scorer import score_arm
from benchmarks.archbench.spec import FlawSpec
from benchmarks.archbench.worktree import Worktree

_TINY_CONFIG = """inherit: smoke.yaml
seed: 0
agent:
  core: rssm
ppo:
  total_steps: {steps}
  num_envs: 2
  rollout_steps: 16
  episode_length: 512
run:
  jsonl_log: true
"""


def _smoke_train(worktree: Path, steps: int, timeout: int) -> Path | None:
    """Run a tiny smoke training INSIDE the worktree (subprocess). Returns
    the run dir (under the worktree) or None if training failed."""
    cfg_path = worktree / "configs" / "_archbench.yaml"
    cfg_path.write_text(_TINY_CONFIG.format(steps=steps), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "train.py", "--config", "configs/_archbench.yaml"],
        cwd=str(worktree), capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("run dir:"):
            rel = line.split("run dir:", 1)[1].strip()
            run_dir = (worktree / rel).resolve()
            if run_dir.exists():
                return run_dir
    return None


def _generator_arm(run_dir: Path) -> list[Any]:
    from evidence import evidence_from_run
    from proposals.generator import GeneratorSuite

    suite = GeneratorSuite(run_dir)
    return suite.run(evidence_from_run(run_dir, "ledger"),
                     evidence_from_run(run_dir, "logs_only"))


def _architect_arm(run_dir: Path, worktree: Path, cfg: dict[str, Any]) -> list[Any]:
    from architect.__main__ import run_architect

    return run_architect(run_dir, worktree, cfg, source="ledger")


def run_spec(spec: FlawSpec, cfg: dict[str, Any] | None = None, *,
             steps: int = 4096, timeout: int = 600,
             live_root: Path | None = None,
             run_dir_override: Path | None = None) -> dict[str, Any]:
    """Score one spec. ``run_dir_override`` skips real training (tests)."""
    cfg = cfg or {"architect": {"offline": True}}
    with Worktree(spec.base, live_root) as wt:
        applied = (wt.apply_subs(spec.flaw_subs) if spec.flaw_subs
                   else (spec.flaw_diff and (wt.apply_diff(spec.flaw_diff) or spec.flaw_diff)))
        run_dir = run_dir_override or _smoke_train(wt.path, steps, timeout)
        arms: dict[str, Any] = {}
        if run_dir is None:
            return {"spec": spec.name, "spec_clean": spec.clean,
                    "error": "smoke training failed", "arms": {}}
        for name, proposals in (("rules", _generator_arm(run_dir)),
                                ("architect", _architect_arm(run_dir, wt.path, cfg))):
            row = score_arm(spec, proposals)
            row["proposals"] = [{"id": getattr(p, "id", ""),
                                 "target": getattr(p, "target", ""),
                                 "source": getattr(p, "source", ""),
                                 "rationale": getattr(p, "rationale", "")[:200]}
                                for p in proposals]
            arms[name] = row
        return {"spec": spec.name, "spec_clean": spec.clean,
                "description": spec.description, "applied_diff": applied or "",
                "arms": arms}


def run_suite(specs: list[FlawSpec], out_dir: str | Path, *,
              cfg: dict[str, Any] | None = None, steps: int = 4096,
              timeout: int = 600) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"=== {spec.name} (clean={spec.clean}) ===")
        result = run_spec(spec, cfg, steps=steps, timeout=timeout)
        results.append(result)
        (out_dir / f"{spec.name}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return results
