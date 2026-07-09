"""The Architect (stage-D): an EXPERIMENTER-SIDE instrument that reads the
repository and a run's evidence and DRAFTS proposals for a human to review.

It is NOT the agent and NOT the in-run researcher. Run code and analysis
code never import it (tests/test_architect_no_execution.py enforces the
separation both ways). Everything it produces is a proposal in the standard
queue, reviewed and executed by a human exactly like every other proposal
(docs/safety_scope.md). The Architect never applies a change; the ARCH-bench
harness applies patches only inside disposable git worktrees under human
invocation.

Structural rules (same posture as proposals/researcher/evidence, plus the
constitutional-files rule):
- Data, never code: no subprocess/exec/eval/compile/importlib/os.system.
  The single allowed side effect is the LLM network call in draft.py.
- Reads the modules under analysis as TEXT (pathlib/open + ast on the
  source string); never imports world/training/agent/memory/torch.
- Writes confined to runs/<id>/architect/ (paths.write_under_architect) and
  the proposals dir (proposals.save_proposal).
- CONSTITUTION: proposals touching protected paths (CLAUDE.md, the safety
  docs, the no-execution/grad-isolation/review tests, review/, and
  constitution.py itself) are rejected before emission (constitution.py).
"""

from __future__ import annotations
