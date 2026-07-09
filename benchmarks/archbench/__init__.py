"""ARCH-bench v0 (stage-D6): score the Architect (and the rule-generator
suite as the control arm) on a battery of injected repository flaws.

For each flaw spec: create a disposable git worktree, apply the flaw diff,
run a quick smoke training to generate evidence, invoke both arms on the
flawed worktree, and score detection / localization / clean-control false
positives. Predicted-vs-realized benefit (apply the proposed fix, smoke
A/B) is harness-applied under human invocation. Rationale quality is
persisted for human rubric scoring, never auto-judged in v0.

Reports land in benchmarks/results/<date>/archbench.md in the house style:
per-flaw, both arms, nulls with full prominence, scale stamped.
"""

from __future__ import annotations
