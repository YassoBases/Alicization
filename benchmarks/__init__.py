"""Experimenter benchmarks. Unlike architect/proposals/researcher/evidence,
this tree MAY run subprocesses — but ARCH-bench (benchmarks/archbench) does
so ONLY inside disposable git worktrees it creates under a temp dir, never
the live checkout (benchmarks/archbench/worktree.py; the live-repo guard is
tested).
"""

from __future__ import annotations
