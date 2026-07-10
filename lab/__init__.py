"""The Architecture Laboratory (lab/): experimenter-side machinery for
treating every subsystem as a falsifiable scientific hypothesis.

Phase 1 adds the Assumption Registry (lab/assumptions.yaml) and its renderer
(lab/render.py -> docs/assumptions.md). Later stages add the comparison-
harness glue. This is analysis/experimenter tooling, not agent-side code; it
never enters an observation, a loss, or a run.
"""

from __future__ import annotations
