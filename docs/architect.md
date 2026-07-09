# The Architect (Stage D)

An EXPERIMENTER-SIDE instrument that reads the repository and a run's
evidence and DRAFTS proposals for a human to review. It is not the agent
and not the in-run researcher; **run code and analysis code never import
it**, and it never applies a change. Everything it produces is a proposal
in the standard queue, reviewed and executed by a human exactly like every
other proposal.

## Pipeline (`python -m architect --run runs/<id> [--offline]`)

1. **analysis** (`architect/analysis.py`, deterministic, LLM-free): a module
   map (line counts + import graph), invariants (CLAUDE.md hard-rule bullets
   and the structural tests' banned-import sets), and symptom linkage
   (anomalous scalar tags → the modules that emit them). Written to
   `runs/<id>/architect/analysis.json`.
2. **draft** (`architect/draft.py`): builds a prompt from the analysis + the
   evidence bundle summary + capped source excerpts and calls the LLM for
   STRICT-JSON schema-v2 proposals. The network call is the Architect's ONE
   allowed side effect, confined to this module; a stub client is injected
   in tests and the real client is built lazily. Malformed output gets one
   repair round-trip then the batch is discarded. Every prompt is hashed
   into each proposal's provenance.
3. **critique** (`architect/critique.py`): a deterministic citation gate
   (every `supporting_observation` must resolve against the run's evidence —
   real tb tag/step, present competence report, in-range `code:` span — or
   the proposal is discarded) plus, online, an adversarial LLM review that
   attacks the citations/criteria/cost and revises confidence.
4. **emit** (`architect/__main__.py`): surviving proposals go into
   `runs/<id>/proposals/`; every draft/critique/emit decision is appended to
   `runs/<id>/architect/decisions.jsonl`.

`architect.offline: true` (config default) makes drafting a no-op, so the
whole pipeline runs and is tested offline; the analysis + citation gate are
deterministic and meaningful even then.

## Structural rules (tests/test_architect_no_execution.py)

- **Data, never code**: no subprocess/exec/eval/compile/importlib/os.system.
  The LLM network call is the single exception and is not a subprocess.
- **Reads modules under analysis as TEXT**; never imports
  `world`/`training`/`agent`/`memory` or `torch`. Allowed project imports:
  `evidence`, `proposals`, `ledger.competence`, and its own submodules.
- **The instrument stays separable**: proposals/review/researcher/evidence
  and the agent/training/world/memory layers never import `architect`
  (asserted both ways).
- **Writes confined** to `runs/<id>/architect/` (guarded writer) and the
  proposals queue (`proposals.save_proposal`).

## Constitutional files (`architect/constitution.py`)

A proposal whose `target` or attached diff touches a protected path is
rejected (`ConstitutionViolation`) before it can enter the queue — the
Architect may propose changes to the agent, its models, configs, and
experiment machinery, but not to the rules that contain it:

- `CLAUDE.md`, `docs/safety_scope.md`
- `tests/test_*_no_execution.py`, `tests/test_grad_isolation.py`,
  `tests/test_review_state_machine.py`, `tests/test_mirror.py`
- `review/` (the human review layer)
- `architect/constitution.py` itself (it cannot widen its own list)

Kept out of `proposals/schema.py` on purpose: `proposals/` must not import
`architect/`, so the check lives here and the Architect applies it at emit.

## Provenance (reproducibility standing rule)

Every architect proposal records `provenance = {evidence_bundle_hash,
generator_id: "architect", prompt_hash, model_id}` and, after critique, a
`critique` record. `source` is `architect:<model_id>`. An unapplied unified
diff, when present, is stored as an artifact under
`runs/<id>/architect/diffs/` and referenced in `artifacts`.

## ARCH-bench (`python -m benchmarks.archbench`)

Scores the Architect (and the rule-generator suite as the control arm) on a
battery of injected repository flaws. It MAY run subprocesses — but ONLY
inside disposable git worktrees under the temp dir, never the live checkout
(the live-repo guard is tested). Per flaw: worktree → inject the flaw →
tiny smoke train → run both arms → score detection, localization
precision/recall, and clean-control false positives. **Rationale quality is
persisted for human rubric scoring, not auto-judged in v0; predicted-vs-
realized benefit (apply the ground-truth fix, smoke A/B) is harness-applied
under human invocation.**

Reading a report (`benchmarks/results/<date>/archbench.md`): one section per
arm, a per-flaw table, nulls shown in full, scale stamped. **Offline the
architect arm drafts nothing, so its column is honestly empty** — the
rules-vs-architect comparison needs the online arm to be meaningful; the
offline table exists to prove the harness completes and to baseline the
rule arm.

## The Architect improving itself

Changes to the Architect's own prompts or passes are themselves proposals
through the same human-gated queue (docs/safety_scope.md). Nothing here
self-modifies.
