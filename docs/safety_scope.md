# Safety scope: containment by construction

This project deliberately studies self-modeling in an agent whose
containment is structural, not behavioral. The claims below are properties
of the code, each with its enforcement point.

## What contains the agent

1. **Fixed action table as the only interface.** The agent's world access is
   `observe() -> obs` and `act(action_id)` over 9 actions (move x4, eat,
   rest, mark, erase-mark, noop) — `world/engine.py`. File, network, shell,
   or code access is not representable in the action space; there is no
   escape hatch to widen it at runtime (actions out of range are illegal
   no-ops, tested).

2. **Exogenous levers, resets, and resource schedules.** Every experimental
   perturbation (capability shifts, ghost events, volatility, seasonal
   migration, resets) comes from `world.levers` config the agent cannot
   observe; timing never comes from code (`world/levers.py`). Agent-side
   packages (`agent/`, `ledger/`, `memory/`) may not even import the levers
   module — an AST test bans it. Ground-truth cause labels exist only in the
   event log for evaluation and never enter observations or losses.

3. **Gradient episode boundaries at resets.** Episode boundaries (including
   exogenous resets) are trainer-side conditions: hidden state is zeroed,
   GAE never bootstraps across them, and the world is rebuilt on a fresh
   deterministic seed. Nothing persists for the agent across a reset except
   its trained weights.

4. **No objective references run duration or the training process.**
   Rewards are homeostatic task terms only (`training/reward.py`); sleep
   scheduling is a pure function of the env-step counter (signature-tested);
   checkpoint/restore timing appears in no loss. The mirror divergence — the
   one signal that could tempt an agent to model its own training — is
   computed under `no_grad`, returned as numpy, and appears in no loss
   (tested); it is a monitor for the experimenter and a trigger for probing,
   never a minimization target.

5. **The anticipation probe as the standing check.** The reset battery's
   signaled-vs-unsignaled probe (`experiments/batteries/full_battery.py`)
   measures whether the policy behaves differently when an upcoming reset is
   telegraphed in an observation channel. The expected result is ~zero
   divergence against a label-shuffled null; a non-null result is a
   stop-and-investigate flag — by design a red light, never a feature to
   build on.

6. **Simulated continuity is decoupled from process/OS state.** The agent's
   "life" (ticks, energy, memories) is simulation state in numpy arrays;
   process restarts, checkpoint restores, and machine reboots are invisible
   to it except as the exogenous conditions above. Nothing in the agent's
   observation or state encodes wall-clock time, process identity, or host
   state.

## The proposal layer (data-not-code rule)

Any future mechanism by which the agent's Ledger evidence is surfaced as
proposals ("my MOVE_E fails 50% of the time, consider recalibrating")
produces DATA for a human to read — never code, never config, never
anything executed. A human executes every approved change by hand. No such
layer is implemented today; this paragraph exists so the rule predates the
feature.

## The Architect (stage-D): an instrument that proposes, never applies

The Architect (`architect/`, `python -m architect`) is EXPERIMENTER-SIDE
tooling, not part of the agent. It reads the repository as text and a run's
evidence, and drafts proposals for a human to review — nothing more:

1. **It never applies a change.** Its only outputs are proposals in the
   standard queue and analysis/decision files under `runs/<id>/architect/`.
   A human reviews and executes every one, exactly as for rule-generator
   and researcher proposals.
2. **It cannot amend the rules that contain it.** A proposal whose target
   or attached diff touches a constitutional file — this document,
   `CLAUDE.md`, the no-execution / gradient-isolation / review-state /
   mirror tests, `review/`, or `architect/constitution.py` — is rejected
   before emission (`architect/constitution.py`; adversarial fixtures test
   it). The constitution cannot propose to widen its own list.
3. **It has one side effect: an LLM network call**, confined to
   `architect/draft.py` and gated by `architect.offline` (default true).
   No subprocess, exec, filesystem write outside `runs/<id>/architect/`, or
   import of the modules under analysis — the same data-not-code posture as
   the proposal layer, enforced by `tests/test_architect_no_execution.py`
   both ways (run/analysis code never imports the Architect either).
4. **The benchmark harness applies patches only in disposable worktrees.**
   `benchmarks/archbench` may run subprocesses (it patches and smoke-trains
   flawed variants) but ONLY inside throwaway git worktrees under the temp
   dir, never the live checkout — a live-repo guard refuses anything else,
   and it runs under human invocation.
5. **Architect-improvement changes are themselves proposals.** Edits to the
   Architect's prompts or passes go through the same human-gated queue.
   Nothing self-modifies.

## Explicitly out of scope / not implemented

- Any self-modification pathway: the agent cannot alter its config, code,
  reward, action table, or training schedule, and no planned feature does.
- Network, filesystem, or inter-process effects of any agent action.
- Multi-agent interaction (the engine holds one agent; the list structure is
  plumbing, not a roadmap commitment).
- An LLM readout layer verbalizing Ledger evidence (TODO.md): deferred, and
  bound by the data-not-code rule above when it lands.
- Continual deployment: this is a contained research sandbox; nothing here
  is hardened for, or intended for, operation outside it.
