# Reflective Cartographer

A research prototype studying **explicit self-modeling in a contained
agent** — and the layers of machinery you can safely build *on top of* a
system that models itself, without ever letting it act on the world beyond
a fixed action table.

A recurrent agent lives in a seeded 2D gridworld (terrain, food, water,
shelter, day/night cycles) and carries a persistent internal state — a GRU
or recurrent state-space model (RSSM). Alongside the policy, a set of
**"Ledger"** heads estimate the agent's *own* properties from that state:
what its actions actually do (body model), whether a state change was caused
by itself or the world (attribution), how far its memories can be trusted
(reliability), where its interoceptive variables are heading under each plan
(forecaster), and whether its two self-estimates disagree (mirror
divergence). Every Ledger estimate is fed back to the policy as **detached**
input features — information, never gradient.

The scientific question is whether these explicit self-estimates earn their
keep: does an agent that can *read* its own capability estimates detect and
recover from unannounced capability shifts faster than an identical agent
that cannot? Experiment "levers" — capability shifts, ghost events that
mimic agent actions, regional food volatility, seasonal migrations,
exogenous resets — are injected from world config the agent can never
observe, and every lever leaves a ground-truth trail used strictly for
evaluation. Batteries under `experiments/` run each capability against its
matched ablation, and **negative results are reported alongside positive
ones** (a standing rule: log it, don't tune it away).

## Containment by construction

Containment is a property of the code, not a policy layered on top:

- The agent's only interface is `observe() -> obs` and `act(action_id)`
  against a **fixed 9-action table** — file, network, shell, or code access
  is not representable in its action space.
- Resets and checkpoints are **exogenous** experimental conditions; no
  objective, reward, or loss may reference run duration or the training
  process.
- **Gradient isolation**: every self-model head consumes `h.detach()`; its
  losses never reach the core. Enforced by `tests/test_grad_isolation.py`.
- The mirror divergence — the one signal that could tempt an agent to model
  its own training — is computed under `no_grad`, returned as numpy, and
  appears in no loss.
- Every layer that turns the agent's diagnostics into recommendations is
  **data, never code**: a human reviews and executes each approved change by
  hand. Structural tests ban execution machinery and confine writes.

See [docs/safety_scope.md](docs/safety_scope.md) for the full scope
statement and its enforcement points.

## The layers

The project is built in layers, each with its own structural guarantees and
preserved acceptance evidence:

1. **The agent + Ledger** (`agent/`, `ledger/`, `memory/`) — the recurrent
   policy and its self-model heads, trained by recurrent PPO or a circadian
   wake/sleep trainer with Dreamer-style imagination.
2. **The proposal layer** (`proposals/`, `review/`) — the agent's
   diagnostics generate blind-reviewed recommendations ("MOVE_E fails 90%;
   consider recalibrating"). **Proposals are data, never code**; a human
   executes every approved change. A dual-source control (ledger vs
   logs-only) keeps the science honest.
3. **The internal researcher** (`researcher/`) — a hypothesis registry with
   sequential-change monitors (CUSUM), and a ranked research agenda of what
   the agent doesn't understand and which experiment would reduce that
   uncertainty most efficiently. It only *proposes*; scope is limited to the
   agent-in-the-world and its own models (schema-enforced).
4. **The evidence plane** (`evidence/`) — one read-only, content-hashed
   store over a run's artifacts (JSONL, TB scalars, competence, a repo
   snapshot), shared by the proposal and researcher layers with citation
   provenance.
5. **The Architect** (`architect/`) + **ARCH-bench** (`benchmarks/archbench/`)
   — an *experimenter-side* instrument that reads the repo as text and an
   LLM drafts + self-critiques proposals for a human to review (it never
   applies, and run code never imports it). A **constitution** rejects any
   proposal that touches the safety files. ARCH-bench scores it against
   injected repository flaws — but only inside disposable git worktrees,
   never the live checkout.
6. **SelfQ** (`selfq/`) — one unified conditional self-model that can replace
   the separate body model + forecaster behind adapters, selected by
   `ledger.impl: heads | selfq` (default `heads` — no silent swap).

## Quickstart

```bash
pip install torch numpy pyyaml pytest matplotlib pygame imageio imageio-ffmpeg streamlit
# (--break-system-packages if your Python is externally managed)

pytest -x -q -m "not slow"                   # fast test suite (400+ tests, ~1-2 min)
python train.py --config configs/smoke.yaml  # < 3 min CPU smoke train
python -m viz.viewer --live runs/<id>        # watch it live (or --replay)
streamlit run viz/dashboard.py               # run browser / timeline / proposals / agenda
python -m experiments.batteries.full_battery --seeds 5 --scale full   # the full battery
python -m architect --run runs/<id> --offline                          # draft proposals (offline)
```

(`make test` / `make soak` wrap these on platforms with `make`; on Windows
run the underlying `pytest` commands directly.)

The long measurement sweeps (`full_battery`, `scripts/scale_curves.py`,
`scripts/verify_selfq.py`) are **resumable**: they cache each finished unit
and skip it on re-launch, so you can stop a run (or power off the laptop)
and continue later by re-running the same command — losing at most the one
short run in flight.

## Repo map

| path | contents |
|------|----------|
| `world/` | grid engine (`engine.py`), config resolution, experiment levers (`levers.py` — agent-side code may never import this) |
| `agent/` | observation encoder, GRU core, RSSM core (world model + pose head + dynamics ensemble), actor-critic, macro-plan arbiter (`drives.py`) |
| `ledger/` | the self-model heads: `body_model.py`, `attribution.py`, `reliability.py`, `forecaster.py`, `mirror.py`, `competence.py` — all consume `h.detach()` |
| `selfq/` | the unified conditional self-model + adapters (stage-E) |
| `memory/` | surprise-gated episodic store with spatial retrieval |
| `training/` | recurrent PPO (`ppo.py`), circadian wake/sleep trainer (`sleep.py`), prioritized replay, checkpoints, loggers, monitors |
| `evidence/` | the shared read-only evidence plane: store, source-scoped views, content-hashed bundles |
| `proposals/` | proposal schema (v2), dual-source generators — data, never code |
| `review/` | blind review queue + CLI; approval emits a human-run ticket |
| `researcher/` | hypothesis registry + CUSUM monitors, questions, ranked agenda (v1 heuristic / v2 EIG) |
| `architect/` | experimenter-side analysis + LLM drafting + self-critique + constitution |
| `benchmarks/archbench/` | flaw battery scoring the Architect vs the rule generators, in disposable worktrees |
| `experiments/` | metrics (`metrics.py`), runner + evaluation ladder, batteries |
| `viz/` | pygame viewer, streamlit dashboard (6 pages), matplotlib report plots |
| `configs/` | `base.yaml` (canonical defaults), `smoke.yaml` (<3 min CPU), `full.yaml` (overnight) |
| `scripts/` | per-stage acceptance scripts (`verify_*.py`), scale curves, 1M-tick soak |
| `tests/` | 400+ fast tests + slow-marked soak/train smokes; structural tests enforce the hard rules |
| `docs/acceptance/` | preserved acceptance evidence per stage (reports, plots, CSVs) |

## Build stages

The system was built one stage per session, each committed and **tagged**,
with acceptance evidence preserved under `docs/acceptance/<stage>/`:

| tag | what landed |
|-----|-------------|
| `stage-1` … `stage-6a` | world, encoder/GRU/RSSM cores, PPO + circadian training, the Ledger heads (body, attribution, reliability, forecaster), episodic memory, mirror + kidnapped-agent test |
| `stage-7a` … `stage-7f` | competence tracker, proposal schema + dual-source generators, blind review queue + CLI, continuity metric, dashboard proposals page, the flagship proposal-quality battery |
| `stage-8a` … `stage-8d` | hypothesis registry + contradiction monitors, questions + ranked agenda v1, EIG ranker v2, researcher-value battery + dashboard agenda page |
| `stage-A` | baselines become evidence: kidnapped-config alignment, minimum-viable-scale contracts + evidence stamping, head-convergence scale curves |
| `stage-B` | sequential CUSUM monitors that catch the onset-detector's blind spot |
| `stage-C` | the evidence plane, proposal schema v2 (provenance/intervention-class), agenda unified into the queue, the tier-0/tier-1 evaluation ladder |
| `stage-D` | the Architect (analysis → LLM draft → self-critique → emit) + constitution + ARCH-bench v0 |
| `stage-E` | SelfQ unified self-model + parity gate |

## Documentation

- [docs/architecture.md](docs/architecture.md) — components, wake/sleep data
  flow, the gradient-isolation rule, the evidence plane, config reference
- [docs/experiments.md](docs/experiments.md) — every battery: protocol,
  metrics, minimum-viable-scale contracts, how to run and read the outputs
- [docs/proposals.md](docs/proposals.md) — proposal schema, generator
  triggers, blind-review protocol, the evaluation ladder
- [docs/researcher.md](docs/researcher.md) — hypothesis schema, monitor
  templates (incl. CUSUM), ranking formulas v1/v2, the scope rule
- [docs/architect.md](docs/architect.md) — the Architect pipeline,
  structural rules, constitutional files, how to read an ARCH-bench report
- [docs/training.md](docs/training.md) — canonical hyperparameters, stage
  notes, scale-up path and expected wall-clock
- [docs/logging.md](docs/logging.md) — JSONL schema + every TensorBoard scalar
- [docs/visualization.md](docs/visualization.md) — viewer keys, dashboard
  pages, plot inventory
- [docs/safety_scope.md](docs/safety_scope.md) — containment by construction;
  what is explicitly out of scope
- [TODO.md](TODO.md) — deferred work with blocking dependencies

## A note on evidence and scale

Most committed numbers are **smoke-scale** — evidence that the *machinery*
works end-to-end, deliberately below the minimum viable scale for
architecture claims (which the stage-A `MIN_VIABLE_SCALE` contracts make
explicit: a metric below its viable scale is stamped machinery-only, and the
aggregation refuses to pool it with real evidence). Several honest nulls are
recorded as prominently as any positive. The full-scale confirmation
(overnight `full_battery --scale full` + `scale_curves` + full-scale SelfQ
parity) is the standing follow-up described in
[docs/acceptance/stage-A](docs/acceptance/stage-A/README.md).
