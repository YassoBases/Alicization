"""Expected-information-gain ranker v2 (config-selectable; v1 remains the
ablation).

APPROXIMATIONS (per spec, documented here):

Directed visits (world_uncertainty questions) — Plan2Explore-style: roll
the RSSM prior along an imagined visit and estimate the expected reduction
in ensemble disagreement over the region's cells. The world model is
INJECTED as a duck-typed adapter (built in experiments/model_adapter.py) —
this package imports neither torch nor agent/ (structural rules). The
adapter contract:

    adapter.region_disagreement(region) -> float          current mean
    adapter.imagined_visit_reduction(region, steps) -> float
        expected fractional reduction in disagreement from visiting, i.e.
        (before - after) / before estimated by imagined rollouts.

EIG(directed_visit) = disagreement * imagined_visit_reduction.

Probe-action batches (capability questions) — expected Brier reduction
from n additional labeled transitions under a Beta posterior on the
action's success rate: with a Beta(a, b) posterior (a = successes+1,
b = failures+1), the expected squared error of the point estimate is the
posterior variance V = ab / ((a+b)^2 (a+b+1)); after n more observations
the variance shrinks to approximately V * (a+b+1) / (a+b+n+1).
EIG(probe) = V_now - V_after. This treats "Brier reduction" as posterior-
variance reduction about the true success rate — a first-order
approximation that ignores drift during the probe batch.

The noisy-TV guard SURVIVES v2: EIG is still multiplied by the competence
tractability term, so an irreducibly random region (max disagreement, zero
learning progress) cannot buy its way to the top with a big predicted gain
the world model cannot actually realize.

Every ranked item records predicted_gain so the researcher-value battery
(P9.4) can score EIG calibration against realized reductions.
"""

from __future__ import annotations

from typing import Any, Protocol

from ledger.competence import CompetenceReport
from proposals.schema import Proposal
from researcher.agenda import AgendaItem, _novelty, _tractability, rank_v1
from researcher.questions import Question


class ModelAdapter(Protocol):
    """What the caller must inject (see experiments/model_adapter.py)."""

    def region_disagreement(self, region: tuple[int, int]) -> float: ...

    def imagined_visit_reduction(self, region: tuple[int, int],
                                 steps: int) -> float: ...


def beta_posterior_variance(successes: int, failures: int) -> float:
    """Var of a Beta(successes+1, failures+1) posterior."""
    a, b = successes + 1, failures + 1
    return (a * b) / (((a + b) ** 2) * (a + b + 1))


def probe_batch_eig(successes: int, failures: int, n_probes: int) -> float:
    """Expected posterior-variance (~Brier) reduction from n more labels."""
    v_now = beta_posterior_variance(successes, failures)
    a_b = successes + failures + 2
    v_after = v_now * (a_b + 1) / (a_b + n_probes + 1)
    return v_now - v_after


def rank_v2(
    questions: list[Question],
    proposals: list[Proposal],
    competence: CompetenceReport | None,
    adapter: ModelAdapter | None = None,
    action_counts: dict[int, tuple[int, int]] | None = None,
    history: list[dict[str, Any]] | None = None,
    visit_steps: int = 15,
) -> list[AgendaItem]:
    """v1 scoring with the VALUE term replaced by an EIG estimate where one
    is computable (world_uncertainty via the adapter, capability via the
    Beta math); everything else falls back to the v1 heuristic. Items carry
    predicted_gain for later calibration scoring."""
    items = rank_v1(questions, proposals, competence, history)
    by_ref = {q.id: q for q in questions}
    history = history or []

    rescored: list[AgendaItem] = []
    for item in items:
        if item.kind != "question":
            rescored.append(item)
            continue
        q = by_ref[item.ref]
        gain: float | None = None
        if q.type == "world_uncertainty" and adapter is not None and q.region:
            disagreement = adapter.region_disagreement(q.region)
            reduction = adapter.imagined_visit_reduction(q.region, visit_steps)
            gain = float(disagreement * max(0.0, reduction))
        elif q.type == "capability_gap" and action_counts:
            action = int(q.params.get("action", -1))
            if action in action_counts:
                s, f = action_counts[action]
                n = int(next((e.get("n", 32) for e in q.candidate_experiments
                              if e.get("name") == "probe_action_batch"), 32))
                gain = probe_batch_eig(s, f, n)
        if gain is not None:
            tract = _tractability(q.region, competence)
            novelty = _novelty(q.id, history)
            cost = float(item.experiment.get("cost", 1.0))
            item.predicted_gain = gain
            item.decomposition = {"value": gain, "tractability": tract,
                                  "novelty": novelty, "cost": cost}
            item.score = gain * tract * novelty / cost
        rescored.append(item)

    rescored.sort(key=lambda i: (-i.score, i.id))
    return rescored
