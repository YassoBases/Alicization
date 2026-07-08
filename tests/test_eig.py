"""EIG ranker v2: known-value Beta math, plausible divergence from v1,
and the noisy-TV guard surviving the upgrade."""

from __future__ import annotations

import pytest

from researcher.questions import Question
from researcher.agenda import TRACTABILITY_FLOOR, rank_v1
from tests.test_agenda import competence, q, region

class FakeAdapter:
    """Region (0,0): huge disagreement the model CANNOT reduce (noisy TV);
    region (1,1): moderate disagreement, highly reducible."""

    def region_disagreement(self, region):
        return {(0, 0): 1.0, (1, 1): 0.4}[tuple(region)]

    def imagined_visit_reduction(self, region, steps):
        return {(0, 0): 0.9, (1, 1): 0.8}[tuple(region)]


def test_beta_probe_eig_known_values() -> None:
    from researcher.eig import beta_posterior_variance, probe_batch_eig

    # Beta(1,1) uniform: var = 1/12.
    assert beta_posterior_variance(0, 0) == pytest.approx(1 / 12)
    gain = probe_batch_eig(0, 0, n_probes=32)
    assert 0 < gain < 1 / 12                 # reduces, never below zero
    assert probe_batch_eig(500, 500, 32) < gain  # well-known action: tiny gain


def test_rank_v2_differs_plausibly_and_keeps_noisy_tv_guard() -> None:
    from researcher.eig import rank_v2

    noisy_tv = q("q-noise", 1.0, region=(0, 0))
    learnable = q("q-learn", 0.4, region=(1, 1))
    comp = competence(region(0, 0, 0.0),   # zero progress: the guard bites
                      region(1, 1, 0.05))

    v1 = rank_v1([noisy_tv, learnable], [], comp)
    v2 = rank_v2([noisy_tv, learnable], [], comp, adapter=FakeAdapter())

    # v2 rescored the learnable question with an EIG value (0.4 * 0.8),
    # different from its raw uncertainty (0.4) -> scores differ from v1.
    learn_v2 = next(i for i in v2 if i.ref == "q-learn")
    assert learn_v2.predicted_gain == pytest.approx(0.32)
    learn_v1 = next(i for i in v1 if i.ref == "q-learn")
    assert learn_v2.score != learn_v1.score

    # The noisy-TV region has the LARGEST predicted gain (0.9) but zero
    # learning progress: tractability floors it — never the top slot.
    assert v2[0].ref == "q-learn"
    noise_v2 = next(i for i in v2 if i.ref == "q-noise")
    assert noise_v2.predicted_gain == pytest.approx(0.9)
    assert noise_v2.decomposition["tractability"] == TRACTABILITY_FLOOR


def test_rank_v2_capability_uses_beta_eig() -> None:
    from researcher.eig import probe_batch_eig, rank_v2

    cap_q = Question(
        id="q-cap-2", type="capability_gap",
        statement="is capability (action 2) degraded or mis-modeled?",
        evidence_refs=["jsonl:action_success:action=2"],
        candidate_experiments=[{"name": "probe_action_batch", "action": 2,
                                "n": 32, "cost": 1.0}],
        params={"action": 2}, uncertainty=0.6,
    )
    ranked = rank_v2([cap_q], [], None, action_counts={2: (5, 5)})
    assert ranked[0].predicted_gain == pytest.approx(probe_batch_eig(5, 5, 32))
