"""SelfQ (stage-E): one conditional self-model that unifies the body model
and the forecaster.

Both today's heads answer "what happens to my self-observable state if I do
X?" at different horizons — the body model at k=1 per action (dpos, success,
Delta-energy), the forecaster at k in {1, 10} per macro-plan (intero). SelfQ
is a single model conditioned on (h.detach(), an action/plan intent, a
learned horizon embedding) that outputs the whole self-observable vector at
horizon k, with adapter wrappers (selfq/adapters.py) presenting the exact
BodyModel and Forecaster interfaces so the policy features, arbiter, and
mirror are unchanged.

Gradient isolation is identical to the heads it replaces: SelfQ consumes
h.detach() and its output is detached before entering the policy; its loss
never reaches the core (tests/test_grad_isolation.py extends to it).
Selected by ``ledger.impl: heads | selfq`` (default heads — no silent swap).
"""

from __future__ import annotations

from selfq.model import (
    INTENT_ACTION,
    INTENT_PLAN,
    SelfPrediction,
    SelfQ,
    body_outputs,
    selfq_body_losses,
    selfq_forecaster_nll,
)

__all__ = [
    "INTENT_ACTION",
    "INTENT_PLAN",
    "SelfPrediction",
    "SelfQ",
    "body_outputs",
    "selfq_body_losses",
    "selfq_forecaster_nll",
]
