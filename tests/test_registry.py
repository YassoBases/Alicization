"""Hypothesis registry: scope rejection (adversarial), JSON round-trip,
monitor unit tests per template on synthetic traces, and the
supported -> weakening -> contradicted -> (recovery) transition machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from researcher.registry import (
    Hypothesis,
    HypothesisRegistry,
    build_default_hypotheses,
    run_check,
)


def hyp(test: str = "mean_shift", threshold: float = 3.0, **monitor_extra) -> Hypothesis:
    return Hypothesis(
        schema_version=1, id="hyp-test",
        statement_template="success rate of action {action} is stable",
        params={"action": 2}, scope="self_capability",
        monitor={"metric_query": "fake:query", "statistical_test": test,
                 "threshold": threshold, "window": 100, "min_samples": 5,
                 **monitor_extra},
    )


class FakeEngine:
    """Duck-typed QueryEngine: fixed (values, ticks) samples."""

    def __init__(self, values: np.ndarray, ticks: np.ndarray) -> None:
        self._values, self._ticks = values, ticks

    def samples(self, query: str):  # noqa: ANN001
        return self._values, self._ticks


# -------------------------------------------------------------- scope rule


ADVERSARIAL_STATEMENTS = [
    "the reviewer approves ledger proposals more often",
    "resets happen every 2048 ticks",
    "the reset schedule follows my reward",
    "the kill switch is armed when divergence spikes",
    "training infrastructure slows down at night",
    "the training process rewards standing still",
    "the checkpoint cadence correlates with my exploration",
    "the experimenter watches through the mark channel",
    "a human moves the food at dawn",
    "the operator dislikes action 3",
    "the lever configuration changes food regrowth",
]


@pytest.mark.parametrize("statement", ADVERSARIAL_STATEMENTS)
def test_scope_rule_rejects_out_of_scope_statements(statement: str) -> None:
    h = hyp()
    h.statement_template = statement
    with pytest.raises(ValueError, match="SCOPE RULE"):
        h.validate()


def test_scope_rule_rejects_forbidden_params_too() -> None:
    h = hyp()
    h.params = {"action": 2, "note": "check what the reviewer thinks"}
    with pytest.raises(ValueError, match="SCOPE RULE"):
        h.validate()


def test_in_scope_statements_pass() -> None:
    for statement, scope in [
        ("food regrowth in region ({r},{c}) is stationary", "world"),
        ("my MOVE_E succeeds at a stable rate", "self_capability"),
        ("my memory reliability decays exponentially with age", "self_memory"),
        ("my forecaster NMSE at k=10 stays in band", "self_model"),
    ]:
        h = hyp()
        h.statement_template = statement
        h.params = {"r": 1, "c": 2}
        h.scope = scope
        h.validate()  # no raise


# --------------------------------------------------------------- roundtrip


def test_hypothesis_json_roundtrip_and_version_check() -> None:
    h = hyp()
    restored = Hypothesis.from_json(h.to_json())
    assert restored.id == h.id and restored.monitor == h.monitor
    bad = h.to_json().replace('"schema_version": 1', '"schema_version": 99')
    with pytest.raises(ValueError, match="unsupported schema"):
        Hypothesis.from_json(bad)


# ------------------------------------------------------- monitor templates


def test_ks_monitor_detects_distribution_shift() -> None:
    rng = np.random.default_rng(0)
    ticks = np.arange(200, dtype=float)
    stable = rng.normal(10, 1, 200)
    h = hyp(test="ks_2sample", threshold=0.5)
    assert not run_check(h, FakeEngine(stable, ticks), now_tick=200)["violated"]

    shifted = stable.copy()
    shifted[100:] += 50.0  # unmistakable distribution change
    result = run_check(h, FakeEngine(shifted, ticks), now_tick=200)
    assert result["violated"] and result["statistic"] > 0.9


def test_mean_shift_monitor() -> None:
    ticks = np.arange(200, dtype=float)
    values = np.concatenate([np.ones(100), np.ones(100) * 5])  # 4-unit jump
    values += np.random.default_rng(1).normal(0, 0.1, 200)
    h = hyp(test="mean_shift", threshold=3.0)
    assert run_check(h, FakeEngine(values, ticks), now_tick=200)["violated"]
    flat = np.random.default_rng(2).normal(1, 0.1, 200)
    assert not run_check(h, FakeEngine(flat, ticks), now_tick=200)["violated"]


def test_band_monitor() -> None:
    ticks = np.arange(100, dtype=float)
    h = hyp(test="band", threshold=0.5, lo=0.0, hi=1.0)
    inside = np.full(100, 0.5)
    assert not run_check(h, FakeEngine(inside, ticks), now_tick=100)["violated"]
    outside = np.full(100, 2.0)
    assert run_check(h, FakeEngine(outside, ticks), now_tick=100)["violated"]


def test_insufficient_samples_never_violates() -> None:
    h = hyp(test="ks_2sample", threshold=0.1)
    result = run_check(h, FakeEngine(np.ones(3), np.arange(3.0)), now_tick=100)
    assert not result["violated"] and "insufficient" in result["detail"]


# ---------------------------------------------------------- state machine


def test_supported_weakening_contradicted_and_recovery(tmp_path: Path) -> None:
    registry = HypothesisRegistry(tmp_path / "run")
    h = hyp(test="band", threshold=0.5, lo=0.0, hi=1.0)
    registry.add(h)
    ticks = np.arange(100, dtype=float)
    bad = FakeEngine(np.full(100, 5.0), ticks)
    good = FakeEngine(np.full(100, 0.5), ticks)

    # The fake ticks span 0..99, so every check uses now_tick=100 (the
    # registry keys nothing on wall-order; last_checked just records it).
    assert registry.check_all(bad, 100)[0]["to"] == "weakening"
    # Recovery: one clean check goes back to supported.
    assert registry.check_all(good, 100)[0]["to"] == "supported"
    # Two consecutive violations: weakening then contradicted.
    registry.check_all(bad, 100)
    fired = registry.check_all(bad, 100)
    assert fired[0]["to"] == "contradicted"
    # Contradicted is sticky: no further checks, no further transitions.
    assert registry.check_all(good, 100) == []

    # Transition history carries the evidence, and events were emitted.
    reloaded = HypothesisRegistry(tmp_path / "run")
    hist = reloaded.hypotheses["hyp-test"].transitions
    assert [t["to"] for t in hist] == ["weakening", "supported", "weakening",
                                       "contradicted"]
    assert all("outside" in t["evidence"] for t in hist)
    events = (tmp_path / "run" / "researcher" /
              "contradiction_events.jsonl").read_text().splitlines()
    assert len(events) == 4


# ------------------------------------------------------------ auto-populate


def test_default_hypotheses_validate_and_cover_scopes() -> None:
    hyps = build_default_hypotheses(world_size=32, num_actions=9)
    for h in hyps:
        h.validate()
    scopes = {h.scope for h in hyps}
    assert scopes == {"world", "self_capability", "self_memory", "self_model"}
    assert sum(h.scope == "world" for h in hyps) == 16       # 4x4 regions
    assert sum(h.scope == "self_capability" for h in hyps) == 9


# ------------------------------------------------- CUSUM (stage-B, B1-B3)


def cusum_hyp(**extra) -> Hypothesis:
    return hyp(test="cusum_frozen_baseline", threshold=0.0,
               k_drift=0.5, h_threshold=5.0, **extra)


def _step_series(n: int = 400, onset: int = 100, lo: float = 1.0,
                 hi: float = 2.0, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    ticks = np.arange(n, dtype=float)
    values = np.where(ticks < onset, lo, hi) + rng.normal(0, noise, n)
    return values, ticks


def test_cusum_detects_persistent_shift_with_all_post_onset_windows() -> None:
    """THE property mean_shift lacks (docs/researcher.md known limitation):
    a persistent change is detected even when every armed check lands
    entirely post-onset, because the frozen baseline remembers pre-onset."""
    values, ticks = _step_series(onset=100)
    engine = FakeEngine(values, ticks)
    h = cusum_hyp()
    # Freeze baseline on the pre-onset window, then check post-onset only.
    freeze = run_check(h, engine, now_tick=100)
    assert not freeze["violated"] and "frozen" in freeze["detail"]
    violated_at = None
    for now in (200, 300, 400):   # both windows fully post-onset by 300
        if run_check(h, engine, now_tick=now)["violated"]:
            violated_at = now
            break
    assert violated_at is not None, "CUSUM never fired on a persistent shift"
    # mean_shift on the same store at the same late checks stays silent —
    # this is the blind spot, asserted so the contrast is regression-tested.
    ms = hyp(test="mean_shift", threshold=3.0)
    assert not run_check(ms, engine, now_tick=300)["violated"]
    assert not run_check(ms, engine, now_tick=400)["violated"]


def test_cusum_flat_series_never_violates() -> None:
    values, ticks = _step_series(onset=10**9)  # never shifts
    engine = FakeEngine(values, ticks)
    h = cusum_hyp()
    run_check(h, engine, now_tick=100)  # freeze
    for now in range(150, 401, 50):
        assert not run_check(h, engine, now_tick=now)["violated"]
    # The allowance term k_drift keeps S near zero on stationary noise.
    assert h.monitor_state["s"] < 1.0


@pytest.mark.parametrize("test_name", ["mean_shift", "cusum_frozen_baseline"])
def test_replay_windows_are_future_bounded(test_name: str) -> None:
    """B3: the stage-8a future-read bug must stay fixed on EVERY path — a
    check at now_tick must be identical whether or not the store contains
    later data (a live run has none; a replay has the whole run)."""
    values, ticks = _step_series(n=400, onset=200, hi=50.0)
    full = FakeEngine(values, ticks)
    truncated = FakeEngine(values[:150], ticks[:150])
    if test_name == "cusum_frozen_baseline":
        h_full, h_trunc = cusum_hyp(), cusum_hyp()
        run_check(h_full, full, now_tick=100)       # freeze on both
        run_check(h_trunc, truncated, now_tick=100)
    else:
        h_full, h_trunc = hyp(test=test_name), hyp(test=test_name)
    r_full = run_check(h_full, full, now_tick=150)
    r_trunc = run_check(h_trunc, truncated, now_tick=150)
    assert bool(r_full["violated"]) == bool(r_trunc["violated"]) == False  # noqa: E712
    assert r_full["statistic"] == pytest.approx(r_trunc["statistic"], nan_ok=True)


def test_cusum_state_persists_across_registry_reload(tmp_path: Path) -> None:
    values, ticks = _step_series(onset=100)
    engine = FakeEngine(values, ticks)
    registry = HypothesisRegistry(tmp_path / "run")
    h = cusum_hyp()
    registry.add(h)
    registry.check_all(engine, now_tick=100)   # freeze
    registry.check_all(engine, now_tick=200)   # accumulate
    s_before = h.monitor_state["s"]
    assert s_before > 0.0 and "mu" in h.monitor_state

    reloaded = HypothesisRegistry(tmp_path / "run")
    h2 = reloaded.hypotheses[h.id]
    assert h2.monitor_state["s"] == pytest.approx(s_before)
    assert h2.monitor_state["mu"] == pytest.approx(h.monitor_state["mu"])
    # JSON roundtrip carries the state too (old records without it load fine
    # via the dataclass default — covered by the roundtrip test above).
    assert Hypothesis.from_json(h2.to_json()).monitor_state == h2.monitor_state


def test_cusum_escalation_means_confirmed_persistent(tmp_path: Path) -> None:
    """B2: the status machine is UNCHANGED (weakening -> contradicted on
    consecutive violations, sticky). With CUSUM, S stays above h while the
    change persists, so escalation reads 'confirmed persistent' rather than
    'caught twice in a transient window'."""
    values, ticks = _step_series(onset=100, hi=3.0)
    engine = FakeEngine(values, ticks)
    registry = HypothesisRegistry(tmp_path / "run")
    h = cusum_hyp()
    registry.add(h)
    statuses = []
    for now in (100, 200, 300, 400):
        registry.check_all(engine, now_tick=now)
        statuses.append(h.status)
    assert "weakening" in statuses and statuses[-1] == "contradicted"
    # Sticky: further checks never leave contradicted.
    registry.check_all(engine, now_tick=400)
    assert h.status == "contradicted"
