"""Stage-A2: MIN_VIABLE_SCALE stamping and the pooling refusal.

The battery stamps every summary row 'evidence' (this scale meets the
test's premise contract) or 'machinery-only' (plumbing validation only);
experiments/metrics.py must refuse to aggregate across that boundary.
"""

from __future__ import annotations

import pytest

from experiments import metrics as M
from experiments.batteries.full_battery import (
    MIN_VIABLE_SCALE,
    SCALES,
    TESTS,
    evidence_stamp,
)

# ------------------------------------------------------------- stamping


def test_every_test_has_a_contract() -> None:
    # Adding a battery test without declaring its viability contract must
    # fail loudly, not stamp '?'.
    assert set(MIN_VIABLE_SCALE) == set(TESTS)


def test_known_expectations_at_quick_scale() -> None:
    sc = SCALES["quick"]
    # Trustworthy at quick per results/20260708-1311/ANALYSIS.md:
    assert evidence_stamp("reset_battery", sc) == "evidence"
    assert evidence_stamp("memory_reliability", sc) == "evidence"
    # Demonstrably undertrained at quick:
    for test in ("capability_shift", "ghost_attribution", "forecaster_nmse",
                 "kidnapped_agent", "seasonal_shift", "sleep_ablation"):
        assert evidence_stamp(test, sc) == "machinery-only", test


def test_known_expectations_at_full_scale() -> None:
    sc = SCALES["full"]
    # Demonstrated sufficient at or below full's budget:
    assert evidence_stamp("forecaster_nmse", sc) == "evidence"   # stage-4c 50k/100
    assert evidence_stamp("kidnapped_agent", sc) == "evidence"   # stage-6a 24576/150
    # UNKNOWN minimum stays machinery-only even at full — never guessed:
    assert evidence_stamp("capability_shift", sc) == "machinery-only"
    assert evidence_stamp("seasonal_shift", sc) == "machinery-only"
    # Known-sufficient scale (200k) above full's 50k budget:
    assert evidence_stamp("ghost_attribution", sc) == "machinery-only"


def test_unknown_minimum_never_stamps_evidence() -> None:
    # Even an absurdly large scale cannot satisfy an UNKNOWN (None) minimum.
    huge = {"train_ticks": 10**9, "sleep_grad_steps": 10**6,
            "kidnapped_sleep_grad_steps": 10**6}
    assert evidence_stamp("capability_shift", huge) == "machinery-only"


def test_kidnapped_uses_its_own_calibration_budget() -> None:
    # The kidnapped contract checks the test's own consolidation knob, not
    # the battery-wide one.
    sc = dict(SCALES["full"], sleep_grad_steps=1,
              kidnapped_sleep_grad_steps=150)
    assert evidence_stamp("kidnapped_agent", sc) == "evidence"
    sc["kidnapped_sleep_grad_steps"] = 40
    assert evidence_stamp("kidnapped_agent", sc) == "machinery-only"


# ------------------------------------------------------------- pooling


def _row(value: float, stamp: str | None) -> dict:
    r: dict = {"delta": value}
    if stamp is not None:
        r["evidence_stamp"] = stamp
    return r


def test_pool_within_one_stamp_matches_mean_ci() -> None:
    rows = [_row(1.0, "evidence"), _row(3.0, "evidence")]
    assert M.pooled_mean_ci(rows, "delta") == M.mean_and_ci95([1.0, 3.0])
    only_smoke = [_row(1.0, "machinery-only"), _row(2.0, "machinery-only")]
    mean, _ = M.pooled_mean_ci(only_smoke, "delta")
    assert mean == pytest.approx(1.5)


def test_pool_refuses_mixed_stamps() -> None:
    rows = [_row(1.0, "evidence"), _row(100.0, "machinery-only")]
    with pytest.raises(ValueError, match="refusing to pool"):
        M.pooled_mean_ci(rows, "delta")


def test_pool_refuses_unstamped_rows() -> None:
    with pytest.raises(ValueError, match="unstamped"):
        M.pooled_mean_ci([_row(1.0, "evidence"), _row(2.0, None)], "delta")
