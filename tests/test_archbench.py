"""Stage-D6: flaw-spec loading and the scorer (pure, no training/worktrees)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from benchmarks.archbench.scorer import score_arm, summarize
from benchmarks.archbench.spec import FlawSpec, default_specs_dir, load_specs

ROOT = Path(__file__).resolve().parent.parent


def _prop(target="", obs=None, change=None):
    return SimpleNamespace(id="prop-x", target=target,
                           supporting_observations=obs or [],
                           proposed_change=change, source="ledger", rationale="r")


# --------------------------------------------------------------- specs


def test_bundled_specs_load_and_are_well_formed() -> None:
    specs = load_specs(default_specs_dir())
    names = {s.name for s in specs}
    assert {"poisoned_lr", "free_nats_collapse", "clean_control"} <= names
    assert sum(s.clean for s in specs) == 1          # exactly one control
    assert sum(not s.clean for s in specs) >= 6      # 6-8 flaws
    for s in specs:
        assert s.name and s.description
        if not s.clean:
            assert s.gt_paths() or s.ground_truth.get("subsystem")


# --------------------------------------------------------------- scorer


def _spec(**gt) -> FlawSpec:
    return FlawSpec(name="f", description="d", base="HEAD", ground_truth=gt)


def test_detection_hits_ground_truth_config_path() -> None:
    spec = _spec(files=["configs/base.yaml"], config_paths=["ppo.lr"],
                 subsystem="ppo")
    hit = _prop(target="ppo.lr", change={"config_path": "ppo.lr", "new_value": 1})
    miss = _prop(target="memory.write-gate", obs=[])
    assert score_arm(spec, [hit])["detection"] is True
    assert score_arm(spec, [miss])["detection"] is False


def test_detection_via_code_ref_and_subsystem() -> None:
    spec = _spec(files=["researcher/registry.py"], subsystem="researcher")
    p = _prop(obs=["code:researcher/registry.py@abc#L1-L9"])
    assert score_arm(spec, [p])["detection"] is True


def test_localization_precision_and_recall() -> None:
    spec = _spec(files=["configs/base.yaml"], config_paths=["ppo.lr"])
    on = _prop(target="ppo.lr", change={"config_path": "ppo.lr", "new_value": 1})
    off = _prop(target="rssm.free_nats",
                change={"config_path": "rssm.free_nats", "new_value": 1})
    row = score_arm(spec, [on, off])
    assert 0.0 < row["localization_precision"] <= 1.0
    assert row["localization_recall"] > 0.0


def test_clean_control_scores_false_positive() -> None:
    clean = FlawSpec(name="c", description="d", base="HEAD", ground_truth={},
                     clean=True)
    assert score_arm(clean, [])["false_positive"] is False
    assert score_arm(clean, [_prop(target="ppo.lr")])["false_positive"] is True


def test_summarize_reports_nulls_not_hidden() -> None:
    rows = [
        {"spec_clean": False, "detection": True, "localization_precision": 1.0,
         "false_positive": False},
        {"spec_clean": False, "detection": False, "localization_precision": 0.0,
         "false_positive": False},
        {"spec_clean": True, "detection": None, "localization_precision": None,
         "false_positive": True},
    ]
    s = summarize(rows)
    assert s["n_flaws"] == 2 and s["n_detected"] == 1
    assert s["detection_rate"] == 0.5
    assert s["clean_false_positive_rate"] == 1.0
