"""Stage-C evidence plane: EvidenceStore over a run dir, source-scoped views
(ledger vs logs_only strip), content-hashed immutable bundles, code refs,
and the repo snapshot. All offline (synthetic run dir; no training, no git
subprocess in the package)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evidence import (
    EvidenceBundle,
    EvidenceStore,
    EvidenceView,
    RepoSnapshot,
    evidence_from_run,
)
from ledger.competence import REPORT_SCHEMA_VERSION, CompetenceReport, RegionCompetence


def _make_run(tmp_path: Path, with_competence: bool = True) -> Path:
    """A minimal run dir: JSONL events, a config, an optional competence
    report, and a repo snapshot — but NO TB dir (scalars are injected via
    the monkeypatched reader in tests that need them)."""
    run_dir = tmp_path / "runs" / "20990101-000000"
    run_dir.mkdir(parents=True)
    with open(run_dir / "events-000000000.jsonl", "w", encoding="utf-8") as f:
        for t in range(20):
            f.write(json.dumps({"tick": t, "pos": [t % 8, t // 8],
                                "action": t % 4, "success": t % 2 == 0}) + "\n")
    (run_dir / "config.json").write_text(
        json.dumps({"rssm": {"free_nats": 1.0}, "seed": 7}), encoding="utf-8")
    (run_dir / "repo_snapshot.json").write_text(
        json.dumps({"git_sha": "abc1234", "dirty": False}), encoding="utf-8")
    if with_competence:
        (run_dir / "competence").mkdir()
        report = CompetenceReport(
            schema_version=REPORT_SCHEMA_VERSION, tick=19, run_id=run_dir.name,
            regions=[RegionCompetence(
                region=(0, 0), task="all", n_samples=100, wm_loss_ema=1.0,
                wm_loss_ratio=1.0, body_brier_ema=0.1, body_brier_ratio=1.0,
                forecaster_nmse_ema=float("nan"), reward_rate_ema=0.5,
                reward_ratio=1.0, learning_progress=0.01,
                adaptation_status="stable", replay_coverage=0.1)])
        (run_dir / "competence" / "report-000019.json").write_text(
            report.to_json(), encoding="utf-8")
    return run_dir


SCALARS = {
    "reward/rollout": ([0, 1, 2], [0.1, 0.2, 0.3]),
    "ledger/reliability_ece": ([0, 1, 2], [0.2, 0.2, 0.2]),
    "rssm/kl": ([0, 1, 2], [1.0, 1.0, 1.0]),
    "sleep/forecaster_nmse_k10": ([0], [0.8]),
}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvidenceStore:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {k: (list(s), list(v))
                                    for k, (s, v) in SCALARS.items()})
    return EvidenceStore(run_dir)


# ------------------------------------------------------------------ views


def test_ledger_view_keeps_everything(store: EvidenceStore) -> None:
    v = store.view("ledger")
    assert isinstance(v, EvidenceView) and v.source == "ledger"
    assert v.competence is not None
    assert np.allclose(v.series("ledger/reliability_ece"), [0.2, 0.2, 0.2])
    assert v.positions is not None and v.actions is not None
    assert v.config["seed"] == 7


def test_logs_only_view_strips_ledger_signals(store: EvidenceStore) -> None:
    v = store.view("logs_only")
    assert v.competence is None
    # Ledger-derived scalar tags are gone; raw task logs remain.
    assert v.series("ledger/reliability_ece").size == 0
    assert v.series("rssm/kl").size == 0
    assert v.series("sleep/forecaster_nmse_k10").size == 0
    assert np.allclose(v.series("reward/rollout"), [0.1, 0.2, 0.3])
    # positions/actions are raw logs: still present.
    assert v.actions is not None


def test_first_tag_picks_present_series(store: EvidenceStore) -> None:
    v = store.view("ledger")
    assert v.first_tag("nope/absent", "rssm/kl") == "rssm/kl"
    assert v.first_tag("only/missing", "still/missing") == "only/missing"


# ------------------------------------------------------------- bundle hash


def test_bundle_is_content_hashed_and_immutable(store: EvidenceStore) -> None:
    b = store.bundle("ledger")
    assert isinstance(b, EvidenceBundle)
    assert isinstance(b.content_hash, str) and len(b.content_hash) == 16
    # Frozen dataclass: attributes cannot be reassigned.
    with pytest.raises(Exception):
        b.content_hash = "0" * 16  # type: ignore[misc]
    # Deterministic: same store, same source -> same hash.
    assert store.bundle("ledger").content_hash == b.content_hash


def test_ledger_and_logs_bundles_differ(store: EvidenceStore) -> None:
    assert store.bundle("ledger").content_hash != store.bundle("logs_only").content_hash


def test_view_carries_its_bundle_hash(store: EvidenceStore) -> None:
    v = store.view("ledger")
    assert v.bundle_hash == store.bundle("ledger").content_hash


def test_bundle_hash_changes_with_evidence(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {"reward/rollout": ([0, 1], [0.1, 0.2])})
    h1 = EvidenceStore(run_dir).bundle("ledger").content_hash
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {"reward/rollout": ([0, 1], [0.1, 9.9])})
    h2 = EvidenceStore(run_dir).bundle("ledger").content_hash
    assert h1 != h2


# ------------------------------------------------------- repo snapshot + refs


def test_repo_snapshot_read_from_run_dir(store: EvidenceStore) -> None:
    assert store.repo_snapshot == RepoSnapshot(git_sha="abc1234", dirty=False)
    assert store.bundle("ledger").repo_snapshot.git_sha == "abc1234"


def test_repo_snapshot_injected_overrides_file(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr("evidence.store._read_tb_scalars", lambda _d: {})
    snap = RepoSnapshot(git_sha="deadbee", dirty=True)
    assert EvidenceStore(run_dir, repo_snapshot=snap).repo_snapshot == snap


def test_repo_snapshot_unknown_when_absent(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / "repo_snapshot.json").unlink()
    monkeypatch.setattr("evidence.store._read_tb_scalars", lambda _d: {})
    snap = EvidenceStore(run_dir).repo_snapshot
    assert snap.git_sha is None and snap.dirty is None


def test_code_ref_format(store: EvidenceStore) -> None:
    v = store.view("ledger")
    assert v.code_ref("researcher/eig.py", 10, 25) == "code:researcher/eig.py@abc1234#L10-L25"
    # Unknown sha degrades gracefully rather than lying.
    v.repo_snapshot = RepoSnapshot(None, None)
    assert v.code_ref("a.py", 1, 2) == "code:a.py@unknown#L1-L2"


def test_tb_ref_points_at_a_real_step(store: EvidenceStore) -> None:
    v = store.view("ledger")
    assert v.ref("reward/rollout") == "tb:reward/rollout@step=2"
    assert v.ref("absent/tag") == "tb:absent/tag"


# --------------------------------------------------------- compat shim


def test_evidence_from_run_shim(tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr("evidence.store._read_tb_scalars",
                        lambda _d: {"reward/rollout": ([0], [0.5])})
    # The old proposals.evidence entry point still works and returns a view.
    from proposals.evidence import Evidence, evidence_from_run as shim
    v = shim(run_dir, "ledger")
    assert isinstance(v, Evidence) and v.source == "ledger"
    assert evidence_from_run(run_dir, "logs_only").competence is None
