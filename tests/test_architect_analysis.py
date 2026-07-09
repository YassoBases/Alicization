"""Stage-D2: the deterministic analysis passes, run on THIS repo. Asserts
stable structural facts (relationships, presence, invariants) rather than a
brittle full-file golden — line counts and file lists drift as code changes,
the relationships this pins do not."""

from __future__ import annotations

import json
from pathlib import Path

from architect.analysis import (
    GENERATOR_TRIGGER_TAGS,
    AnalysisReport,
    analyze,
)

ROOT = Path(__file__).resolve().parent.parent


def test_report_is_json_serializable_and_roundtrips() -> None:
    report = analyze(ROOT)
    blob = report.to_json()
    data = json.loads(blob)
    assert data["repo_root"] and data["modules"] and data["import_graph"]
    assert isinstance(report, AnalysisReport)


def test_module_map_covers_known_files_with_positive_loc() -> None:
    modules = {m.path: m for m in analyze(ROOT).modules}
    for known in ("agent/core_rssm.py", "training/ppo.py", "evidence/store.py",
                  "architect/constitution.py", "proposals/schema.py"):
        assert known in modules, known
        assert modules[known].loc > 5


def test_import_graph_reflects_the_layering() -> None:
    graph = analyze(ROOT).import_graph
    # Stage-C wiring: proposals reads evidence; researcher reads proposals.
    assert "evidence" in graph["proposals"]
    assert "proposals" in graph["researcher"]
    # The Architect NEVER imports the modules under analysis (its core rule,
    # also enforced by test_architect_no_execution — surfaced here as data).
    # (It gains evidence/proposals imports once it emits + validates, D3-D5.)
    assert not ({"world", "training", "agent", "memory"} & set(graph["architect"]))


def test_invariants_capture_hard_rules_and_bans() -> None:
    inv = analyze(ROOT).invariants
    joined = " ".join(inv.hard_rules).upper()
    assert "GRADIENT ISOLATION" in joined
    assert "PROPOSALS ARE DATA" in joined
    assert "SCOPE RULE" in joined
    # Banned-import tables parsed from the structural tests.
    prop_scope = inv.banned_imports["proposals/review/researcher/evidence"]
    assert {"torch", "world", "training"} <= set(prop_scope)
    assert {"torch", "world", "training", "agent"} <= set(
        inv.banned_imports["architect"])


def test_symptom_linkage_points_at_emitting_modules() -> None:
    links = analyze(ROOT).tag_emitters
    # reward/rollout is logged in the trainers.
    assert any(m.startswith("training/") for m in links.get("reward/rollout", []))
    # Every reported emitter actually contains the tag literal.
    for tag, mods in links.items():
        for m in mods:
            assert tag in (ROOT / m).read_text(encoding="utf-8")


def test_symptom_linkage_scopes_to_given_tags() -> None:
    links = analyze(ROOT, anomalous_tags=("reward/rollout",)).tag_emitters
    assert set(links) <= {"reward/rollout"}
    # Default vocabulary is broader.
    assert len(analyze(ROOT).tag_emitters) >= 3
    assert "reward/rollout" in GENERATOR_TRIGGER_TAGS
