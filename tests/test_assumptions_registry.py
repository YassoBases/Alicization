"""Phase-1 gate: the Assumption Registry validates (schema + every citation
resolves to a real file/anchor), covers the required components, and renders."""

from __future__ import annotations

import pytest

from lab.render import (
    MATURITIES,
    STATUSES,
    iter_citations,
    load_registry,
    render,
    resolve_citation,
    validate,
)

# The minimum coverage the directive requires.
REQUIRED_IDS = {
    "recurrent-core", "latent-representation", "episodic-memory",
    "memory-reliability-weighting", "consolidation-imagination",
    "wake-sleep-scheduling", "drives-intrinsic", "planning-arbiter",
    "ledger-body-model", "ledger-attribution", "ledger-forecaster",
    "ledger-mirror", "ledger-competence", "selfq-unified",
    "gradient-isolation", "researcher-monitors", "proposal-generators-dual-source",
    "architect-instrument", "training-procedure",
    # untested placeholders to be filled by their stages:
    "hydration-degradation", "multi-agent-attribution", "causal-self-theory",
}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_registry_is_valid(registry) -> None:
    """Schema + every citation resolves. This is the Gate-P1 assertion."""
    problems = validate(registry)
    assert not problems, "registry problems:\n  " + "\n  ".join(problems)


def test_every_citation_resolves_individually(registry) -> None:
    # Redundant with validate() but pinpoints a broken citation directly.
    for eid, field, cite in iter_citations(registry):
        assert resolve_citation(cite) is None, f"{eid}.{field}: {cite}"


def test_required_components_are_covered(registry) -> None:
    ids = {e["id"] for e in registry["assumptions"]}
    missing = REQUIRED_IDS - ids
    assert not missing, f"registry missing required components: {sorted(missing)}"


def test_enums_and_confidence_bounds(registry) -> None:
    for e in registry["assumptions"]:
        assert e["status"] in STATUSES
        assert 0.0 <= e["confidence"] <= 1.0
        for c in e.get("replacement_candidates") or []:
            assert c["maturity"] in MATURITIES


def test_unsupported_entries_name_a_removal_or_redesign(registry) -> None:
    """An 'unsupported' component must offer at least one replacement/removal
    candidate — a dead assumption is an actionable one."""
    for e in registry["assumptions"]:
        if e["status"] == "unsupported":
            assert e.get("replacement_candidates"), \
                f"{e['id']} is unsupported but names no replacement/removal"


def test_renders_without_error(registry) -> None:
    md = render(registry)
    assert md.startswith("# Assumption Registry")
    assert "memory-reliability-weighting" in md and "unsupported" in md
