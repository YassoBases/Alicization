"""Stage-8b acceptance: on the stage-8a lever fixture, the ranked agenda's
top items point at the lever-affected capability, the output artifacts are
written, and ranking is deterministic on the frozen store.

Reuses the runs_8a fixture (capability_shift on MOVE_E): replays the
registry monitors to reach the contradicted state, generates questions,
ranks with v1, and writes agenda_<tick>.json + research_agenda.md.

Usage: python scripts/verify_agenda.py --run-dir runs_8a/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researcher.agenda import rank_v1, write_agenda  # noqa: E402
from researcher.questions import generate_questions  # noqa: E402
from researcher.registry import (  # noqa: E402
    HypothesisRegistry,
    QueryEngine,
    build_default_hypotheses,
)

MOVE_E = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=24576)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    # Reach the post-lever registry state (same monitor geometry as the
    # stage-8a acceptance; see verify_registry.py for the rationale).
    registry = HypothesisRegistry(run_dir)
    for h in build_default_hypotheses(world_size=32, num_actions=9):
        h.monitor["window"] = 1000
        if "arm_after" in h.monitor:
            h.monitor["arm_after"] = 1500
        registry.add(h)
    engine = QueryEngine(run_dir)
    max_tick = args.steps // 4
    for now_tick in range(250, max_tick + 1, 250):
        registry.check_all(engine, now_tick)
    target = registry.hypotheses[f"hyp-capability-success-{MOVE_E}"]
    assert target.status == "contradicted", (
        f"fixture precondition failed: MOVE_E hypothesis is {target.status}"
    )

    questions = generate_questions(run_dir, registry)
    print(f"{len(questions)} questions generated:")
    for q in questions:
        print(f"  [{q.type}] {q.statement[:90]}")

    items = rank_v1(questions, proposals=[], competence=None)
    items_again = rank_v1(questions, proposals=[], competence=None)
    assert [i.id for i in items] == [i.id for i in items_again], (
        "ranking is not deterministic on a frozen store"
    )

    print("\ntop 5:")
    for rank, item in enumerate(items[:5], start=1):
        print(f"  {rank}. {item.score:.3f} [{item.kind}] {item.statement[:80]}")

    # The lever's fingerprint must be at the top: the MOVE_E contradiction
    # (assumption_violation, uncertainty 1.0) and/or the MOVE_E capability
    # gap must appear in the top 3 items.
    top_refs = " ".join(
        i.ref + " " + json.dumps(i.experiment) for i in items[:3])
    assert (f"q-violation-hyp-capability-success-{MOVE_E}" in top_refs
            or f"q-capability-{MOVE_E}" in top_refs), (
        f"lever-affected capability not in the top 3: "
        f"{[i.ref for i in items[:3]]}"
    )
    # And its hypothesis link must name the contradicted hypothesis.
    linked = [i for i in items[:3]
              if f"hyp-capability-success-{MOVE_E}" in i.hypothesis_links]
    assert linked, "top items carry no link to the contradicted hypothesis"

    json_path, md_path = write_agenda(items, run_dir, tick=max_tick)
    assert json_path.exists() and md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "score" in md and "would move" in md
    print(f"\nagenda written: {json_path.name}, {md_path.name}")
    print("verify_agenda OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
