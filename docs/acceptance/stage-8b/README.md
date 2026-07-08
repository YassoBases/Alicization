# Stage-8b acceptance: questions + ranked agenda v1

**Criteria:** (1) agenda generation is deterministic on a frozen log
store; (2) on the stage-8a lever fixture, the top items point at the
lever-affected capability; (3) the noisy-TV guard ranks
high-uncertainty/zero-progress regions LOW (unit-tested in
tests/test_agenda.py); (4) agenda_<tick>.json + research_agenda.md are
written with score decompositions and hypothesis links.

**Fixture:** the stage-8a run (capability_shift on MOVE_E @4501, frozen
policy). Replay: registry monitors reach `contradicted` on
hyp-capability-success-2, then questions are generated and ranked with
v1 (`score = value x tractability x novelty / cost`).

**Result** (`verify_agenda_replay.txt`, agenda artifacts alongside):

- 7 questions generated across all four types (world_uncertainty x3,
  capability_gap x2, assumption_violation x2)
- top 3 items: the MOVE_W capability gap (7.1 sd — the genuine
  downstream effect of the lever, see stage-8a) and the two
  assumption_violation questions for the contradicted MOVE_E / MOVE_W
  stability hypotheses, each carrying its hypothesis link
- ranking identical across two passes on the same store (and unit-tested
  with proposals in the same agenda)

Also fixed here: the capability_gap shift statistic used a bare epsilon
denominator and reported "shifted 357308584.7 sd" on an all-success
baseline window — same std_floor treatment as the registry (0.05).

Reproduce: `python scripts/verify_agenda.py --run-dir runs_8a/<id>`.
