"""The internal researcher: a metacognitive layer above the Ledger and the
proposal subsystem. Its ONLY output is a prioritized research agenda — what
the agent does not understand, which experiment would reduce that
uncertainty most efficiently, which capability is weakest, which standing
assumption recent data has contradicted. It never acts.

All proposal-layer structural rules apply verbatim (data-not-code, confined
writes, no world/training imports — tests/test_proposals_no_execution.py
covers this package), plus the SCOPE RULE (CLAUDE.md): hypotheses and
questions may be about the agent-in-the-world and its own models only;
statements about the reviewer, reset schedule, or training infrastructure
are rejected at the schema level.
"""
