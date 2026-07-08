"""Human review queue for the proposal layer.

DATA, NEVER CODE (CLAUDE.md Hard rules): this package renders proposals for
a human, records decisions, and emits experiment TICKETS — markdown stubs a
human executes by hand. Nothing here runs anything; the structural tests in
tests/test_proposals_no_execution.py cover this package too.
"""
