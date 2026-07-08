"""Proposal layer: DATA, NEVER CODE (CLAUDE.md Hard rules).

Everything in this package reads diagnostics and emits JSON records for a
human to review. Nothing here executes, imports execution machinery, or
writes outside runs/<id>/proposals/ — enforced by
tests/test_proposals_no_execution.py.
"""
