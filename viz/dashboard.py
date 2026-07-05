"""Streamlit dashboard (stub): Ledger calibration + training curves across runs.

Usage: streamlit run viz/dashboard.py
"""

from __future__ import annotations

from pathlib import Path


def render_dashboard(runs_root: str | Path = "runs") -> None:
    """Build the streamlit page for all runs under ``runs_root``."""
    raise NotImplementedError


if __name__ == "__main__":
    render_dashboard()
