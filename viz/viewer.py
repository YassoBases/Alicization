"""Live run viewer (stub): renders the grid + agent from a run's JSONL log.

Usage: python -m viz.viewer --run runs/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path


def view_run(run_dir: str | Path, follow: bool = True) -> None:
    """Render the run at ``run_dir``; if ``follow``, tail the log as it grows."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory (e.g. runs/latest)")
    args = parser.parse_args()
    view_run(args.run)


if __name__ == "__main__":
    main()
