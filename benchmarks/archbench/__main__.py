"""`python -m benchmarks.archbench` — run the flaw battery, score both arms,
write benchmarks/results/<date>/archbench.md.

Experimenter-invoked; runs real smoke training inside disposable worktrees.
Offline architect arm by default (no API key needed to complete).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from benchmarks.archbench.report import render
from benchmarks.archbench.runner import run_suite
from benchmarks.archbench.spec import default_specs_dir, load_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default=None, help="flaw-spec dir")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these spec names")
    parser.add_argument("--steps", type=int, default=4096,
                        help="smoke-training env steps per worktree")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-worktree training timeout (s)")
    parser.add_argument("--online", action="store_true",
                        help="let the architect arm draft (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    specs = load_specs(Path(args.specs) if args.specs else default_specs_dir())
    if args.only:
        specs = [s for s in specs if s.name in args.only]
    if not specs:
        parser.error("no specs to run")

    date = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out or f"benchmarks/results/{date}")
    cfg = {"architect": {"offline": not args.online}}
    results = run_suite(specs, out_dir, cfg=cfg, steps=args.steps,
                        timeout=args.timeout)
    report = render(results, out_dir, scale=f"{args.steps} steps",
                    online=args.online)
    print(f"report: {report}")
    print("Write ANALYSIS by reading the per-spec JSON — not auto-generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
