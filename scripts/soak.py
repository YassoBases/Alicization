"""Soak test: random agent for 1,000,000 ticks with full JSONL logging.

Asserts no exceptions and stable memory (RSS growth < 100 MB), prints ticks/sec.

Usage: python scripts/soak.py [--ticks N] [--seed S] [--run-root DIR] [--no-log]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import JsonlRunLogger, create_run_dir  # noqa: E402
from world.config import load_config  # noqa: E402
from world.engine import NUM_ACTIONS, World  # noqa: E402

RSS_LIMIT_BYTES = 100 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--no-log", action="store_true", help="skip JSONL logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    world = World(cfg)

    logger: JsonlRunLogger | None = None
    if not args.no_log:
        run_dir = create_run_dir(args.run_root)
        logger = JsonlRunLogger(run_dir)
        print(f"logging to {run_dir}")

    action_rng = np.random.default_rng(args.seed + 1)
    actions = action_rng.integers(0, NUM_ACTIONS, size=args.ticks)

    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    t_start = time.perf_counter()

    for i in range(args.ticks):
        action = int(actions[i])
        obs, infos = world.step([action])
        if logger is not None:
            info = infos[0]
            logger.log_tick(
                tick=info["tick"],
                pos=info["pos"],
                action=action,
                success=info["realized"]["success"],
                intero=obs[0]["intero"],
                reward=0.0,
                events=world.drain_events(),
            )
        else:
            world.drain_events()

    elapsed = time.perf_counter() - t_start
    if logger is not None:
        logger.close()

    rss_growth = proc.memory_info().rss - rss_start
    ticks_per_sec = args.ticks / elapsed
    print(f"ticks: {args.ticks}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"ticks/sec: {ticks_per_sec:,.0f}")
    print(f"rss growth: {rss_growth / 1024 / 1024:.1f} MB")

    assert rss_growth < RSS_LIMIT_BYTES, (
        f"RSS grew {rss_growth / 1024 / 1024:.1f} MB (limit 100 MB)"
    )
    print("soak OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
