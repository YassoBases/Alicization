"""Training entry point.

Usage:
    python train.py --config configs/smoke.yaml [--dry-run] [--resume PATH]
                    [--device cpu|cuda] [--allow-config-mismatch]
"""

from __future__ import annotations

import argparse
import sys

import yaml

from world.config import config_hash, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved config and exit"
    )
    parser.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    parser.add_argument(
        "--device", default=None, help="override config device (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="resume even if the checkpoint was written under a different config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device

    if args.dry_run:
        print(f"# resolved config: {args.config}")
        print(f"# config_hash: {config_hash(cfg)}")
        print(yaml.dump(cfg, sort_keys=False), end="")
        return 0

    # Deferred import: keeps --dry-run usable without torch installed.
    from training.loggers import create_run_dir
    from training.ppo import PPOTrainer

    run_dir = create_run_dir(cfg["run"]["run_dir"])
    print(f"run dir: {run_dir}")
    trainer = PPOTrainer(cfg, run_dir=run_dir)
    print(f"device: {trainer.device}")
    trainer.train(
        resume_from=args.resume, allow_config_mismatch=args.allow_config_mismatch
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
