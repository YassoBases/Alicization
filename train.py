"""Training entry point.

Usage:
    python train.py --config configs/smoke.yaml [--dry-run] [--resume PATH]
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
        "--allow-config-mismatch",
        action="store_true",
        help="resume even if the checkpoint was written under a different config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.dry_run:
        print(f"# resolved config: {args.config}")
        print(f"# config_hash: {config_hash(cfg)}")
        print(yaml.dump(cfg, sort_keys=False), end="")
        return 0

    # Deferred import: keeps --dry-run usable without torch installed.
    from training.checkpoints import load_checkpoint
    from training.ppo import PPOTrainer

    if args.resume is not None:
        # Validate the checkpoint against the active config up front, without a
        # model: PPOTrainer re-loads it into its modules inside train().
        ckpt = load_checkpoint(
            args.resume,
            cfg=cfg,
            restore_rng=False,
            allow_config_mismatch=args.allow_config_mismatch,
        )
        print(f"resuming from {args.resume} at step {ckpt.step}")

    trainer = PPOTrainer(cfg)
    trainer.train(resume_from=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
