"""Config loading and resolution.

YAML configs live in configs/. A config may declare ``inherit: <relative path>``;
the parent is loaded first and the child is deep-merged on top. The resolved config
is a plain nested dict; ``config_hash`` produces a stable digest of it for
checkpoint compatibility checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged onto ``base``, recursing into dicts."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving the ``inherit`` chain relative to the file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    parent_rel = cfg.pop("inherit", None)
    if parent_rel is not None:
        parent = load_config(path.parent / parent_rel)
        cfg = deep_merge(parent, cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """SHA-256 hex digest of the resolved config (key-order independent)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
