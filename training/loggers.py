"""Run logging: per-tick JSONL event log + TensorBoard scalar wrapper.

Schema for the JSONL log lives in docs/logging.md; keep them in sync.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

import numpy as np


def create_run_dir(root: str | Path = "runs") -> Path:
    """Create and return runs/<timestamp>/ (UTC, second resolution, unique)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = root / stamp
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"
    run_dir.mkdir()
    return run_dir


class JsonlRunLogger:
    """One JSON record per tick; files rotate every ``rotate_every`` ticks.

    Chunk files are named ``events-<chunk_index:09d>.jsonl`` where
    chunk_index = tick // rotate_every.
    """

    def __init__(self, run_dir: str | Path, rotate_every: int = 100_000) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_every = rotate_every
        self._chunk: int | None = None
        self._fh: IO[str] | None = None

    def _file_for(self, tick: int) -> IO[str]:
        chunk = tick // self.rotate_every
        if chunk != self._chunk:
            if self._fh is not None:
                self._fh.close()
            path = self.run_dir / f"events-{chunk:09d}.jsonl"
            self._fh = open(path, "a", encoding="utf-8", buffering=1 << 16)
            self._chunk = chunk
        assert self._fh is not None
        return self._fh

    def log_tick(
        self,
        tick: int,
        pos: tuple[int, int],
        action: int,
        success: bool,
        intero: np.ndarray,
        reward: float,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append one per-tick record (see docs/logging.md for the schema)."""
        record: dict[str, Any] = {
            "tick": tick,
            "pos": [int(pos[0]), int(pos[1])],
            "action": int(action),
            "success": bool(success),
            "intero": [float(v) for v in intero],
            "reward": float(reward),
        }
        if events:
            record["events"] = events
        self._file_for(tick).write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._chunk = None


def write_viz_state(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically dump the live-viewer state snapshot (viz/viewer.py --live).

    Written every ``run.viz_dump_every`` ticks by the trainer; the viewer
    polls the file's mtime. Atomic via temp-file + replace so a reader never
    sees a torn write.
    """
    import pickle

    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def read_viz_state(path: str | Path) -> dict[str, Any] | None:
    """Read a viz state dump; None if absent or torn mid-write."""
    import pickle

    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None


class TBLogger:
    """Thin TensorBoard SummaryWriter wrapper (scalars only for now)."""

    def __init__(self, log_dir: str | Path) -> None:
        # Deferred import: tensorboard is not needed by headless tools.
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def text(self, tag: str, text: str, step: int) -> None:
        """Text annotation at a step (e.g. lever events at their tick)."""
        self._writer.add_text(tag, text, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
