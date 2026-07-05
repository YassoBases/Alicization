"""JSONL run logger: schema fields, rotation, and event pass-through."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.loggers import JsonlRunLogger

INTERO = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)


def test_rotation_and_schema(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path, rotate_every=100)
    for tick in range(1, 251):
        events = [{"tick": tick, "type": "food_regrown", "cause": "world"}] if tick == 7 else None
        logger.log_tick(
            tick=tick, pos=(3, 4), action=8, success=True,
            intero=INTERO, reward=0.0, events=events,
        )
    logger.close()

    files = sorted(tmp_path.glob("events-*.jsonl"))
    assert [f.name for f in files] == [
        "events-000000000.jsonl", "events-000000001.jsonl", "events-000000002.jsonl",
    ]

    records = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    assert len(records) == 250
    assert [r["tick"] for r in records] == list(range(1, 251))

    first = records[0]
    assert first["pos"] == [3, 4]
    assert first["action"] == 8
    assert first["success"] is True
    assert first["intero"] == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert first["reward"] == 0.0
    assert "events" not in first  # omitted when empty

    with_events = [r for r in records if "events" in r]
    assert len(with_events) == 1
    assert with_events[0]["events"][0]["cause"] == "world"
