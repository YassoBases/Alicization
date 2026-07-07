"""Pygame run viewer: live attach, JSONL replay, and mp4 recording.

Renders the grid (terrain/food/water/shelter/marks/agent), a day/night tint,
intero bars (energy/fatigue/memory_pressure), the current action, and the
tick counter.

Overlay toggles:
    [e] epistemic-map heat overlay (live mode; from the trainer's state dump)
    [m] stored-memory positions, dot size scaled by predicted reliability
    [d] mirror-divergence sparkline (last 256 ticks)

Modes:
    --live runs/<id>     attach to a running training process via the run
                         dir's viz_state.pkl (written every
                         run.viz_dump_every ticks; polled by mtime)
    --replay runs/<id>   scrub the per-tick JSONL log with LEFT/RIGHT
                         (SHIFT for +-100 ticks; SPACE toggles autoplay).
                         The static world layout is reconstructed from the
                         run's config.json (deterministic seeding) and food/
                         mark state is replayed from the logged events.
                         Epistemic/memory/divergence overlays come from the
                         live dump only and show "n/a" here.
    --record out.mp4     headless render of the replay (1 frame / 5 ticks)
                         via imageio-ffmpeg.

Keys: e/m/d overlays, LEFT/RIGHT scrub, SPACE pause/autoplay, q/ESC quit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.loggers import read_viz_state  # noqa: E402
from world.engine import ACTION_NAMES, World  # noqa: E402

CELL_MIN, BOARD_PX = 6, 640
HUD_W = 240
SPARK_H = 60

TERRAIN_COLORS = [(150, 150, 140), (120, 120, 105), (90, 90, 75)]
FOOD = (60, 170, 60)
WATER = (70, 110, 200)
SHELTER = (140, 95, 50)
MARK = (170, 80, 170)
AGENT = (220, 50, 50)
NIGHT_TINT = (30, 30, 80)
HUD_BG = (24, 24, 28)
TEXT = (230, 230, 230)
BAR_COLORS = {"energy": (80, 200, 80), "fatigue": (220, 160, 60),
              "memory_pressure": (120, 120, 220)}


# --------------------------------------------------------------- frame sources


class LiveSource:
    """Polls run_dir/viz_state.pkl (written by the trainer every N ticks)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "viz_state.pkl"
        self._mtime = 0.0
        self._state: dict[str, Any] | None = None

    def frame(self) -> dict[str, Any] | None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return self._state
        if mtime != self._mtime:
            state = read_viz_state(self.path)
            if state is not None:
                self._state, self._mtime = state, mtime
        return self._state


class ReplaySource:
    """Scrubbable frames from a run dir's JSONL log + reconstructed layout.

    The static layout (terrain/water/shelter + initial food) is rebuilt from
    config.json's deterministic seeding: env 0's episode-k world uses seed
    ``cfg.seed * 100_000 + k * num_envs`` (training/vecenv.py's counter, all
    envs rebuilding together at the shared episode boundary). Food and mark
    changes are replayed from logged events; a tick DECREASE between records
    marks an episode boundary (fresh world).
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        cfg_path = self.run_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} missing — replay needs the run's resolved config "
                "(written by trainers since the viz stage)"
            )
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = []
        for chunk in sorted(self.run_dir.glob("events-*.jsonl")):
            with open(chunk, encoding="utf-8") as f:
                self.records.extend(json.loads(line) for line in f if line.strip())
        if not self.records:
            raise FileNotFoundError(f"no events-*.jsonl records under {self.run_dir}")

        # Episode segmentation: tick decreases => new world for env 0.
        self.episode_starts = [0]
        for i in range(1, len(self.records)):
            if self.records[i]["tick"] < self.records[i - 1]["tick"]:
                self.episode_starts.append(i)

        self.index = 0
        self._episode = -1
        self._world: World | None = None
        self._food: np.ndarray | None = None
        self._mark: np.ndarray | None = None
        self._replayed_to = -1

    def __len__(self) -> int:
        return len(self.records)

    def _episode_of(self, index: int) -> int:
        ep = 0
        for k, start in enumerate(self.episode_starts):
            if index >= start:
                ep = k
        return ep

    def _load_episode(self, ep: int) -> None:
        cfg = copy.deepcopy(self.cfg)
        num_envs = int(cfg["ppo"]["num_envs"])
        cfg["seed"] = int(self.cfg["seed"]) * 100_000 + ep * num_envs
        self._world = World(cfg)
        self._food = self._world.food.copy()
        self._mark = self._world.mark.copy()
        self._episode = ep
        self._replayed_to = self.episode_starts[ep] - 1

    def _apply_events(self, record: dict[str, Any]) -> None:
        assert self._food is not None and self._mark is not None
        for ev in record.get("events", []):
            etype = ev.get("type")
            if etype == "food_consumed":
                x, y = ev["pos"]
                self._food[y, x] = False
            elif etype == "food_regrown":
                x, y = ev["pos"]
                self._food[y, x] = True
            elif etype == "food_relocated":
                if ev.get("had_food"):
                    sx, sy = ev["src"]
                    dx, dy = ev["pos"]
                    self._food[sy, sx] = False
                    self._food[dy, dx] = True
            elif etype == "mark_placed":
                x, y = ev["pos"]
                self._mark[y, x] = True
            elif etype == "mark_erased":
                x, y = ev["pos"]
                self._mark[y, x] = False

    def seek(self, index: int) -> None:
        self.index = int(np.clip(index, 0, len(self.records) - 1))

    def frame(self) -> dict[str, Any]:
        ep = self._episode_of(self.index)
        if ep != self._episode or self.index < self._replayed_to:
            self._load_episode(ep)
        for i in range(self._replayed_to + 1, self.index + 1):
            self._apply_events(self.records[i])
        self._replayed_to = self.index

        rec = self.records[self.index]
        world = self._world
        assert world is not None and self._food is not None and self._mark is not None
        intero = rec["intero"]
        # Time-of-day from the logged sin/cos (indices 3, 4).
        day_frac = (math.atan2(intero[3], intero[4]) / (2 * math.pi)) % 1.0
        return {
            "tick": rec["tick"],
            "world_size": world.size,
            "terrain": world.terrain,
            "food": self._food,
            "water": world.water,
            "shelter": world.shelter,
            "mark": self._mark,
            "agent_pos": tuple(rec["pos"]),
            "intero": intero,
            "action": rec["action"],
            "day_frac": day_frac,
            "night_start_frac": self.cfg["world"]["night_start_frac"],
            "epistemic_map": None,
            "memory": None,
            "divergence_tail": None,
        }


# -------------------------------------------------------------------- renderer


class Renderer:
    """Draws a viz-state dict onto a pygame surface. Import-safe: pygame is
    only touched at construction."""

    def __init__(self, world_size: int) -> None:
        import pygame

        self.pygame = pygame
        self.cell = max(CELL_MIN, BOARD_PX // world_size)
        self.board_px = self.cell * world_size
        self.size = (self.board_px + HUD_W, self.board_px + SPARK_H)
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 14)
        self.overlays = {"epistemic": False, "memory": False, "divergence": False}

    def toggle(self, name: str) -> None:
        self.overlays[name] = not self.overlays[name]

    def render(self, surface: Any, state: dict[str, Any]) -> None:
        pg = self.pygame
        surface.fill(HUD_BG)
        n = state["world_size"]
        cell = self.cell
        terrain = state["terrain"]
        food, water = state["food"], state["water"]
        shelter, mark = state["shelter"], state["mark"]

        for y in range(n):
            for x in range(n):
                color = TERRAIN_COLORS[int(terrain[y, x]) % len(TERRAIN_COLORS)]
                if water[y, x]:
                    color = WATER
                elif shelter[y, x]:
                    color = SHELTER
                elif food[y, x]:
                    color = FOOD
                elif mark[y, x]:
                    color = MARK
                surface.fill(color, (x * cell, y * cell, cell, cell))

        # Epistemic heat overlay.
        emap = state.get("epistemic_map")
        if self.overlays["epistemic"] and emap is not None:
            peak = float(np.max(emap)) or 1.0
            heat = pg.Surface((self.board_px, self.board_px), pg.SRCALPHA)
            for y in range(n):
                for x in range(n):
                    v = emap[y, x] / peak
                    if v > 0:
                        heat.fill((255, 40, 40, int(160 * v)),
                                  (x * cell, y * cell, cell, cell))
            surface.blit(heat, (0, 0))

        # Memory positions sized by predicted reliability.
        memory = state.get("memory")
        if self.overlays["memory"] and memory is not None and len(memory["positions"]):
            for (x, y), rel in zip(memory["positions"], memory["reliability"]):
                r = 2 + int(4 * float(rel))
                pg.draw.circle(surface, (250, 250, 80),
                               (int(x) * cell + cell // 2, int(y) * cell + cell // 2), r, 1)

        # Agent on top.
        ax, ay = state["agent_pos"]
        pg.draw.rect(surface, AGENT, (ax * cell, ay * cell, cell, cell))

        # Day/night tint.
        if state["day_frac"] >= state["night_start_frac"]:
            tint = pg.Surface((self.board_px, self.board_px), pg.SRCALPHA)
            tint.fill((*NIGHT_TINT, 110))
            surface.blit(tint, (0, 0))

        self._hud(surface, state)
        self._sparkline(surface, state)

    def _hud(self, surface: Any, state: dict[str, Any]) -> None:
        x0 = self.board_px + 12
        y = 12

        def line(text: str, color=TEXT) -> None:
            nonlocal y
            surface.blit(self.font.render(text, True, color), (x0, y))
            y += 20

        line(f"tick {state['tick']}")
        action = state.get("action")
        line(f"action {ACTION_NAMES[action] if action is not None else '--'}")
        phase = "NIGHT" if state["day_frac"] >= state["night_start_frac"] else "day"
        line(f"time {state['day_frac']:.2f} ({phase})")
        y += 8
        intero = state["intero"]
        for i, name in enumerate(("energy", "fatigue", "memory_pressure")):
            v = float(np.clip(intero[i], 0.0, 1.0))
            line(f"{name} {v:.2f}", BAR_COLORS[name])
            surface.fill((60, 60, 66), (x0, y - 4, 180, 8))
            surface.fill(BAR_COLORS[name], (x0, y - 4, int(180 * v), 8))
            y += 14
        y += 8
        for key, label in (("epistemic", "[e] epistemic"),
                           ("memory", "[m] memory"),
                           ("divergence", "[d] divergence")):
            have = {
                "epistemic": state.get("epistemic_map") is not None,
                "memory": state.get("memory") is not None,
                "divergence": state.get("divergence_tail") is not None,
            }[key]
            status = ("ON" if self.overlays[key] else "off") if have else "n/a"
            line(f"{label}: {status}")

    def _sparkline(self, surface: Any, state: dict[str, Any]) -> None:
        if not self.overlays["divergence"]:
            return
        tail = state.get("divergence_tail")
        if tail is None or len(tail) < 2:
            return
        pg = self.pygame
        y0 = self.board_px + 8
        h = SPARK_H - 16
        tail = np.asarray(tail, dtype=float)
        peak = max(float(tail.max()), 1e-6)
        pts = [
            (int(i * self.board_px / (len(tail) - 1)),
             y0 + h - int(h * min(v / peak, 1.0)))
            for i, v in enumerate(tail)
        ]
        pg.draw.lines(surface, (250, 120, 120), False, pts, 1)
        surface.blit(self.font.render(
            f"divergence (max {peak:.1f})", True, (250, 120, 120)), (4, y0 + h + 2))


# ------------------------------------------------------------------ main loops


def record_mp4(source: ReplaySource, out_path: str | Path,
               every: int = 5, fps: int = 20) -> Path:
    """Headless render of a replay to mp4 (1 frame per ``every`` ticks)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import imageio.v2 as imageio
    import pygame

    pygame.init()
    source.seek(0)
    first = source.frame()
    renderer = Renderer(first["world_size"])
    surface = pygame.Surface(renderer.size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                macro_block_size=8)
    try:
        for i in range(0, len(source), every):
            source.seek(i)
            renderer.render(surface, source.frame())
            frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
            writer.append_data(frame)
    finally:
        writer.close()
        pygame.quit()
    return out_path


def run_viewer(source: LiveSource | ReplaySource, fps: int = 30) -> None:
    import pygame

    pygame.init()
    state = source.frame()
    if state is None:
        print("no state available yet (is the training process writing viz_state.pkl?)")
        return
    renderer = Renderer(state["world_size"])
    screen = pygame.display.set_mode(renderer.size)
    pygame.display.set_caption("reflective-cartographer viewer")
    clock = pygame.time.Clock()
    autoplay = isinstance(source, LiveSource)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                step = 100 if mods & pygame.KMOD_SHIFT else 1
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_e:
                    renderer.toggle("epistemic")
                elif event.key == pygame.K_m:
                    renderer.toggle("memory")
                elif event.key == pygame.K_d:
                    renderer.toggle("divergence")
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif event.key == pygame.K_LEFT and isinstance(source, ReplaySource):
                    source.seek(source.index - step)
                elif event.key == pygame.K_RIGHT and isinstance(source, ReplaySource):
                    source.seek(source.index + step)
        if autoplay and isinstance(source, ReplaySource):
            source.seek(source.index + 1)
        state = source.frame()
        if state is not None:
            renderer.render(screen, state)
            pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", metavar="RUN_DIR")
    mode.add_argument("--replay", metavar="RUN_DIR")
    parser.add_argument("--record", metavar="OUT_MP4", default=None,
                        help="headless mp4 render (replay mode only)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.record:
        if not args.replay:
            parser.error("--record requires --replay")
        out = record_mp4(ReplaySource(args.replay), args.record)
        print(f"wrote {out}")
        return 0
    source = LiveSource(args.live) if args.live else ReplaySource(args.replay)
    if isinstance(source, LiveSource):
        # Wait briefly for the first dump when attaching early.
        for _ in range(50):
            if source.frame() is not None:
                break
            time.sleep(0.2)
    run_viewer(source, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
