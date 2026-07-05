"""Config-driven experiment levers, invisible to the agent.

Nothing in agent/ or ledger/ may import this module (grep-able rule, enforced
by tests). All schedules, rates, and windows come exclusively from the
``world.levers`` config section; timing never comes from code. Every applied
perturbation is logged through World.log_event with a ground-truth
cause={self|world} label — evaluation-only; it never enters any observation
or loss.

Config schema (all keys optional)::

    levers:
      capability_shift:            # reversible transition edits, one per entry
        - action: 0                # target action id
          start: 1000              # first tick the shift is active
          end: 5000                # first tick it is inactive again (null = never)
          fail_prob: 0.5           # action attempt fails with this probability
          energy_mult: 1.0         # energy cost multiplier while active
          effect_delta: [0, 1]     # optional: override the move (dx, dy) while
                                   # active (e.g. two shift entries with swapped
                                   # effect_delta implement an "effect-swap")
      ghost_events:
        rate: 0.01                 # per-tick probability of one ghost event
        kinds: [push, consume_food]
      region_volatility:
        regions:
          - rect: [x0, y0, x1, y1] # inclusive bounds
            interval: 500          # relocate food patches in rect every N ticks
      seasonal_shift:
        interval: 10000            # migrate the whole food distribution every N
      exogenous_reset:
        ticks: [50000]             # trainer-facing markers; world just logs them
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from world.engine import World

_PUSH_DELTAS = ((0, -1), (0, 1), (1, 0), (-1, 0))


class LeverEngine:
    """Applies scheduled perturbations inside World.step and logs ground truth.

    Stateless apart from config: every decision derives from the world tick and
    the world-owned RNG, so World.snapshot/restore covers lever behavior.
    """

    def __init__(self, cfg: dict[str, Any] | None) -> None:
        """``cfg`` is the ``world.levers`` config section (or None: all levers off)."""
        cfg = cfg or {}
        self.shifts: list[dict[str, Any]] = [
            dict(s) for s in cfg.get("capability_shift") or []
        ]
        ghosts = cfg.get("ghost_events") or {}
        self.ghost_rate: float = float(ghosts.get("rate", 0.0))
        self.ghost_kinds: list[str] = list(ghosts.get("kinds", ["push", "consume_food"]))
        volatility = cfg.get("region_volatility") or {}
        self.regions: list[dict[str, Any]] = [
            dict(r) for r in volatility.get("regions") or []
        ]
        seasonal = cfg.get("seasonal_shift") or {}
        self.season_interval: int = int(seasonal.get("interval", 0))
        resets = cfg.get("exogenous_reset") or {}
        self.reset_ticks: set[int] = set(resets.get("ticks") or [])

    # ------------------------------------------------------------- capability

    def capability(
        self, action: int, tick: int
    ) -> tuple[float, float, tuple[int, int] | None]:
        """Effective (fail_prob, energy_mult, effect_delta) for ``action`` at
        ``tick``. ``effect_delta`` overrides the action's normal (dx, dy) move
        delta while active (None: no override, i.e. the last active shift for
        this action that specifies one wins; multiple simultaneous overrides
        for the same action are not resolved beyond "last in list wins")."""
        fail_prob, energy_mult, effect_delta = 0.0, 1.0, None
        for s in self.shifts:
            if s["action"] != action or tick < s["start"]:
                continue
            if s.get("end") is not None and tick >= s["end"]:
                continue
            fail_prob = max(fail_prob, float(s.get("fail_prob", 0.0)))
            energy_mult *= float(s.get("energy_mult", 1.0))
            if s.get("effect_delta") is not None:
                effect_delta = tuple(s["effect_delta"])
        return fail_prob, energy_mult, effect_delta

    # -------------------------------------------------------------- post step

    def post_step(self, world: "World") -> None:
        """Apply ghost events / volatility / seasonal shifts / reset markers."""
        tick = world.tick
        for s in self.shifts:
            if tick == s["start"]:
                world.log_event(
                    "capability_shift_start", "world", action=s["action"],
                    fail_prob=s.get("fail_prob", 0.0),
                    energy_mult=s.get("energy_mult", 1.0),
                )
            if s.get("end") is not None and tick == s["end"]:
                world.log_event("capability_shift_end", "world", action=s["action"])

        if self.ghost_rate > 0.0 and world.rng.random() < self.ghost_rate:
            self._ghost_event(world)

        for region in self.regions:
            if tick > 0 and tick % int(region["interval"]) == 0:
                self._relocate_patches(world, tuple(region["rect"]), "food_relocated")

        if self.season_interval and tick > 0 and tick % self.season_interval == 0:
            n = world.size
            moved = self._relocate_patches(world, (0, 0, n - 1, n - 1), "food_relocated")
            world.log_event("seasonal_shift", "world", patches_moved=moved)

        if tick in self.reset_ticks:
            world.log_event("exogenous_reset", "world")

    def _ghost_event(self, world: "World") -> None:
        """One action-mimicking perturbation NOT caused by the agent."""
        agent_idx = int(world.rng.integers(len(world.agents)))
        kind = self.ghost_kinds[int(world.rng.integers(len(self.ghost_kinds)))]
        if kind == "push":
            dx, dy = _PUSH_DELTAS[int(world.rng.integers(4))]
            world.move_agent(agent_idx, dx, dy, "world")  # logs cause=world
        elif kind == "consume_food":
            a = world.agents[agent_idx]
            candidates = [
                (a.x + dx, a.y + dy)
                for dx, dy in _PUSH_DELTAS
                if 0 <= a.x + dx < world.size
                and 0 <= a.y + dy < world.size
                and world.food[a.y + dy, a.x + dx]
            ]
            if candidates:
                x, y = candidates[int(world.rng.integers(len(candidates)))]
                world.consume_food(x, y, "world")  # logs cause=world
        else:
            raise ValueError(f"unknown ghost event kind: {kind!r}")

    def _relocate_patches(
        self, world: "World", rect: tuple[int, int, int, int], event_type: str
    ) -> int:
        """Move every food patch inside ``rect`` to a fresh cell inside it."""
        x0, y0, x1, y1 = rect
        occupied = {(p["x"], p["y"]) for p in world.patches}
        moved = 0
        for patch in world.patches:
            if not (x0 <= patch["x"] <= x1 and y0 <= patch["y"] <= y1):
                continue
            dest = self._pick_free_cell(world, rect, occupied)
            if dest is None:
                continue
            src = (patch["x"], patch["y"])
            had_food = bool(world.food[src[1], src[0]])
            if had_food:
                world.set_food(src[0], src[1], False)
                world.set_food(dest[0], dest[1], True)
            occupied.discard(src)
            occupied.add(dest)
            patch["x"], patch["y"] = dest
            world.log_event(
                event_type, "world", pos=list(dest), src=list(src), had_food=had_food
            )
            moved += 1
        return moved

    @staticmethod
    def _pick_free_cell(
        world: "World",
        rect: tuple[int, int, int, int],
        occupied: set[tuple[int, int]],
        attempts: int = 100,
    ) -> tuple[int, int] | None:
        """Random cell in rect avoiding water, shelter, and other patches."""
        x0, y0, x1, y1 = rect
        for _ in range(attempts):
            x = int(world.rng.integers(x0, x1 + 1))
            y = int(world.rng.integers(y0, y1 + 1))
            if (x, y) in occupied or world.water[y, x] or world.shelter[y, x]:
                continue
            return x, y
        return None

    # ---------------------------------------------------------------- state

    def get_state(self) -> dict[str, Any]:
        """Mutable lever state for World.snapshot (empty: levers are stateless)."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`get_state`."""
        del state
