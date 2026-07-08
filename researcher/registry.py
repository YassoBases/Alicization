"""Hypothesis registry + contradiction monitors.

A hypothesis is a machine-checkable predicate over LOGGED metrics — a
statement template with params, a monitor spec (metric query, statistical
test, threshold, window), and a status that transitions
supported -> weakening -> contradicted as consecutive checks fail (one
clean check recovers weakening -> supported; contradicted is sticky until a
human retires or re-seeds it). Every transition records the specific
evidence (test statistic, windows) that triggered it.

Monitors run on the sleep-phase cadence over the run's log store (JSONL
event log + TB scalars), driven by the experiment harness — this package
never imports world/ or training/ (structural tests). A contradiction
emits a structured event to runs/<id>/researcher/contradiction_events.jsonl
for the agenda (P9.2) and the dashboard.

SCOPE RULE (CLAUDE.md, schema-enforced here): statements may concern the
agent-in-the-world and its own models only. Statements referencing the
reviewer, reset schedules, or training infrastructure raise at validation
(adversarially tested).

Metric-query mini-language (data-only, resolved by QueryEngine):
    jsonl:regrowth_intervals:region=R,C     per-region food regrowth gaps
    jsonl:action_success:action=A           per-action success indicators
    jsonl:action_denergy:action=A           per-action energy deltas
    tb:<scalar tag>                         any TensorBoard series
Statistical tests:
    ks_2sample   KS statistic between consecutive-window samples > threshold
    mean_shift   |mean(now) - mean(prev)| / (std(prev)+eps) > threshold
    band         fraction of window outside [lo, hi] > threshold
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HYPOTHESIS_SCHEMA_VERSION = 1

SCOPES = ("world", "self_capability", "self_memory", "self_model")
STATUSES = ("supported", "weakening", "contradicted", "retired")

# SCOPE RULE: any of these in a statement/params rejects the hypothesis.
_FORBIDDEN_PATTERNS = (
    "reviewer", "review process", "reset schedule", "resets", "kill switch",
    "shutdown", "training infrastructure", "training process", "checkpoint",
    "experimenter", "human", "lever", "operator",
)


@dataclass
class Hypothesis:
    schema_version: int
    id: str
    statement_template: str
    params: dict[str, Any]
    scope: str
    monitor: dict[str, Any]     # {metric_query, statistical_test, threshold,
    #                              window, ...test-specific keys}
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_tick: int = 0
    status: str = "supported"
    last_checked: int = -1
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def statement(self) -> str:
        return self.statement_template.format(**self.params)

    def validate(self) -> None:
        if self.schema_version != HYPOTHESIS_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version}")
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        text = (self.statement_template + " " + json.dumps(self.params)).lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in text:
                raise ValueError(
                    f"SCOPE RULE: hypothesis references {pattern!r} — "
                    f"statements about the reviewer, reset schedule, or "
                    f"training infrastructure are rejected (CLAUDE.md)"
                )
        for key in ("metric_query", "statistical_test", "threshold", "window"):
            if key not in self.monitor:
                raise ValueError(f"monitor missing {key!r}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Hypothesis":
        h = Hypothesis(**json.loads(text))
        h.validate()
        return h


# ---------------------------------------------------------------- log store


class QueryEngine:
    """Resolves metric queries against one run dir's logs. Read-only."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self._jsonl_cache: list[dict[str, Any]] | None = None
        self._tb_cache: dict[str, tuple[list[int], list[float]]] | None = None

    def _jsonl(self) -> list[dict[str, Any]]:
        if self._jsonl_cache is None:
            records = []
            for chunk in sorted(self.run_dir.glob("events-*.jsonl")):
                with open(chunk, encoding="utf-8") as f:
                    records.extend(json.loads(line) for line in f)
            self._jsonl_cache = records
        return self._jsonl_cache

    def _tb(self) -> dict[str, tuple[list[int], list[float]]]:
        if self._tb_cache is None:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            out: dict[str, tuple[list[int], list[float]]] = {}
            tb_dir = self.run_dir / "tb"
            if tb_dir.exists():
                acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
                acc.Reload()
                for tag in acc.Tags().get("scalars", []):
                    events = acc.Scalars(tag)
                    out[tag] = ([e.step for e in events], [e.value for e in events])
            self._tb_cache = out
        return self._tb_cache

    def invalidate(self) -> None:
        self._jsonl_cache = None
        self._tb_cache = None

    # (samples, ticks) pairs for a query.
    def samples(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        if query.startswith("tb:"):
            steps, values = self._tb().get(query[3:], ([], []))
            return np.asarray(values, dtype=float), np.asarray(steps, dtype=float)

        match = re.match(r"jsonl:(\w+):(.*)$", query)
        if not match:
            raise ValueError(f"bad metric query {query!r}")
        kind, arg = match.group(1), match.group(2)
        records = self._jsonl()

        if kind == "regrowth_intervals":
            r, c = (int(x) for x in arg.split("=")[1].split(","))
            last_regrow: float | None = None
            intervals, ticks = [], []
            for rec in records:
                for ev in rec.get("events", []):
                    if ev.get("type") != "food_regrown":
                        continue
                    x, y = ev["pos"]
                    if y // 8 == r and x // 8 == c:
                        if last_regrow is not None:
                            intervals.append(rec["tick"] - last_regrow)
                            ticks.append(rec["tick"])
                        last_regrow = rec["tick"]
            return np.asarray(intervals, dtype=float), np.asarray(ticks, dtype=float)

        if kind in ("action_success", "action_denergy"):
            action = int(arg.split("=")[1])
            values, ticks = [], []
            for rec in records:
                if rec["action"] != action:
                    continue
                values.append(float(rec["success"]) if kind == "action_success"
                              else float(rec.get("denergy", 0.0)))
                ticks.append(rec["tick"])
            return np.asarray(values, dtype=float), np.asarray(ticks, dtype=float)

        raise ValueError(f"unknown jsonl query kind {kind!r}")


# ------------------------------------------------------------------ checks


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS D statistic (numpy-only)."""
    both = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), both, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), both, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def run_check(h: Hypothesis, engine: QueryEngine, now_tick: int) -> dict[str, Any]:
    """One monitor evaluation. Returns {violated, statistic, detail}."""
    m = h.monitor
    values, ticks = engine.samples(m["metric_query"])
    window = float(m["window"])
    test = m["statistical_test"]
    # Both windows are bounded ABOVE by now_tick: in a live run no future
    # data exists, but a post-hoc replay's store holds the whole run, and an
    # unbounded "now" window read the future — checks before a lever saw the
    # lever's effect and "detected" it early (observed as impossible 10-sd
    # pre-lever shifts on a frozen policy).
    now_mask = (ticks >= now_tick - window) & (ticks < now_tick)
    prev_mask = (ticks >= now_tick - 2 * window) & (ticks < now_tick - window)
    now_vals, prev_vals = values[now_mask], values[prev_mask]
    min_n = int(m.get("min_samples", 5))

    if test == "ks_2sample":
        if len(now_vals) < min_n or len(prev_vals) < min_n:
            return {"violated": False, "statistic": float("nan"),
                    "detail": f"insufficient samples ({len(prev_vals)}/{len(now_vals)})"}
        d = _ks_statistic(prev_vals, now_vals)
        return {"violated": d > m["threshold"], "statistic": d,
                "detail": f"KS D={d:.3f} vs {m['threshold']} "
                          f"(n={len(prev_vals)}/{len(now_vals)})"}
    if test == "mean_shift":
        if len(now_vals) < min_n or len(prev_vals) < min_n:
            return {"violated": False, "statistic": float("nan"),
                    "detail": "insufficient samples"}
        # std floor: a constant baseline window (e.g. all-success) has
        # std ~ 0, and one natural stray sample would otherwise divide to an
        # astronomical "shift". The floor sets the minimum meaningful unit
        # of change (0.05 suits rate-like data; override per hypothesis).
        floor = float(m.get("std_floor", 0.05))
        denom = max(float(prev_vals.std()), floor)
        shift = abs(now_vals.mean() - prev_vals.mean()) / denom
        return {"violated": shift > m["threshold"], "statistic": float(shift),
                "detail": f"mean shift {shift:.2f} sd vs {m['threshold']}"}
    if test == "band":
        if len(now_vals) < min_n:
            return {"violated": False, "statistic": float("nan"),
                    "detail": "insufficient samples"}
        lo, hi = m.get("lo", -np.inf), m.get("hi", np.inf)
        frac_out = float(((now_vals < lo) | (now_vals > hi)).mean())
        return {"violated": frac_out > m["threshold"], "statistic": frac_out,
                "detail": f"{frac_out:.2f} outside [{lo}, {hi}]"}
    raise ValueError(f"unknown statistical test {test!r}")


# ---------------------------------------------------------------- registry


class HypothesisRegistry:
    """All hypotheses of one run + the contradiction event stream."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "researcher" / "hypotheses"
        self.events_path = self.run_dir / "researcher" / "contradiction_events.jsonl"
        self.hypotheses: dict[str, Hypothesis] = {}
        if self.dir.exists():
            for path in sorted(self.dir.glob("hyp-*.json")):
                h = Hypothesis.from_json(path.read_text(encoding="utf-8"))
                self.hypotheses[h.id] = h

    def add(self, h: Hypothesis) -> None:
        h.validate()
        self.hypotheses[h.id] = h
        self._save(h)

    def _save(self, h: Hypothesis) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{h.id}.json").write_text(h.to_json(), encoding="utf-8")

    def _emit_event(self, h: Hypothesis, transition: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "hypothesis_id": h.id, "statement": h.statement(),
                "scope": h.scope, **transition,
            }) + "\n")

    def check_all(self, engine: QueryEngine, now_tick: int) -> list[dict[str, Any]]:
        """Sleep-cadence monitor pass. Returns the transitions that fired."""
        fired = []
        for h in self.hypotheses.values():
            if h.status in ("contradicted", "retired"):
                continue
            # Behavior-coupled monitors arm only after the policy's early
            # settling (monitor.arm_after, in ticks): success rates change
            # genuinely while the position distribution stabilizes, and
            # flagging that as a capability contradiction is the same
            # calibrate-then-arm failure the mirror had (stage-6a).
            # BOTH comparison windows must lie past the boundary — a check
            # at arm_after still drags settling-era data in via the prev
            # window — so the first armed check is arm_after + 2*window.
            arm_after = h.monitor.get("arm_after", 0)
            if arm_after and now_tick < arm_after + 2 * h.monitor["window"]:
                continue
            result = run_check(h, engine, now_tick)
            h.last_checked = now_tick
            old = h.status
            if result["violated"]:
                h.status = "contradicted" if old == "weakening" else "weakening"
            elif old == "weakening":
                h.status = "supported"  # one clean check recovers
            if h.status != old:
                transition = {"tick": now_tick, "from": old, "to": h.status,
                              "evidence": result["detail"],
                              "statistic": result["statistic"]}
                h.transitions.append(transition)
                self._emit_event(h, transition)
                fired.append({"hypothesis_id": h.id, **transition})
            self._save(h)
        return fired


# ----------------------------------------------------------- auto templates


def build_default_hypotheses(
    world_size: int, num_actions: int, created_tick: int = 0,
    region_size: int = 8, max_regions: int = 16,
) -> list[Hypothesis]:
    """Auto-population at startup / after each battery: stationarity per
    region, capability stability per action, memory-decay validity,
    forecaster validity per horizon, calibration stability."""
    out: list[Hypothesis] = []
    n_regions = world_size // region_size
    idx = 0
    for r in range(n_regions):
        for c in range(n_regions):
            if idx >= max_regions:
                break
            idx += 1
            out.append(Hypothesis(
                schema_version=1, id=f"hyp-stationary-regrowth-{r}-{c}",
                statement_template=("food regrowth intervals in region "
                                    "({r},{c}) are stationary"),
                params={"r": r, "c": c}, scope="world",
                monitor={"metric_query": f"jsonl:regrowth_intervals:region={r},{c}",
                         "statistical_test": "ks_2sample", "threshold": 0.5,
                         "window": 4000, "min_samples": 8},
                created_tick=created_tick,
            ))
    for action in range(num_actions):
        out.append(Hypothesis(
            schema_version=1, id=f"hyp-capability-success-{action}",
            statement_template="success rate of action {action} is stable",
            params={"action": action}, scope="self_capability",
            monitor={"metric_query": f"jsonl:action_success:action={action}",
                     "statistical_test": "mean_shift", "threshold": 3.0,
                     "window": 4000, "min_samples": 20,
                     "arm_after": 2000},  # behavior-coupled: settle first
            created_tick=created_tick,
        ))
    out.append(Hypothesis(
        schema_version=1, id="hyp-memory-decay-valid",
        statement_template="the memory-reliability calibration stays in band",
        params={}, scope="self_memory",
        monitor={"metric_query": "tb:ledger/reliability_ece",
                 "statistical_test": "band", "threshold": 0.5,
                 "lo": 0.0, "hi": 0.25, "window": 8000, "min_samples": 3},
        created_tick=created_tick,
    ))
    out.append(Hypothesis(
        schema_version=1, id="hyp-forecaster-valid-k10",
        statement_template="forecaster NMSE at k=10 stays in its expected band",
        params={}, scope="self_model",
        monitor={"metric_query": "tb:sleep/forecaster_nmse_k10",
                 "statistical_test": "band", "threshold": 0.5,
                 "lo": 0.0, "hi": 50.0, "window": 8000, "min_samples": 3},
        created_tick=created_tick,
    ))
    return out


def load_yaml_hypotheses(path: str | Path, created_tick: int = 0) -> list[Hypothesis]:
    """Human-added hypotheses (researcher/hypotheses.yaml), validated on load."""
    import yaml

    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    out = []
    for entry in raw:
        h = Hypothesis(schema_version=HYPOTHESIS_SCHEMA_VERSION,
                       created_tick=created_tick, **entry)
        h.validate()
        out.append(h)
    return out
