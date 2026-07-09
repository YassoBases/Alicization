"""LLM drafting pass (stage-D3): turn the analysis + a run's evidence into
STRICT-JSON schema-v2 proposals.

The network call is the Architect's ONE allowed side effect and is confined
to this module. A stub client is INJECTED in tests (the EIG ModelAdapter
seam); the real Anthropic client is built lazily so nothing here imports the
SDK unless actually going online. The `architect.offline` config kill switch
makes drafting a no-op so the whole pipeline stays testable offline.

Malformed model output gets ONE repair round-trip, then the batch is
discarded with a logged reason. Every call's prompt is hashed into each
proposal's provenance (reproducibility standing rule). Proposals touching
constitutional files are rejected here, before they can be emitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from architect.analysis import AnalysisReport
from architect.constitution import ConstitutionViolation, validate_proposal
from architect.paths import write_under_architect
from evidence import EvidenceView
from proposals.schema import INTERVENTION_CLASSES, TYPES, Proposal

SYSTEM_PROMPT = (
    "You are an experiment architect for a contained RL research sandbox. "
    "You read analysis + run evidence and PROPOSE changes for a human to "
    "review; you never apply anything. Emit ONLY a JSON array of proposal "
    "objects, no prose. Each object: {type (one of the allowed types), "
    "intervention_class ('config' or 'architecture'), target, rationale, "
    "supporting_observations (list of evidence refs like "
    "'tb:<tag>@step=N', 'competence:...', or 'code:<path>@<sha>#Lx-Ly' that "
    "MUST exist in the provided evidence), expected_benefit "
    "{metric,direction,magnitude_estimate}, success_criteria "
    "{metric,threshold,eval_window_ticks}, estimated_cost "
    "{human_hours,gpu_hours}, risks (list), confidence (0..1), "
    "proposed_change {config_path,new_value} or null, and an optional 'diff' "
    "(a unified diff string). NEVER propose changes to CLAUDE.md, the safety "
    "docs, the review layer, the no-execution / gradient-isolation / mirror "
    "tests, or the architect's constitution — those are off limits."
)


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, prompt: str, model: str,
                 temperature: float, max_tokens: int) -> str: ...


class StubClient:
    """Injected in tests: returns canned responses in order (the last one
    repeats), recording the prompts it saw."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, prompt: str, model: str,
                 temperature: float, max_tokens: int) -> str:
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def build_anthropic_client() -> LLMClient:  # pragma: no cover (network)
    """Lazily wrap the Anthropic SDK. Only reached on the online path; the
    key comes from ANTHROPIC_API_KEY (never a config/secret in the repo)."""
    import os

    import anthropic  # imported here so offline runs never need the SDK

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    class _Client:
        def complete(self, *, system: str, prompt: str, model: str,
                     temperature: float, max_tokens: int) -> str:
            msg = client.messages.create(
                model=model, system=system, temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if b.type == "text")

    return _Client()


@dataclass
class DraftResult:
    proposals: list[Proposal] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    prompt_hash: str = ""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_prompt(report: AnalysisReport, view: EvidenceView,
                 acfg: dict[str, Any]) -> str:
    """Prompt = analysis summary + evidence summary + capped source excerpts
    for the modules symptom-linkage flagged."""
    inv = report.invariants
    lines = [
        f"# Run {view.run_id} @ tick {view.tick} (source={view.source})",
        f"repo sha: {view.repo_snapshot.git_sha or 'unknown'}",
        "",
        "## Invariants (do not violate; do not propose to change these files)",
        *[f"- {r}" for r in inv.hard_rules[:12]],
        "",
        "## Anomalous scalar tags -> emitting modules",
        *[f"- {tag}: {', '.join(mods)}" for tag, mods in report.tag_emitters.items()],
        "",
        "## Available evidence refs (cite only these)",
        *[f"- tb:{tag}" for tag in sorted(view.scalars)][:40],
    ]
    if view.competence is not None:
        lines.append(f"- competence:report-{view.competence.tick}")
    # Capped source excerpts of the flagged modules.
    max_files = int(acfg.get("max_source_files", 6))
    max_lines = int(acfg.get("max_source_lines", 200))
    flagged: list[str] = []
    for mods in report.tag_emitters.values():
        for m in mods:
            if m not in flagged:
                flagged.append(m)
    lines += ["", "## Source excerpts"]
    root = Path(report.repo_root)
    for rel in flagged[:max_files]:
        try:
            body = (root / rel).read_text(encoding="utf-8").splitlines()[:max_lines]
        except OSError:
            continue
        lines += [f"### {rel}", "```python", *body, "```"]
    return "\n".join(lines)


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array, tolerating ```json fences and leading/trailing
    prose. Raises ValueError if no array is recoverable."""
    text = raw.strip()
    if "```" in text:
        # take the first fenced block's body
        parts = text.split("```")
        for chunk in parts:
            chunk = chunk.lstrip()
            if chunk.startswith("json"):
                chunk = chunk[4:]
            chunk = chunk.strip()
            if chunk.startswith("["):
                text = chunk
                break
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON array")
    return data


def _to_proposal(d: dict[str, Any], view: EvidenceView, run_dir: Path,
                 model_id: str, prompt_hash: str) -> Proposal:
    if d.get("type") not in TYPES:
        raise ValueError(f"bad or missing type {d.get('type')!r}")
    ic = d.get("intervention_class", "architecture")
    if ic not in INTERVENTION_CLASSES:
        raise ValueError(f"bad intervention_class {ic!r}")
    proposal = Proposal.new(
        type=d["type"], intervention_class=ic, created_tick=view.tick,
        run_id=view.run_id, source=f"architect:{model_id}",
        rationale=d.get("rationale", ""),
        expected_benefit=d["expected_benefit"], confidence=float(d.get("confidence", 0.5)),
        supporting_observations=list(d.get("supporting_observations", [])),
        estimated_cost=d["estimated_cost"], risks=list(d.get("risks", [])),
        success_criteria=d["success_criteria"], target=d.get("target", ""),
        proposed_change=d.get("proposed_change"),
        provenance={"evidence_bundle_hash": view.bundle_hash,
                    "generator_id": "architect", "prompt_hash": prompt_hash,
                    "model_id": model_id})
    diff = d.get("diff")
    if isinstance(diff, str) and diff.strip():
        rel = f"diffs/{proposal.id}.diff"
        write_under_architect(run_dir, rel, diff)
        proposal.artifacts = [f"architect/{rel}"]
        proposal.validate()
    # Constitution: reject before this can be emitted (also defends the
    # system-prompt instruction with a hard check).
    validate_proposal(proposal, run_dir)
    return proposal


def draft_proposals(report: AnalysisReport, view: EvidenceView,
                    run_dir: str | Path, cfg: dict[str, Any],
                    client: LLMClient | None = None) -> DraftResult:
    """Draft schema-v2 proposals from the analysis + evidence. Offline (or
    no client) -> a logged no-op. Malformed JSON -> one repair, then discard."""
    run_dir = Path(run_dir)
    acfg = cfg.get("architect", {})
    if acfg.get("offline", True) and client is None:
        return DraftResult(decisions=[{"action": "skip",
            "reason": "architect.offline: drafting disabled (no network)"}])
    client = client or build_anthropic_client()
    model_id = acfg.get("model_id", "claude-sonnet-5")
    prompt = build_prompt(report, view, acfg)
    prompt_hash = _hash(SYSTEM_PROMPT + prompt)

    raw = client.complete(system=SYSTEM_PROMPT, prompt=prompt, model=model_id,
                          temperature=float(acfg.get("temperature", 0.2)),
                          max_tokens=int(acfg.get("max_tokens", 4096)))
    decisions: list[dict[str, Any]] = []
    try:
        dicts = _extract_json_array(raw)
    except ValueError:
        # one repair round-trip
        repair = client.complete(
            system=SYSTEM_PROMPT,
            prompt="Your previous output was not a valid JSON array. Return "
                   "ONLY the JSON array of proposals, nothing else.",
            model=model_id, temperature=0.0,
            max_tokens=int(acfg.get("max_tokens", 4096)))
        try:
            dicts = _extract_json_array(repair)
            decisions.append({"action": "repair", "reason": "malformed JSON repaired"})
        except ValueError as exc:
            return DraftResult(decisions=[{"action": "discard_batch",
                "reason": f"malformed JSON after repair: {exc}",
                "prompt_hash": prompt_hash}], prompt_hash=prompt_hash)

    result = DraftResult(prompt_hash=prompt_hash)
    for d in dicts:
        try:
            proposal = _to_proposal(d, view, run_dir, model_id, prompt_hash)
        except ConstitutionViolation as exc:
            decisions.append({"action": "discard", "reason": f"constitution: {exc}"})
            continue
        except (ValueError, KeyError) as exc:
            decisions.append({"action": "discard", "reason": f"malformed proposal: {exc}"})
            continue
        result.proposals.append(proposal)
        decisions.append({"action": "draft", "proposal_id": proposal.id,
                          "target": proposal.target})
    result.decisions = decisions
    return result
