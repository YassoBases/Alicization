"""Mirror monitor: self-state divergence between world model and body model.

Two independent estimates of "where will I be next tick":
  * decoder-implied — the RSSM decoder's allocentric pose head sees the
    agent-as-object inside its world model (agent/core_rssm.py's
    ``implied_pose``), and
  * body-model-implied — the egocentric body model's predicted dpos for the
    chosen action, anchored at the proprioceptive position.

Divergence = Euclidean distance between the two implied positions (in
cells). It is logged every tick and is a MONITOR ONLY: it is computed under
``torch.no_grad`` from detached inputs and returns plain numpy — no loss
term may contain it (asserted structurally in tests/test_mirror.py). A
self-model that never disagrees with the world model has been optimized
into agreement, not accuracy.

Threshold crossing triggers, per env:
  (a) a SELF-CHECK routine — four probe actions with known expected
      outcomes (NOOP stay, REST stay, MOVE_E then MOVE_W return-trip); the
      realized transitions immediately refresh the body model (one focused
      gradient step on just those probes), and
  (b) MPC deliberation — for a few ticks the habitual policy is replaced by
      model-predictive control: sample candidate action sequences, roll the
      RSSM prior forward, score by predicted reward, execute the best first
      action. Deliberate re-planning while the self-estimate is suspect.

State machine per env: NORMAL -> (divergence > threshold) -> PROBE (4
ticks) -> refresh body model -> MPC (mpc_ticks) -> NORMAL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ledger.body_model import DPOS_CLASSES, BodyModel, compute_body_losses

# Probe script: (action_id, expected dpos class index). Known outcomes under
# an intact body: NOOP/REST realize "stay"; MOVE_E then MOVE_W is a return
# trip (E may legitimately fail at the map edge — the refresh uses whatever
# actually happened, the EXPECTATIONS are only for logging).
_NOOP, _REST = 8, 5
_MOVE_E, _MOVE_W = 2, 3
PROBE_SCRIPT: tuple[tuple[int, int], ...] = (
    (_NOOP, 0), (_REST, 0), (_MOVE_E, 3), (_MOVE_W, 4),
)

NORMAL, PROBING, DELIBERATING = 0, 1, 2


class MirrorMonitor:
    """Per-env divergence monitor + probe/MPC trigger state machine."""

    def __init__(self, cfg: dict[str, Any], num_envs: int, world_size: int) -> None:
        """``cfg`` is the ``mirror`` config section."""
        self.enabled: bool = cfg.get("enabled", False)
        self.threshold: float = cfg.get("threshold", 3.0)
        # Responses stay disarmed for the first warmup_ticks step_state calls:
        # early-training divergence is dominated by the untrained pose head,
        # and probing/MPC on that noise corrupts the very training that would
        # bring it down (measured: >1k spurious triggers per smoke run when
        # armed from tick 0). Calibrate first, then arm.
        self.warmup_ticks: int = cfg.get("warmup_ticks", 0)
        self._ticks_seen = 0
        self.mpc_ticks: int = cfg.get("mpc_ticks", 4)
        self.mpc_horizon: int = cfg.get("mpc_horizon", 6)
        self.mpc_candidates: int = cfg.get("mpc_candidates", 64)
        self.world_size = world_size
        self.num_envs = num_envs
        self.state = np.full(num_envs, NORMAL, dtype=np.int64)
        self._phase_step = np.zeros(num_envs, dtype=np.int64)
        self._probe_transitions: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self.divergence_history: list[np.ndarray] = []
        self.trigger_count = 0
        # Set by step_state on a PROBING -> DELIBERATING transition: the
        # trainer refreshes the body model on the collected probes then.
        self.just_finished_probing = np.zeros(num_envs, dtype=bool)
        self.rng = np.random.default_rng(cfg.get("seed", 0))

    # ------------------------------------------------------------ divergence

    @torch.no_grad()
    def divergence(
        self,
        core: Any,
        body_model: BodyModel,
        features: torch.Tensor,
        actions: torch.Tensor,
        positions: list[tuple[int, int]],
    ) -> np.ndarray:
        """(N,) distance in cells between the two implied next positions.

        MONITORED, NEVER MINIMIZED: computed under no_grad from detached
        inputs; returns plain numpy that cannot enter any autograd graph.
        """
        onehot = F.one_hot(actions, body_model.num_actions).float()
        decoder_pos = core.implied_pose(features.detach(), onehot).cpu().numpy()
        decoder_pos = decoder_pos * self.world_size  # denormalize

        body_out = body_model.predict_action(features.detach(), onehot)
        dpos_class = body_out["dpos_logits"].argmax(dim=-1).cpu().numpy()
        body_pos = np.asarray(positions, dtype=np.float64)
        for i, cls in enumerate(dpos_class):
            dx, dy = DPOS_CLASSES[int(cls)]
            body_pos[i, 0] += dx
            body_pos[i, 1] += dy

        div = np.linalg.norm(decoder_pos - body_pos, axis=1)
        self.divergence_history.append(div)
        return div

    # ---------------------------------------------------------- state machine

    def step_state(self, divergence: np.ndarray) -> None:
        """Advance the per-env trigger state machine one tick.

        Call AFTER ``divergence``; action overrides for the new state apply
        from the NEXT tick (``override_actions``).
        """
        self.just_finished_probing[:] = False
        self._ticks_seen += 1
        if not self.enabled or self._ticks_seen <= self.warmup_ticks:
            return
        for i in range(self.num_envs):
            if self.state[i] == NORMAL:
                if divergence[i] > self.threshold:
                    self.state[i] = PROBING
                    self._phase_step[i] = 0
                    self._probe_transitions[i].clear()
                    self.trigger_count += 1
            elif self.state[i] == PROBING:
                if self._phase_step[i] >= len(PROBE_SCRIPT):
                    self.state[i] = DELIBERATING
                    self._phase_step[i] = 0
                    self.just_finished_probing[i] = True
            elif self.state[i] == DELIBERATING:
                if self._phase_step[i] >= self.mpc_ticks:
                    self.state[i] = NORMAL
                    self._phase_step[i] = 0

    def override_actions(
        self,
        default_actions: torch.Tensor,
        core: Any,
        body_model: BodyModel,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Replace actions for envs in PROBE (scripted) or MPC (planned) mode."""
        if not self.enabled or bool((self.state == NORMAL).all()):
            return default_actions
        actions = default_actions.clone()
        mpc_envs = np.nonzero(self.state == DELIBERATING)[0]
        if len(mpc_envs):
            planned = self._mpc_plan(core, body_model, features[mpc_envs])
            for j, env in enumerate(mpc_envs):
                actions[env] = int(planned[j])
        for env in np.nonzero(self.state == PROBING)[0]:
            actions[env] = PROBE_SCRIPT[int(self._phase_step[env])][0]
        self._phase_step[self.state != NORMAL] += 1
        return actions

    # -------------------------------------------------------------- probing

    def record_probe_result(
        self,
        env: int,
        h_detached: torch.Tensor,
        action: int,
        real_dpos_class: int,
        real_success: float,
        real_denergy: float,
    ) -> None:
        """Store one probe transition (called by the trainer for PROBING envs)."""
        self._probe_transitions[env].append({
            "h": h_detached.cpu(), "action": action,
            "dpos_class": real_dpos_class, "success": real_success,
            "denergy": real_denergy,
        })

    def refresh_body_model(
        self, body_model: BodyModel, body_opt: torch.optim.Optimizer, env: int
    ) -> float | None:
        """One focused gradient step on this env's collected probe results."""
        probes = self._probe_transitions[env]
        if not probes:
            return None
        # Callers sit inside the rollout's no_grad scope; this focused update
        # is a legitimate Ledger training step (detached inputs), so re-enable.
        with torch.enable_grad():
            h = torch.stack([p["h"] for p in probes])
            actions = torch.tensor([p["action"] for p in probes], dtype=torch.long)
            onehot = F.one_hot(actions, body_model.num_actions).float()
            outputs = body_model.predict_action(h, onehot)
            losses = compute_body_losses(
                outputs,
                torch.tensor([p["dpos_class"] for p in probes], dtype=torch.long),
                torch.tensor([p["success"] for p in probes], dtype=torch.float32),
                torch.tensor([p["denergy"] for p in probes], dtype=torch.float32),
            )
            body_opt.zero_grad()
            losses["total"].backward()
            body_opt.step()
        self._probe_transitions[env].clear()
        return float(losses["total"].item())

    # ------------------------------------------------------------------ MPC

    @torch.no_grad()
    def _mpc_plan(
        self, core: Any, body_model: BodyModel, features: torch.Tensor
    ) -> np.ndarray:
        """Random-shooting MPC: roll ``mpc_candidates`` action sequences of
        length ``mpc_horizon`` through the RSSM prior (ensemble-mean dynamics),
        score by the reward head's cumulative prediction, return the best
        first action per env. (B, F) -> (B,) actions."""
        b = features.shape[0]
        k, horizon = self.mpc_candidates, self.mpc_horizon
        num_actions = body_model.num_actions
        seqs = self.rng.integers(0, num_actions, size=(b, k, horizon))
        h = features.repeat_interleave(k, dim=0)  # (B*K, F)
        total_reward = torch.zeros(b * k)
        for t in range(horizon):
            acts = torch.from_numpy(seqs[:, :, t].reshape(-1)).long()
            onehot = F.one_hot(acts, num_actions).float()
            means_t, _, _ = core.ensemble_stats(h, onehot)
            embed_pred = means_t.mean(dim=0)
            deter = core._step_deter(embed_pred, h)
            prior_mean, _ = core._stats(core.prior_net(deter))
            h = torch.cat([deter, prior_mean], dim=-1)
            total_reward += core.reward_head(h).squeeze(-1).cpu()
        best = total_reward.reshape(b, k).argmax(dim=1).numpy()
        return seqs[np.arange(b), best, 0]

    # ----------------------------------------------------------------- state

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.copy(), "phase_step": self._phase_step.copy(),
            "trigger_count": self.trigger_count,
            "ticks_seen": self._ticks_seen,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = state["state"].copy()
        self._phase_step = state["phase_step"].copy()
        self.trigger_count = state["trigger_count"]
        self._ticks_seen = state.get("ticks_seen", 0)
        self.rng.bit_generator.state = state["rng_state"]
