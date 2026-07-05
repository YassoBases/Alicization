"""Recurrent PPO trainer.

Hidden-state discipline (the classic silent-failure modes, made explicit):
- During collection, the hidden state fed into the core at step t is
  ``h_{t-1} * (1 - done_{t-1})`` — zeroed exactly at episode boundaries.
- Rollouts are split into BPTT segments of ``ppo.seq_len``. The (already
  masked) input hidden at each segment start is stored and replayed during
  updates; within a segment, ``replay_core`` re-applies the same done masks.
- GAE never bootstraps across a done: ``delta_t`` uses ``(1 - done_t)``.

Core trains only on world-prediction + task loss; Ledger heads (future) train
on detached hidden states (CLAUDE.md Hard rules).
"""

from __future__ import annotations

import contextlib
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from agent.core_gru import GRUCore
from agent.core_rssm import RSSMCore
from agent.encoder import ObsEncoder
from agent.policy import ActorCritic
from ledger.attribution import (
    AttributionHead,
    compute_attribution_loss,
    pseudo_label,
    residual_features,
)
from ledger.body_model import (
    BodyModel,
    RollingMean,
    build_policy_features,
    compute_body_losses,
    dpos_to_class,
)
from training.attribution_eval import AttributionAccuracyTracker, ground_truth_label
from training.checkpoints import load_checkpoint, prune_checkpoints, save_checkpoint
from training.loggers import TBLogger
from training.monitors import ParticipationRatioMonitor
from training.vecenv import VecWorld
from world.engine import NOOP, NUM_ACTIONS


def resolve_device(name: str) -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise pass through."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE advantages for (T, N) tensors; ``dones[t]=1`` blocks bootstrapping."""
    horizon = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for t in reversed(range(horizon)):
        v_next = next_value if t == horizon - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv


def replay_core(
    core: GRUCore | RSSMCore, embeds: torch.Tensor, h0: torch.Tensor, dones: torch.Tensor
) -> torch.Tensor:
    """Replay a segment through the core with done masking.

    ``embeds`` (T, B, E), ``h0`` (B, H) — the already-masked input hidden for
    the first step — and ``dones`` (T, B). The hidden state is zeroed AFTER any
    step where done=1, i.e. before it is used as the next step's input.
    Returns per-step core outputs (T, B, core.output_dim).
    """
    h = h0
    outputs = []
    for t in range(embeds.shape[0]):
        out, h = core(embeds[t], h)
        outputs.append(out)
        h = h * (1.0 - dones[t]).unsqueeze(-1)
    return torch.stack(outputs)


class PPOModel(nn.Module):
    """Encoder + recurrent core + actor-critic heads as one checkpointable module.

    ``cfg["agent"]`` has no separate embedding-width key: the observation
    embedding and the GRU input share one width, ``agent.hidden_size``, by
    convention (see agent/encoder.py). Unless ``agent.use_ledger_features`` is
    set false, the policy/value heads take the core output PLUS the body
    model's detached per-action success/denergy predictions (see
    ledger.body_model.build_policy_features), hence the ``+ 2 * NUM_ACTIONS``
    — the body model itself lives outside this module (PPOTrainer.body_model),
    trained by its own optimizer, and always trains regardless of this flag
    (see build_policy_features's docstring: this is the capability-shift
    battery's architecture-A/B toggle, experiments/batteries/capability_shift.py).
    """

    def __init__(
        self, cfg: dict[str, Any], grid_channels: int, intero_dim: int, window: int
    ) -> None:
        super().__init__()
        acfg = cfg["agent"]
        self.core_kind: str = acfg.get("core", "gru")
        self.use_ledger_features: bool = acfg.get("use_ledger_features", True)
        if self.core_kind == "rssm":
            # RSSM uses the canonical rssm.embed width for the observation
            # embedding (its posterior/decoder are sized for it).
            embed_dim = cfg["rssm"]["embed"]
            self.encoder = ObsEncoder(acfg, grid_channels, intero_dim, embed_dim, window)
            self.core: GRUCore | RSSMCore = RSSMCore(
                cfg["rssm"], input_dim=embed_dim,
                grid_shape=(grid_channels, window, window),
                intero_dim=intero_dim, num_actions=NUM_ACTIONS,
            )
        elif self.core_kind == "gru":
            embed_dim = acfg["hidden_size"]
            self.encoder = ObsEncoder(acfg, grid_channels, intero_dim, embed_dim, window)
            self.core = GRUCore(acfg, input_dim=embed_dim)
        else:
            raise ValueError(f"unknown agent.core: {self.core_kind!r} (gru|rssm)")
        extra = 2 * NUM_ACTIONS if self.use_ledger_features else 0
        policy_input_dim = self.core.output_dim + extra
        self.heads = ActorCritic(acfg, policy_input_dim, NUM_ACTIONS)


class PPOTrainer:
    """Rollout collection + clipped-surrogate updates for the recurrent policy."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.pcfg = cfg["ppo"]
        self.device = resolve_device(device or cfg.get("device", "auto"))
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        self.vec = VecWorld(cfg)
        grid_c = self.vec.grid_shape[0]
        window = self.vec.grid_shape[1]
        self.model = PPOModel(cfg, grid_c, self.vec.intero_dim, window).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.pcfg["lr"], eps=1e-5)

        lcfg = cfg["ledger"]
        self.body_model = BodyModel(
            lcfg, core_dim=self.model.core.output_dim, num_actions=NUM_ACTIONS
        ).to(self.device)
        self.body_opt = torch.optim.Adam(self.body_model.parameters(), lr=lcfg["lr"])
        ema_decay = lcfg.get("log_ema_decay", 0.98)
        self._body_nll_ema = RollingMean(ema_decay)
        self._success_brier_ema = RollingMean(ema_decay)
        self._denergy_mae_ema = RollingMean(ema_decay)
        self.body_nll_history: list[float] = []

        acfg = lcfg["attribution"]
        self.attribution_model = AttributionHead(acfg).to(self.device)
        self.attr_opt = torch.optim.Adam(self.attribution_model.parameters(), lr=acfg["lr"])
        self.attr_tau_pos = acfg["tau_pos"]
        self.attr_tau_energy = acfg["tau_energy"]
        self._attr_accuracy_ema = RollingMean(ema_decay)
        self.attr_tracker = AttributionAccuracyTracker()
        self.attribution_accuracy_history: list[float] = []

        n = self.pcfg["num_envs"]
        hidden = self.model.core.hidden_dim  # flat state size (layers * hidden_size)
        self._h = torch.zeros(n, hidden, device=self.device)
        self._done_prev = torch.zeros(n, device=self.device)
        self._obs = self.vec.observe()

        # RSSM-only monitoring: epistemic map + participation-ratio collapse
        # detector (training-side bookkeeping; nothing here is agent-visible).
        self.is_rssm = isinstance(self.model.core, RSSMCore)
        world_size = cfg["world"]["size"]
        self.epistemic_map = np.zeros((world_size, world_size), dtype=np.float64)
        self.epistemic_count = np.zeros((world_size, world_size), dtype=np.int64)
        mon_cfg = cfg.get("rssm", {}).get("monitor", {}) or {}
        self.pr_monitor = ParticipationRatioMonitor(
            every_ticks=mon_cfg.get("every_ticks", 1000),
            window=mon_cfg.get("window", 1000),
            collapse_frac=mon_cfg.get("collapse_frac", 0.25),
        )
        self.pr_history: list[float] = []

        self.global_step = 0
        self._last_ckpt_step = 0
        self.reward_history: list[float] = []
        self.last_metrics: dict[str, float] = {}
        self._interrupted = False

        seq = self.pcfg["seq_len"]
        if self.pcfg["rollout_steps"] % seq != 0:
            raise ValueError("ppo.rollout_steps must be a multiple of ppo.seq_len")

        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.tb: TBLogger | None = None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.tb = TBLogger(self.run_dir / "tb")

    # -------------------------------------------------------------- rollouts

    def _obs_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.from_numpy(self._obs["grid"]).to(self.device)
        intero = torch.from_numpy(self._obs["intero"]).to(self.device)
        return grid, intero

    @torch.no_grad()
    def collect_rollout(self) -> dict[str, torch.Tensor]:
        """Collect one on-policy rollout of shape (T, N, ...).

        Also records, per tick, the core output (``core_out``, detached —
        collection runs under ``torch.no_grad()`` throughout) and the realized
        transition (dpos class, success, denergy) from the vecenv info dicts,
        for the body model's online update (see ``update_body_model``); and
        the body model's predicted (dpos class, denergy) for the action taken
        plus the ground-truth cause label (see ``update_attribution_model``).
        The ground-truth label is read here from vecenv info's ``events``
        (world/levers.py's cause={self,world} log) purely to compare against
        the attribution classifier's own prediction for reporting — it is a
        plain Python int by the time it reaches ``buf``, never a tensor
        upstream of any loss.
        """
        p = self.pcfg
        rollout_len, n, seq = p["rollout_steps"], p["num_envs"], p["seq_len"]
        c, w, _ = self.vec.grid_shape
        dev = self.device
        core_dim = self.model.core.output_dim
        self._epistemic_sum, self._aleatoric_sum, self._uncert_n = 0.0, 0.0, 0
        buf = {
            "grid": torch.zeros(rollout_len, n, c, w, w, device=dev),
            "intero": torch.zeros(rollout_len, n, self.vec.intero_dim, device=dev),
            "action": torch.zeros(rollout_len, n, dtype=torch.long, device=dev),
            "logp": torch.zeros(rollout_len, n, device=dev),
            "value": torch.zeros(rollout_len, n, device=dev),
            "reward": torch.zeros(rollout_len, n, device=dev),
            "done": torch.zeros(rollout_len, n, device=dev),
            "h_init": torch.zeros(rollout_len // seq, n, self._h.shape[1], device=dev),
            "core_out": torch.zeros(rollout_len, n, core_dim, device=dev),
            "real_dpos_class": torch.zeros(rollout_len, n, dtype=torch.long, device=dev),
            "real_success": torch.zeros(rollout_len, n, device=dev),
            "real_denergy": torch.zeros(rollout_len, n, device=dev),
            "pred_dpos_class": torch.zeros(rollout_len, n, dtype=torch.long, device=dev),
            "pred_denergy": torch.zeros(rollout_len, n, device=dev),
            "ground_truth_label": torch.zeros(rollout_len, n, dtype=torch.long, device=dev),
        }
        for t in range(rollout_len):
            h_in = self._h * (1.0 - self._done_prev).unsqueeze(-1)
            if t % seq == 0:
                buf["h_init"][t // seq] = h_in
            grid, intero = self._obs_tensors()
            embed = self.model.encoder(grid, intero)
            out, h_new = self.model.core(embed, h_in)
            features, body_out = build_policy_features(
                out, self.body_model, self.model.use_ledger_features
            )
            dist, value = self.model.heads(features)
            action = dist.sample()

            obs, rewards, dones, infos = self.vec.step(action.cpu().numpy())
            buf["grid"][t] = grid
            buf["intero"][t] = intero
            buf["action"][t] = action
            buf["logp"][t] = dist.log_prob(action)
            buf["value"][t] = value
            buf["reward"][t] = torch.from_numpy(rewards).to(dev)
            buf["done"][t] = torch.from_numpy(dones).to(dev)
            buf["core_out"][t] = out
            dpos = torch.tensor(
                [info["realized"]["dpos"] for info in infos], dtype=torch.long, device=dev
            )
            buf["real_dpos_class"][t] = dpos_to_class(dpos)
            buf["real_success"][t] = torch.tensor(
                [float(info["realized"]["success"]) for info in infos], device=dev
            )
            buf["real_denergy"][t] = torch.tensor(
                [info["realized"]["denergy"] for info in infos], device=dev
            )
            action_idx = action.unsqueeze(-1)
            buf["pred_dpos_class"][t] = body_out["dpos_class"].gather(1, action_idx).squeeze(-1)
            buf["pred_denergy"][t] = body_out["denergy"].gather(1, action_idx).squeeze(-1)
            if self.is_rssm:
                onehot = torch.nn.functional.one_hot(action, NUM_ACTIONS).float()
                _, epistemic, aleatoric = self.model.core.ensemble_stats(out, onehot)
                self._epistemic_sum += epistemic.sum().item()
                self._aleatoric_sum += aleatoric.sum().item()
                self._uncert_n += epistemic.numel()
                for e_val, info in zip(epistemic.cpu().tolist(), infos):
                    x, y = info["pos"]
                    cnt = self.epistemic_count[y, x] + 1
                    self.epistemic_count[y, x] = cnt
                    self.epistemic_map[y, x] += (e_val - self.epistemic_map[y, x]) / cnt
                deter = out[:, : self.model.core.deter].cpu().numpy()
                self.pr_monitor.add(deter)
                pr = self.pr_monitor.maybe_compute(self.global_step + (t + 1) * n)
                if pr is not None:
                    self.pr_history.append(pr)
            # Ground truth: evaluation-only bookkeeping, never touches a loss.
            buf["ground_truth_label"][t] = torch.tensor(
                [
                    ground_truth_label(int(a), info["events"])
                    for a, info in zip(action.cpu().tolist(), infos)
                ],
                dtype=torch.long,
                device=dev,
            )
            self._h = h_new
            self._done_prev = buf["done"][t]
            self._obs = obs

        # Bootstrap value for GAE (masked hidden; unused across dones anyway).
        h_in = self._h * (1.0 - self._done_prev).unsqueeze(-1)
        grid, intero = self._obs_tensors()
        out, _ = self.model.core(self.model.encoder(grid, intero), h_in)
        features, _ = build_policy_features(out, self.body_model, self.model.use_ledger_features)
        _, next_value = self.model.heads(features)
        buf["next_value"] = next_value

        self.global_step += rollout_len * n
        return buf

    # --------------------------------------------------------------- updates

    def update(self, buf: dict[str, torch.Tensor]) -> dict[str, float]:
        """PPO epochs over BPTT segments; returns mean scalar metrics.

        ``ppo.minibatch_transitions`` sizes minibatches in raw env-step units;
        it is converted to a count of BPTT sequences here and need not divide
        the rollout evenly (the last minibatch in an epoch may be smaller).
        """
        p = self.pcfg
        rollout_len, n, seq = p["rollout_steps"], p["num_envs"], p["seq_len"]
        n_seg = rollout_len // seq

        adv = compute_gae(
            buf["reward"], buf["value"], buf["done"], buf["next_value"],
            p["gamma"], p["gae_lambda"],
        )
        returns = adv + buf["value"]

        def by_segment(x: torch.Tensor) -> torch.Tensor:
            # (T, N, ...) -> (n_seg, seq, N, ...)
            return x.reshape(n_seg, seq, *x.shape[1:])

        seg = {k: by_segment(buf[k]) for k in
               ("grid", "intero", "action", "logp", "done", "reward")}
        seg_adv, seg_ret, seg_val = by_segment(adv), by_segment(returns), by_segment(buf["value"])

        n_samples = n_seg * n  # number of BPTT sequences in this rollout
        total_transitions = rollout_len * n
        num_minibatches = max(1, min(n_samples, total_transitions // p["minibatch_transitions"]))
        mb_size = max(1, n_samples // num_minibatches)
        amp = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.cfg.get("amp")
            else contextlib.nullcontext()
        )
        metric_keys = ["loss/policy", "loss/value", "loss/total", "entropy",
                       "approx_kl", "clip_frac"]
        if self.is_rssm:
            metric_keys += ["rssm/recon", "rssm/kl", "rssm/ensemble_nll"]
        metrics = {k: 0.0 for k in metric_keys}
        n_mb = 0

        for _ in range(p["epochs"]):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, mb_size):
                idx = perm[start : start + mb_size]
                s_idx, e_idx = idx // n, idx % n  # segment id, env id
                # Gather (seq, M, ...) minibatches.
                mb = {k: v[s_idx, :, e_idx].transpose(0, 1) for k, v in seg.items()}
                mb_adv = seg_adv[s_idx, :, e_idx].transpose(0, 1).reshape(-1)
                mb_ret = seg_ret[s_idx, :, e_idx].transpose(0, 1).reshape(-1)
                mb_val_old = seg_val[s_idx, :, e_idx].transpose(0, 1).reshape(-1)
                h0 = buf["h_init"][s_idx, e_idx]

                if p.get("norm_adv", True) and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with amp:
                    m = mb["grid"].shape[1]
                    flat_grid = mb["grid"].reshape(seq * m, *mb["grid"].shape[2:])
                    flat_intero = mb["intero"].reshape(seq * m, -1)
                    embeds = self.model.encoder(flat_grid, flat_intero)
                    outs = replay_core(
                        self.model.core, embeds.reshape(seq, m, -1), h0, mb["done"]
                    )
                    flat_core_out = outs.reshape(seq * m, -1)
                    features, _ = build_policy_features(
                        flat_core_out, self.body_model, self.model.use_ledger_features
                    )
                    dist, value = self.model.heads(features)
                    # (seq, M) -> flat, matching outs layout
                    flat_action = mb["action"].reshape(-1)
                    new_logp = dist.log_prob(flat_action)
                    entropy = dist.entropy().mean()

                    logratio = new_logp - mb["logp"].reshape(-1)
                    ratio = logratio.exp()
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * ratio.clamp(1.0 - p["clip"], 1.0 + p["clip"])
                    policy_loss = torch.max(pg1, pg2).mean()

                    if p.get("value_clip"):
                        v_clipped = mb_val_old + (value - mb_val_old).clamp(
                            -p["value_clip"], p["value_clip"]
                        )
                        value_loss = 0.5 * torch.max(
                            (value - mb_ret) ** 2, (v_clipped - mb_ret) ** 2
                        ).mean()
                    else:
                        value_loss = 0.5 * ((value - mb_ret) ** 2).mean()

                    loss = (
                        policy_loss
                        + p["value_coef"] * value_loss
                        - p["entropy_coef"] * entropy
                    )

                    if self.is_rssm:
                        # World-prediction loss: the ONLY other loss allowed to
                        # train the core (CLAUDE.md). Shares this backward pass
                        # and optimizer with the task loss.
                        wm = self.model.core.world_model_loss(
                            embeds.reshape(seq, m, -1),
                            h0,
                            mb["done"],
                            mb["action"],
                            mb["grid"],
                            mb["intero"],
                            rewards=mb["reward"],
                        )
                        loss = loss + wm["total"]
                        metrics["rssm/recon"] += (
                            wm["recon_grid"].item() + wm["recon_intero"].item()
                        )
                        metrics["rssm/kl"] += wm["kl"].item()
                        metrics["rssm/ensemble_nll"] += wm["ensemble_nll"].item()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()

                with torch.no_grad():
                    metrics["approx_kl"] += ((ratio - 1.0) - logratio).mean().item()
                    metrics["clip_frac"] += (
                        ((ratio - 1.0).abs() > p["clip"]).float().mean().item()
                    )
                metrics["loss/policy"] += policy_loss.item()
                metrics["loss/value"] += value_loss.item()
                metrics["loss/total"] += loss.item()
                metrics["entropy"] += entropy.item()
                n_mb += 1

        return {k: v / n_mb for k, v in metrics.items()}

    # ---------------------------------------------------------- body model

    def update_body_model(self, buf: dict[str, torch.Tensor]) -> dict[str, float]:
        """One online gradient step for the body model on this rollout's fresh
        transitions (``ppo`` has already been updated for this rollout, above;
        the body model always trails one rollout behind the policy input it
        fed, which is what keeps PPO's old/new logp comparison self-consistent
        within a rollout — see ledger/body_model.py's module docstring)."""
        t, n, core_dim = buf["core_out"].shape
        h_detached = buf["core_out"].reshape(t * n, core_dim).detach()
        action_onehot = torch.nn.functional.one_hot(
            buf["action"].reshape(t * n), self.body_model.num_actions
        ).float()
        outputs = self.body_model.predict_action(h_detached, action_onehot)
        losses = compute_body_losses(
            outputs,
            buf["real_dpos_class"].reshape(t * n),
            buf["real_success"].reshape(t * n),
            buf["real_denergy"].reshape(t * n),
        )

        self.body_opt.zero_grad()
        losses["total"].backward()
        self.body_opt.step()

        body_nll = losses["body_nll"].item()
        success_brier = losses["success_brier"].item()
        denergy_mae = losses["denergy_mae"].item()
        self.body_nll_history.append(body_nll)
        return {
            "ledger/body_nll": body_nll,
            "ledger/body_nll_ema": self._body_nll_ema.update(body_nll),
            "ledger/success_bce": losses["success_bce"].item(),
            "ledger/success_brier": success_brier,
            "ledger/success_brier_ema": self._success_brier_ema.update(success_brier),
            "ledger/denergy_mse": losses["denergy_mse"].item(),
            "ledger/denergy_mae": denergy_mae,
            "ledger/denergy_mae_ema": self._denergy_mae_ema.update(denergy_mae),
        }

    # ------------------------------------------------------- attribution

    def update_attribution_model(self, buf: dict[str, torch.Tensor]) -> dict[str, float]:
        """One online gradient step for the attribution classifier, self-
        supervised from residual-magnitude thresholds (never from
        ``ground_truth_label``, which is used only afterward to score the
        prediction — see ``training/attribution_eval.py``)."""
        t, n = buf["action"].shape
        action = buf["action"].reshape(t * n)
        features = residual_features(
            buf["pred_dpos_class"].reshape(t * n),
            buf["real_dpos_class"].reshape(t * n),
            buf["pred_denergy"].reshape(t * n),
            buf["real_denergy"].reshape(t * n),
            action,
            noop_action=NOOP,
        )
        labels = pseudo_label(features, self.attr_tau_pos, self.attr_tau_energy)
        logits = self.attribution_model(features)
        loss = compute_attribution_loss(logits, labels)

        self.attr_opt.zero_grad()
        loss.backward()
        self.attr_opt.step()

        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
        ground_truth = buf["ground_truth_label"].reshape(t * n)
        for p_i, g_i, a_i in zip(
            predicted.tolist(), ground_truth.tolist(), action.tolist()
        ):
            self.attr_tracker.update(p_i, g_i, a_i)

        rollout_accuracy = (predicted == ground_truth).float().mean().item()
        self.attribution_accuracy_history.append(rollout_accuracy)
        return {
            "ledger/attribution_loss": loss.item(),
            "ledger/attribution_accuracy": rollout_accuracy,
            "ledger/attribution_accuracy_ema": self._attr_accuracy_ema.update(rollout_accuracy),
        }

    # ----------------------------------------------------------------- train

    def train(
        self,
        resume_from: str | Path | None = None,
        max_updates: int | None = None,
        allow_config_mismatch: bool = False,
    ) -> None:
        """Run updates until ppo.total_steps (or ``max_updates``)."""
        p = self.pcfg
        if resume_from is not None:
            self.load(resume_from, allow_config_mismatch=allow_config_mismatch)
            print(f"resumed from {resume_from} at step {self.global_step}")

        lr0 = p["lr"]
        updates_done = 0
        # SIGBREAK covers Windows consoles, where CTRL_C_EVENT cannot target a
        # single process group but CTRL_BREAK_EVENT can.
        signals = [signal.SIGINT]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        prev_handlers: list[tuple[int, Any]] = []
        try:
            for sig in signals:
                prev_handlers.append((sig, signal.signal(sig, self._on_sigint)))
        except ValueError:
            pass  # not in main thread (tests); signal handling skipped

        try:
            while self.global_step < p["total_steps"]:
                if max_updates is not None and updates_done >= max_updates:
                    break
                if p.get("anneal_lr"):
                    frac = 1.0 - self.global_step / p["total_steps"]
                    for group in self.opt.param_groups:
                        group["lr"] = lr0 * frac

                t0 = time.perf_counter()
                buf = self.collect_rollout()
                metrics = self.update(buf)
                metrics.update(self.update_body_model(buf))
                metrics.update(self.update_attribution_model(buf))
                if self.is_rssm and self._uncert_n:
                    metrics["rssm/epistemic"] = self._epistemic_sum / self._uncert_n
                    metrics["rssm/aleatoric"] = self._aleatoric_sum / self._uncert_n
                    if self.pr_history:
                        metrics["rssm/participation_ratio"] = self.pr_history[-1]
                elapsed = time.perf_counter() - t0

                reward_rollout = buf["reward"].sum(dim=0).mean().item()
                self.reward_history.append(reward_rollout)
                metrics["reward/rollout"] = reward_rollout
                metrics["sps"] = p["rollout_steps"] * p["num_envs"] / elapsed
                self.last_metrics = metrics
                if self.tb is not None:
                    for tag, val in metrics.items():
                        self.tb.scalar(tag, val, self.global_step)
                updates_done += 1
                if updates_done % 10 == 1:
                    print(
                        f"step {self.global_step}  reward/rollout {reward_rollout:+.3f}  "
                        f"kl {metrics['approx_kl']:.4f}  body_nll {metrics['ledger/body_nll']:.4f}  "
                        f"attr_acc {metrics['ledger/attribution_accuracy']:.3f}  "
                        f"sps {metrics['sps']:,.0f}"
                    )

                interval = self.cfg["checkpoints"]["interval"]
                if (
                    self.run_dir is not None
                    and self.global_step - self._last_ckpt_step >= interval
                ):
                    self.save()
                if self._interrupted:
                    print(f"SIGINT: checkpointing at step {self.global_step}")
                    self.save()
                    break
        finally:
            for sig, handler in prev_handlers:
                signal.signal(sig, handler)
            if self.tb is not None:
                self.tb.flush()

        completed = self.global_step >= p["total_steps"]
        if self.cfg["run"].get("assert_improvement") and completed and not self._interrupted:
            self._assert_improvement()
        if self.run_dir is not None:
            self.write_report()

    def _on_sigint(self, signum: int, frame: Any) -> None:
        del signum, frame
        self._interrupted = True

    def _assert_improvement(self, k: int = 10) -> None:
        hist = self.reward_history
        if len(hist) < 2 * k:
            raise RuntimeError(f"too few updates ({len(hist)}) to assess improvement")
        first, last = float(np.mean(hist[:k])), float(np.mean(hist[-k:]))
        print(f"reward/rollout rolling mean: first {first:+.3f} -> last {last:+.3f}")
        if last <= first:
            raise RuntimeError(
                f"reward/rollout did not improve: first {first:+.4f}, last {last:+.4f}"
            )

    # ------------------------------------------------------------- report

    def write_report(self) -> Path:
        """Markdown summary of the run so far, written to run_dir/report.md.

        Attribution accuracy is reported here against ``ground_truth_label``
        (world/levers.py's ground-truth cause log) — evaluation only; this
        method runs after training and never feeds back into any loss.
        """
        assert self.run_dir is not None
        t = self.attr_tracker
        reward_tail = float(np.mean(self.reward_history[-10:])) if self.reward_history else float("nan")
        body_nll_tail = float(np.mean(self.body_nll_history[-10:])) if self.body_nll_history else float("nan")
        n_tail = max(1, len(self.attribution_accuracy_history) // 5)  # last ~20% of rollouts
        attr_tail = (
            float(np.mean(self.attribution_accuracy_history[-n_tail:]))
            if self.attribution_accuracy_history else float("nan")
        )
        lines = [
            f"# Run report: {self.run_dir.name}",
            "",
            f"- global_step: {self.global_step}",
            f"- reward/rollout (last 10 mean): {reward_tail:+.4f}",
            f"- ledger/body_nll (last 10 mean): {body_nll_tail:.4f}",
            "",
            "## Attribution vs. ground truth (world/levers.py cause log; evaluation only)",
            "",
            f"- ticks scored (whole run): {t.total}",
            f"- accuracy (whole-run cumulative, includes early self-supervised warm-up): {t.accuracy:.4f}",
            f"- accuracy (steady-state, last {n_tail} rollouts): {attr_tail:.4f}",
            f"- noop-attributed-to-self violations (whole run): {t.noop_self_violations}",
            "",
            "Confusion matrix (rows=ground truth, cols=predicted; self/world/both; whole run):",
            "",
        ]
        for row in t.confusion:
            lines.append(f"    {row}")
        report_path = self.run_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    # ------------------------------------------------------------ checkpoint

    def save(self, path: str | Path | None = None) -> Path:
        """Checkpoint model, optimizer, step, RNG, env snapshots, hidden state."""
        if path is None:
            assert self.run_dir is not None, "no run_dir and no explicit path"
            ckpt_dir = self.run_dir / "checkpoints"
            path = ckpt_dir / f"ckpt-{self.global_step:012d}.pt"
        extra = {
            "vecenv": self.vec.get_state(),
            "hidden": self._h.detach().cpu().numpy(),
            "done_prev": self._done_prev.detach().cpu().numpy(),
            "reward_history": list(self.reward_history),
            "body_model_state": self.body_model.state_dict(),
            "body_opt_state": self.body_opt.state_dict(),
            "body_nll_history": list(self.body_nll_history),
            "body_ema": {
                "body_nll": self._body_nll_ema.value,
                "success_brier": self._success_brier_ema.value,
                "denergy_mae": self._denergy_mae_ema.value,
            },
            "attribution_model_state": self.attribution_model.state_dict(),
            "attr_opt_state": self.attr_opt.state_dict(),
            "attr_accuracy_ema": self._attr_accuracy_ema.value,
            "attr_tracker_state": self.attr_tracker.state_dict(),
            "attribution_accuracy_history": list(self.attribution_accuracy_history),
            "epistemic_map": self.epistemic_map.copy(),
            "epistemic_count": self.epistemic_count.copy(),
            "pr_monitor_state": self.pr_monitor.state_dict(),
            "pr_history": list(self.pr_history),
        }
        out = save_checkpoint(
            path, self.model, self.opt, self.global_step, self.cfg, extra=extra
        )
        self._last_ckpt_step = self.global_step
        keep = self.cfg["checkpoints"].get("keep_last", 0)
        if keep and out.parent.exists():
            prune_checkpoints(out.parent, keep)
        return out

    def load(self, path: str | Path, allow_config_mismatch: bool = False) -> None:
        """Restore a checkpoint for bit-identical continuation."""
        ckpt = load_checkpoint(
            path, self.model, self.opt, cfg=self.cfg,
            allow_config_mismatch=allow_config_mismatch,
        )
        self.global_step = ckpt.step
        self._last_ckpt_step = ckpt.step
        self.vec.set_state(ckpt.extra["vecenv"])
        self._h = torch.from_numpy(ckpt.extra["hidden"]).to(self.device)
        self._done_prev = torch.from_numpy(ckpt.extra["done_prev"]).to(self.device)
        self.reward_history = list(ckpt.extra.get("reward_history", []))
        self.body_model.load_state_dict(ckpt.extra["body_model_state"])
        self.body_opt.load_state_dict(ckpt.extra["body_opt_state"])
        self.body_nll_history = list(ckpt.extra.get("body_nll_history", []))
        ema = ckpt.extra.get("body_ema", {})
        self._body_nll_ema.value = ema.get("body_nll")
        self._success_brier_ema.value = ema.get("success_brier")
        self._denergy_mae_ema.value = ema.get("denergy_mae")
        self.attribution_model.load_state_dict(ckpt.extra["attribution_model_state"])
        self.attr_opt.load_state_dict(ckpt.extra["attr_opt_state"])
        self._attr_accuracy_ema.value = ckpt.extra.get("attr_accuracy_ema")
        if "attr_tracker_state" in ckpt.extra:
            self.attr_tracker.load_state_dict(ckpt.extra["attr_tracker_state"])
        self.attribution_accuracy_history = list(
            ckpt.extra.get("attribution_accuracy_history", [])
        )
        if "epistemic_map" in ckpt.extra:
            self.epistemic_map = np.asarray(ckpt.extra["epistemic_map"]).copy()
            self.epistemic_count = np.asarray(ckpt.extra["epistemic_count"]).copy()
        if "pr_monitor_state" in ckpt.extra:
            self.pr_monitor.load_state_dict(ckpt.extra["pr_monitor_state"])
            self.pr_history = list(ckpt.extra.get("pr_history", []))
        self._obs = self.vec.observe()
