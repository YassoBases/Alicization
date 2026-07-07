"""Circadian wake/sleep trainer (Dreamer-style, RSSM core required).

Wake: env stepping with the current policy (no gradients anywhere near the
core) plus ONLINE Ledger updates only — body model and attribution train on
detached hidden states exactly as in the PPO trainer. Every transition goes
into the prioritized sequence replay (training/replay.py).

Sleep: every ``rssm.sleep_every`` env steps, stepping pauses and
``rssm.sleep_grad_steps`` consolidation steps run, each doing:
  (a) world-model training on replay sequences (KL + reconstruction + reward
      + ensemble; priorities updated with the per-sequence loss), and
  (b) Dreamer-style imagination: roll the RSSM prior forward
      ``imagination_horizon`` steps from (detached) replay posterior states,
      then train the actor (REINFORCE on lambda-returns, entropy bonus) and
      critic (MSE to lambda-returns) with a slow-EMA target critic for
      bootstrapping.

STRUCTURAL RULE (tested): sleep scheduling reads ONLY the env step counter —
``is_sleep_tick(env_steps, sleep_every)`` takes exactly those two integers
and nothing else, so consolidation timing is exogenous and can never couple
to agent state (CLAUDE.md: no objective may reference the training process).

``rssm.sleep: false`` runs wake-only (the sleep-ablation condition): the
replay still fills and the Ledger still trains online, but no world-model,
actor, or critic gradients ever happen.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agent.core_rssm import RSSMCore
from agent.drives import NUM_PLANS, Arbiter, plan_action
from ledger.body_model import build_policy_features
from ledger.forecaster import ForecastTupleStore, Forecaster, forecaster_nll, nmse
from training.checkpoints import load_checkpoint, prune_checkpoints, save_checkpoint
from training.loggers import TBLogger
from training.monitors import ParticipationRatioMonitor
from training.ppo import PPOModel, PPOTrainer, resolve_device
from training.replay import SequenceReplay
from training.vecenv import VecWorld


def is_sleep_tick(env_steps: int, sleep_every: int) -> bool:
    """Exogenous sleep schedule: a pure function of the env step counter.

    HARD RULE: this must never take agent state, Ledger output, reward, or
    any model-derived value — only the two integers. Enforced by
    tests/test_sleep.py (signature + behavior).
    """
    return env_steps > 0 and env_steps % sleep_every == 0


def sleep_windows_due(env_steps: int, sleep_every: int) -> int:
    """How many sleep windows the schedule owes by ``env_steps`` — also a pure
    function of the counter (same HARD RULE as is_sleep_tick). The trainer
    uses this rather than an exact-modulo check because wake advances in
    rollout-sized chunks, so the counter may step OVER a multiple of
    ``sleep_every`` without ever landing on it."""
    return env_steps // sleep_every


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Dreamer lambda-returns over an imagined (H, B) trajectory.

    R_t = r_t + gamma * ((1 - lam) * V_{t+1} + lam * R_{t+1}), with
    R_H seeded by ``bootstrap`` (target-critic value of the last state).
    No done handling: imagination happens in a continuing world (episode
    boundaries are exogenous trainer-side resets, not world terminations).
    """
    horizon = rewards.shape[0]
    out = torch.zeros_like(rewards)
    running = bootstrap
    for t in reversed(range(horizon)):
        v_next = values[t + 1] if t + 1 < horizon else bootstrap
        running = rewards[t] + gamma * ((1.0 - lam) * v_next + lam * running)
        out[t] = running
    return out


class CircadianTrainer:
    """Wake/sleep training loop around an RSSM-cored PPOModel.

    Reuses PPOTrainer for its model construction, Ledger heads/optimizers,
    online Ledger updates, checkpoint plumbing, and monitors — but NOT its
    PPO update: the core/actor/critic train only during sleep, in
    imagination. Wake collects transitions with a frozen policy.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        if cfg["agent"].get("core") != "rssm":
            raise ValueError("CircadianTrainer requires agent.core: rssm")
        self.cfg = cfg
        self.rcfg = cfg["rssm"]
        self.pcfg = cfg["ppo"]
        self.device = resolve_device(device or cfg.get("device", "auto"))

        # Delegate model/Ledger/monitor construction to PPOTrainer, then take
        # over training. Its PPO optimizer (inner.opt) is deliberately never
        # stepped here.
        self._inner = PPOTrainer(cfg, run_dir=None, device=device)
        self.model: PPOModel = self._inner.model
        self.vec: VecWorld = self._inner.vec
        assert isinstance(self.model.core, RSSMCore)

        # World-model optimizer: encoder + core (decoder/ensemble/reward head
        # are core submodules). Actor-critic optimizer: policy heads only.
        wm_params = list(self.model.encoder.parameters()) + list(self.model.core.parameters())
        self.world_opt = torch.optim.Adam(wm_params, lr=self.rcfg["world_lr"])
        self.ac_opt = torch.optim.Adam(self.model.heads.parameters(), lr=self.rcfg["ac_lr"])
        self.target_critic = copy.deepcopy(self.model.heads.v)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.critic_ema_tau: float = self.rcfg.get("critic_ema_tau", 0.02)

        self.replay = SequenceReplay(
            capacity=self.rcfg["replay_capacity"],
            num_envs=self.pcfg["num_envs"],
            grid_shape=self.vec.grid_shape,
            intero_dim=self.vec.intero_dim,
            alpha=self.rcfg.get("priority_alpha", 0.6),
            seed=cfg["seed"],
        )
        self.sleep_enabled: bool = self.rcfg.get("sleep", True)
        self.env_steps = 0
        self._sleep_windows_done = 0

        # --- stage-4c: forecaster (own optimizer) + macro-plan arbiter -----
        lcfg = cfg["ledger"]
        core_dim = self.model.core.output_dim
        self.forecaster = Forecaster(
            lcfg, core_dim=core_dim, intero_dim=self.vec.intero_dim,
            num_plans=NUM_PLANS,
        ).to(self.device)
        self.fore_opt = torch.optim.Adam(self.forecaster.parameters(), lr=lcfg["lr"])
        self.horizons: tuple[int, ...] = tuple(lcfg["horizons"])
        self.tuple_store = ForecastTupleStore(
            capacity=lcfg.get("forecast_buffer", 20000), horizons=self.horizons
        )
        self.forecaster_batch: int = lcfg.get("forecaster_batch", 512)
        self.controller: str = cfg["agent"].get("controller", "actor")
        self.arbiter: Arbiter | None = None
        if self.controller == "arbiter":
            self.arbiter = Arbiter(
                lcfg.get("arbiter", {}) or {}, self.forecaster, seed=cfg["seed"]
            )
            self._plan_commit: int = (lcfg.get("arbiter", {}) or {}).get(
                "plan_commit_ticks", 10
            )
            n = self.pcfg["num_envs"]
            self._plans = np.zeros(n, dtype=np.int64)
            self._plan_age = np.full(n, 10**9)  # force selection on first tick
            self._exec_rng = np.random.default_rng(cfg["seed"] + 1)
            # (h, plan, intero_now, {horizon: remaining_ticks}) per env, plus
            # completed-horizon accumulator; futures crossing a done are dropped.
            self._pending: list[list[dict[str, Any]]] = [[] for _ in range(n)]
            n_terrain = cfg["world"]["terrain"]["num_types"]
            self._ch_food, self._ch_shelter = n_terrain, n_terrain + 2
            self._inner.action_fn = self._arbiter_action_fn
        elif self.controller != "actor":
            raise ValueError(f"unknown agent.controller: {self.controller!r}")
        self.reward_history: list[float] = []
        self.sleep_metrics_history: list[dict[str, float]] = []
        self.last_metrics: dict[str, float] = {}
        self.pr_monitor: ParticipationRatioMonitor = self._inner.pr_monitor

        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.tb: TBLogger | None = None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.tb = TBLogger(self.run_dir / "tb")

    # ------------------------------------------------------------------ wake

    def _policy_fn(self, core_features: torch.Tensor):
        feats, _ = build_policy_features(
            core_features, self._inner.body_model, self.model.use_ledger_features
        )
        return self.model.heads(feats)

    @torch.no_grad()
    def wake_phase(self, num_ticks: int) -> dict[str, float]:
        """Step the envs for ``num_ticks`` with the current (frozen) policy.

        Uses PPOTrainer.collect_rollout for stepping + online Ledger updates
        so wake behavior (hidden masking, Ledger training, epistemic map, PR
        monitor, attribution scoring) is identical to the PPO path — but the
        PPO/core update is never called. Transitions are pushed into replay.
        """
        inner = self._inner
        rollout_len = self.pcfg["rollout_steps"]
        n = self.pcfg["num_envs"]
        ticks_done = 0
        reward_sum = 0.0
        ledger_metrics: dict[str, float] = {}
        while ticks_done < num_ticks:
            buf = inner.collect_rollout()
            with torch.enable_grad():
                ledger_metrics.update(inner.update_body_model(buf))
                ledger_metrics.update(inner.update_attribution_model(buf))
            for t in range(rollout_len):
                self.replay.add_batch(
                    buf["grid"][t].cpu().numpy(),
                    buf["intero"][t].cpu().numpy(),
                    buf["action"][t].cpu().numpy(),
                    buf["reward"][t].cpu().numpy(),
                    buf["done"][t].cpu().numpy(),
                )
            reward_sum += buf["reward"].sum(dim=0).mean().item()
            ticks_done += rollout_len * n
            self.env_steps += rollout_len * n
        reward_mean = reward_sum / max(1, ticks_done // (rollout_len * n))
        self.reward_history.append(reward_mean)
        return {"reward/rollout": reward_mean, **ledger_metrics}

    # --------------------------------------------------------------- arbiter

    def _arbiter_action_fn(self, core_out: torch.Tensor, intero: torch.Tensor) -> torch.Tensor:
        """collect_rollout hook: macro-plan arbiter -> scripted plan executor.

        Also does the forecaster-tuple bookkeeping: every tick each env opens a
        (h, plan, intero_now) tuple whose horizon slots fill as the future
        arrives; tuples whose window crosses an episode boundary are dropped
        (the future on the other side belongs to a different world).
        """
        assert self.arbiter is not None
        inner = self._inner
        n = core_out.shape[0]
        h_det = core_out.detach()
        done_prev = inner._done_prev.cpu().numpy()

        # Complete / purge pending forecast tuples with the CURRENT intero.
        max_h = max(self.horizons)
        for i in range(n):
            if done_prev[i] > 0:
                self._pending[i].clear()  # future crossed an episode boundary
                self._plan_age[i] = 10**9  # force plan re-selection
                continue
            still = []
            for entry in self._pending[i]:
                entry["age"] += 1
                if entry["age"] in self.horizons:
                    entry["future"][entry["age"]] = intero[i].detach()
                if entry["age"] >= max_h:
                    self.tuple_store.add(
                        entry["h"], entry["plan"], entry["intero_now"], entry["future"]
                    )
                else:
                    still.append(entry)
            self._pending[i] = still

        # Re-select plans every plan_commit_ticks (or after a boundary).
        need = self._plan_age >= self._plan_commit
        if need.any():
            fresh = self.arbiter.select_plans(h_det)
            self._plans[need] = fresh[need]
            self._plan_age[need] = 0
        self._plan_age += 1

        # Open a new pending tuple per env for this tick.
        for i in range(n):
            self._pending[i].append({
                "h": h_det[i], "plan": int(self._plans[i]),
                "intero_now": intero[i].detach(), "age": 0, "future": {},
            })

        # Execute the committed plan on the CURRENT observation.
        grid_np = inner._obs["grid"]
        actions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            pos = None
            if inner._last_infos is not None and done_prev[i] == 0:
                pos = tuple(inner._last_infos[i]["pos"])
            actions[i] = plan_action(
                int(self._plans[i]), grid_np[i], self._ch_food, self._ch_shelter,
                self._exec_rng, epistemic_map=inner.epistemic_map, pos=pos,
            )
        return torch.from_numpy(actions)

    def _forecaster_sleep_step(self) -> dict[str, float]:
        """NLL grad steps on stored tuples + NMSE-vs-identity evaluation."""
        batch = self.tuple_store.batch(NUM_PLANS, self.device)
        if batch is None or batch["h"].shape[0] < 32:
            return {}
        total = batch["h"].shape[0]
        steps = self.rcfg["sleep_grad_steps"]
        nll_sum = 0.0
        for _ in range(steps):
            idx = torch.randperm(total, device=self.device)[: self.forecaster_batch]
            out = self.forecaster(batch["h"][idx], batch["plan"][idx])
            loss = torch.stack([
                forecaster_nll(out[k][0], out[k][1], batch["future"][k][idx])
                for k in self.horizons
            ]).sum()
            self.fore_opt.zero_grad()
            loss.backward()
            self.fore_opt.step()
            nll_sum += loss.item()
        metrics = {"sleep/forecaster_nll": nll_sum / steps}
        with torch.no_grad():
            out = self.forecaster(batch["h"], batch["plan"])
            for k in self.horizons:
                metrics[f"sleep/forecaster_nmse_k{k}"] = nmse(
                    out[k][0], batch["intero_now"], batch["future"][k]
                )
        return metrics

    # ----------------------------------------------------------------- sleep

    def sleep_phase(self) -> dict[str, float]:
        """One consolidation window: sleep_grad_steps of world-model training
        + imagination-based actor-critic training on replay sequences."""
        r = self.rcfg
        gamma = self.pcfg["gamma"]
        lam = r.get("imagination_lambda", 0.95)
        ent_coef = r.get("imagination_entropy_coef", self.pcfg["entropy_coef"])
        agg = {"sleep/wm_total": 0.0, "sleep/recon": 0.0, "sleep/kl": 0.0,
               "sleep/actor": 0.0, "sleep/critic": 0.0, "sleep/imagined_reward": 0.0}
        steps_done = 0

        for _ in range(r["sleep_grad_steps"]):
            batch = self.replay.sample(r["batch_seqs"], r["seq_len"], self.device)
            if batch is None:
                break  # not enough replay yet; sleep is a no-op this window

            # --- (a) world-model training on replay sequences
            core = self.model.core
            assert isinstance(core, RSSMCore)
            horizon, b = batch["grid"].shape[0], batch["grid"].shape[1]
            flat_grid = batch["grid"].reshape(horizon * b, *self.vec.grid_shape)
            flat_intero = batch["intero"].reshape(horizon * b, -1)
            embeds = self.model.encoder(flat_grid, flat_intero).reshape(horizon, b, -1)
            h0 = core.initial_state(b, self.device)
            wm = core.world_model_loss(
                embeds, h0, batch["done"], batch["action"],
                batch["grid"], batch["intero"], rewards=batch["reward"],
            )
            self.world_opt.zero_grad()
            wm["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.world_opt.param_groups for p in g["params"]],
                self.pcfg["max_grad_norm"],
            )
            self.world_opt.step()

            # Per-sequence recon error -> priorities (proportional).
            with torch.no_grad():
                seq_feat = wm["features"].detach()
                flat = seq_feat.reshape(horizon * b, -1)
                rec = core.decoder_grid(flat).reshape(horizon, b, *self.vec.grid_shape)
                per_seq = ((rec - batch["grid"]) ** 2).mean(dim=(0, 2, 3, 4))
            self.replay.update_priorities(
                batch["envs"], batch["starts"], r["seq_len"],
                per_seq.cpu().numpy(),
            )

            # --- (b) imagination: actor-critic on imagined lambda-returns
            start_feats = seq_feat.reshape(horizon * b, -1).detach()
            imag = core.imagine(start_feats, self._policy_fn, r["imagination_horizon"])
            feats_im = imag["features"]  # (H, B', F) — carries world-model graph
            with torch.no_grad():
                flat_im = feats_im.reshape(-1, feats_im.shape[-1])
                pf, _ = build_policy_features(
                    flat_im, self._inner.body_model, self.model.use_ledger_features
                )
                v_target = self.target_critic(pf).squeeze(-1).reshape(feats_im.shape[0], -1)
            returns = lambda_returns(
                imag["reward"].detach(), v_target, v_target[-1], gamma, lam
            )

            adv = (returns - v_target).detach()
            if self.rcfg.get("imagination_norm_adv", True) and adv.numel() > 1:
                # Raw imagined advantages are ~1e-2 (step costs); without
                # normalization the entropy bonus dominates and REINFORCE
                # never moves the policy (same failure mode as un-normalized
                # PPO advantages — DreamerV3 normalizes returns for the same
                # reason).
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor_loss = -(imag["logp"] * adv).mean() - ent_coef * imag["entropy"].mean()
            # Critic trains on detached imagined features (no world-model grads).
            pf_c, _ = build_policy_features(
                feats_im.detach().reshape(-1, feats_im.shape[-1]),
                self._inner.body_model, self.model.use_ledger_features,
            )
            v_pred = self.model.heads.v(pf_c).squeeze(-1).reshape(feats_im.shape[0], -1)
            critic_loss = F.mse_loss(v_pred, returns.detach())

            self.ac_opt.zero_grad()
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.heads.parameters(), self.pcfg["max_grad_norm"]
            )
            self.ac_opt.step()

            # Slow EMA target critic.
            with torch.no_grad():
                for tp, sp in zip(
                    self.target_critic.parameters(), self.model.heads.v.parameters()
                ):
                    tp.lerp_(sp, self.critic_ema_tau)

            agg["sleep/wm_total"] += wm["total"].item()
            agg["sleep/recon"] += wm["recon_grid"].item() + wm["recon_intero"].item()
            agg["sleep/kl"] += wm["kl"].item()
            agg["sleep/actor"] += actor_loss.item()
            agg["sleep/critic"] += critic_loss.item()
            agg["sleep/imagined_reward"] += imag["reward"].mean().item()
            steps_done += 1

        if steps_done:
            agg = {k: v / steps_done for k, v in agg.items()}
        agg["sleep/grad_steps"] = float(steps_done)
        agg.update(self._forecaster_sleep_step())
        self.sleep_metrics_history.append(agg)
        return agg

    # ----------------------------------------------------------------- train

    def train(self, max_env_steps: int | None = None) -> None:
        """Alternate wake stretches of ``sleep_every`` ticks with sleep windows
        until ppo.total_steps (or ``max_env_steps``)."""
        total = max_env_steps or self.pcfg["total_steps"]
        sleep_every = self.rcfg["sleep_every"]
        while self.env_steps < total:
            stretch = min(sleep_every, total - self.env_steps)
            t0 = time.perf_counter()
            metrics = self.wake_phase(stretch)
            if self.sleep_enabled and sleep_windows_due(self.env_steps, sleep_every) > self._sleep_windows_done:
                self._sleep_windows_done = sleep_windows_due(self.env_steps, sleep_every)
                metrics.update(self.sleep_phase())
            elapsed = time.perf_counter() - t0
            metrics["sps"] = stretch / elapsed
            self.last_metrics = metrics
            if self.tb is not None:
                for tag, val in metrics.items():
                    self.tb.scalar(tag, val, self.env_steps)
            print(
                f"env_steps {self.env_steps}  reward/rollout {metrics['reward/rollout']:+.3f}  "
                + (f"wm {metrics.get('sleep/wm_total', float('nan')):.3f}  " if self.sleep_enabled else "")
                + f"sps {metrics['sps']:,.0f}"
            )
        if self.tb is not None:
            self.tb.flush()

    # ------------------------------------------------------------ checkpoint

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            assert self.run_dir is not None, "no run_dir and no explicit path"
            path = self.run_dir / "checkpoints" / f"ckpt-{self.env_steps:012d}.pt"
        extra = {
            "vecenv": self.vec.get_state(),
            "inner_hidden": self._inner._h.detach().cpu().numpy(),
            "inner_done_prev": self._inner._done_prev.detach().cpu().numpy(),
            "env_steps": self.env_steps,
            "reward_history": list(self.reward_history),
            "body_model_state": self._inner.body_model.state_dict(),
            "body_opt_state": self._inner.body_opt.state_dict(),
            "attribution_model_state": self._inner.attribution_model.state_dict(),
            "attr_opt_state": self._inner.attr_opt.state_dict(),
            "world_opt_state": self.world_opt.state_dict(),
            "ac_opt_state": self.ac_opt.state_dict(),
            "target_critic_state": self.target_critic.state_dict(),
            "replay_state": self.replay.state_dict(),
            "epistemic_map": self._inner.epistemic_map.copy(),
            "epistemic_count": self._inner.epistemic_count.copy(),
            "pr_monitor_state": self.pr_monitor.state_dict(),
            "forecaster_state": self.forecaster.state_dict(),
            "fore_opt_state": self.fore_opt.state_dict(),
        }
        out = save_checkpoint(
            path, self.model, self.world_opt, self.env_steps, self.cfg, extra=extra
        )
        keep = self.cfg["checkpoints"].get("keep_last", 0)
        if keep and out.parent.exists():
            prune_checkpoints(out.parent, keep)
        return out

    def load(self, path: str | Path, allow_config_mismatch: bool = False) -> None:
        ckpt = load_checkpoint(
            path, self.model, self.world_opt, cfg=self.cfg,
            allow_config_mismatch=allow_config_mismatch,
        )
        e = ckpt.extra
        self.vec.set_state(e["vecenv"])
        self._inner._h = torch.from_numpy(e["inner_hidden"]).to(self.device)
        self._inner._done_prev = torch.from_numpy(e["inner_done_prev"]).to(self.device)
        self.env_steps = e["env_steps"]
        self.reward_history = list(e["reward_history"])
        self._inner.body_model.load_state_dict(e["body_model_state"])
        self._inner.body_opt.load_state_dict(e["body_opt_state"])
        self._inner.attribution_model.load_state_dict(e["attribution_model_state"])
        self._inner.attr_opt.load_state_dict(e["attr_opt_state"])
        self.ac_opt.load_state_dict(e["ac_opt_state"])
        self.target_critic.load_state_dict(e["target_critic_state"])
        self.replay.load_state_dict(e["replay_state"])
        self._inner.epistemic_map = np.asarray(e["epistemic_map"]).copy()
        self._inner.epistemic_count = np.asarray(e["epistemic_count"]).copy()
        self.pr_monitor.load_state_dict(e["pr_monitor_state"])
        if "forecaster_state" in e:
            self.forecaster.load_state_dict(e["forecaster_state"])
            self.fore_opt.load_state_dict(e["fore_opt_state"])
        self._inner._obs = self.vec.observe()


def run_sleep_phase(trainer: CircadianTrainer, cfg: dict[str, Any]) -> dict[str, float]:
    """Back-compat wrapper kept for the original stub's call signature."""
    del cfg
    return trainer.sleep_phase()
