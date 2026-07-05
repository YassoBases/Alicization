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
from agent.encoder import ObsEncoder
from agent.policy import ActorCritic
from training.checkpoints import load_checkpoint, prune_checkpoints, save_checkpoint
from training.loggers import TBLogger
from training.vecenv import VecWorld
from world.engine import NUM_ACTIONS


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
    core: GRUCore, embeds: torch.Tensor, h0: torch.Tensor, dones: torch.Tensor
) -> torch.Tensor:
    """Replay a segment through the core with done masking.

    ``embeds`` (T, B, E), ``h0`` (B, H) — the already-masked input hidden for
    the first step — and ``dones`` (T, B). The hidden state is zeroed AFTER any
    step where done=1, i.e. before it is used as the next step's input.
    Returns per-step core outputs (T, B, H).
    """
    h = h0
    outputs = []
    for t in range(embeds.shape[0]):
        out, h = core(embeds[t], h)
        outputs.append(out)
        h = h * (1.0 - dones[t]).unsqueeze(-1)
    return torch.stack(outputs)


class PPOModel(nn.Module):
    """Encoder + recurrent core + actor-critic heads as one checkpointable module."""

    def __init__(
        self, cfg: dict[str, Any], grid_channels: int, intero_dim: int, window: int
    ) -> None:
        super().__init__()
        mcfg = cfg["model"]
        self.encoder = ObsEncoder(mcfg, grid_channels, intero_dim, window)
        self.core = GRUCore(mcfg, mcfg["obs_embed_dim"])
        self.heads = ActorCritic(mcfg, mcfg["core_hidden"], NUM_ACTIONS)


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

        n, hidden = self.pcfg["num_envs"], cfg["model"]["core_hidden"]
        self._h = torch.zeros(n, hidden, device=self.device)
        self._done_prev = torch.zeros(n, device=self.device)
        self._obs = self.vec.observe()

        self.global_step = 0
        self._last_ckpt_step = 0
        self.reward_history: list[float] = []
        self.last_metrics: dict[str, float] = {}
        self._interrupted = False

        seq = self.pcfg["seq_len"]
        if self.pcfg["rollout_length"] % seq != 0:
            raise ValueError("ppo.rollout_length must be a multiple of ppo.seq_len")
        samples = (self.pcfg["rollout_length"] // seq) * n
        if samples % self.pcfg["num_minibatches"] != 0:
            raise ValueError(
                f"{samples} BPTT sequences per rollout not divisible by "
                f"num_minibatches={self.pcfg['num_minibatches']}"
            )

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
        """Collect one on-policy rollout of shape (T, N, ...)."""
        p = self.pcfg
        rollout_len, n, seq = p["rollout_length"], p["num_envs"], p["seq_len"]
        c, w, _ = self.vec.grid_shape
        dev = self.device
        buf = {
            "grid": torch.zeros(rollout_len, n, c, w, w, device=dev),
            "intero": torch.zeros(rollout_len, n, self.vec.intero_dim, device=dev),
            "action": torch.zeros(rollout_len, n, dtype=torch.long, device=dev),
            "logp": torch.zeros(rollout_len, n, device=dev),
            "value": torch.zeros(rollout_len, n, device=dev),
            "reward": torch.zeros(rollout_len, n, device=dev),
            "done": torch.zeros(rollout_len, n, device=dev),
            "h_init": torch.zeros(rollout_len // seq, n, self._h.shape[1], device=dev),
        }
        for t in range(rollout_len):
            h_in = self._h * (1.0 - self._done_prev).unsqueeze(-1)
            if t % seq == 0:
                buf["h_init"][t // seq] = h_in
            grid, intero = self._obs_tensors()
            embed = self.model.encoder(grid, intero)
            out, h_new = self.model.core(embed, h_in)
            dist, value = self.model.heads(out)
            action = dist.sample()

            obs, rewards, dones, _ = self.vec.step(action.cpu().numpy())
            buf["grid"][t] = grid
            buf["intero"][t] = intero
            buf["action"][t] = action
            buf["logp"][t] = dist.log_prob(action)
            buf["value"][t] = value
            buf["reward"][t] = torch.from_numpy(rewards).to(dev)
            buf["done"][t] = torch.from_numpy(dones).to(dev)
            self._h = h_new
            self._done_prev = buf["done"][t]
            self._obs = obs

        # Bootstrap value for GAE (masked hidden; unused across dones anyway).
        h_in = self._h * (1.0 - self._done_prev).unsqueeze(-1)
        grid, intero = self._obs_tensors()
        out, _ = self.model.core(self.model.encoder(grid, intero), h_in)
        _, next_value = self.model.heads(out)
        buf["next_value"] = next_value

        self.global_step += rollout_len * n
        return buf

    # --------------------------------------------------------------- updates

    def update(self, buf: dict[str, torch.Tensor]) -> dict[str, float]:
        """PPO epochs over BPTT segments; returns mean scalar metrics."""
        p = self.pcfg
        rollout_len, n, seq = p["rollout_length"], p["num_envs"], p["seq_len"]
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
               ("grid", "intero", "action", "logp", "done")}
        seg_adv, seg_ret, seg_val = by_segment(adv), by_segment(returns), by_segment(buf["value"])

        n_samples = n_seg * n
        mb_size = n_samples // p["num_minibatches"]
        amp = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if p.get("amp_bf16")
            else contextlib.nullcontext()
        )
        metrics = {k: 0.0 for k in
                   ("loss/policy", "loss/value", "loss/total", "entropy",
                    "approx_kl", "clip_frac")}
        n_mb = 0

        for _ in range(p["update_epochs"]):
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
                    dist, value = self.model.heads(outs.reshape(seq * m, -1))
                    # (seq, M) -> flat, matching outs layout
                    flat_action = mb["action"].reshape(-1)
                    new_logp = dist.log_prob(flat_action)
                    entropy = dist.entropy().mean()

                    logratio = new_logp - mb["logp"].reshape(-1)
                    ratio = logratio.exp()
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * ratio.clamp(
                        1.0 - p["clip_range"], 1.0 + p["clip_range"]
                    )
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

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()

                with torch.no_grad():
                    metrics["approx_kl"] += ((ratio - 1.0) - logratio).mean().item()
                    metrics["clip_frac"] += (
                        ((ratio - 1.0).abs() > p["clip_range"]).float().mean().item()
                    )
                metrics["loss/policy"] += policy_loss.item()
                metrics["loss/value"] += value_loss.item()
                metrics["loss/total"] += loss.item()
                metrics["entropy"] += entropy.item()
                n_mb += 1

        return {k: v / n_mb for k, v in metrics.items()}

    # ----------------------------------------------------------------- train

    def train(
        self,
        resume_from: str | Path | None = None,
        max_updates: int | None = None,
        allow_config_mismatch: bool = False,
    ) -> None:
        """Run updates until ppo.total_env_steps (or ``max_updates``)."""
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
            while self.global_step < p["total_env_steps"]:
                if max_updates is not None and updates_done >= max_updates:
                    break
                if p.get("anneal_lr"):
                    frac = 1.0 - self.global_step / p["total_env_steps"]
                    for group in self.opt.param_groups:
                        group["lr"] = lr0 * frac

                t0 = time.perf_counter()
                buf = self.collect_rollout()
                metrics = self.update(buf)
                elapsed = time.perf_counter() - t0

                reward_rollout = buf["reward"].sum(dim=0).mean().item()
                self.reward_history.append(reward_rollout)
                metrics["reward/rollout"] = reward_rollout
                metrics["sps"] = p["rollout_length"] * p["num_envs"] / elapsed
                self.last_metrics = metrics
                if self.tb is not None:
                    for tag, val in metrics.items():
                        self.tb.scalar(tag, val, self.global_step)
                updates_done += 1
                if updates_done % 10 == 1:
                    print(
                        f"step {self.global_step}  reward/rollout {reward_rollout:+.3f}  "
                        f"kl {metrics['approx_kl']:.4f}  sps {metrics['sps']:,.0f}"
                    )

                interval = self.cfg["run"]["checkpoint_every"]
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

        completed = self.global_step >= p["total_env_steps"]
        if self.cfg["run"].get("assert_improvement") and completed and not self._interrupted:
            self._assert_improvement()

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
        }
        out = save_checkpoint(
            path, self.model, self.opt, self.global_step, self.cfg, extra=extra
        )
        self._last_ckpt_step = self.global_step
        keep = self.cfg["run"].get("keep_last", 0)
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
        self._obs = self.vec.observe()
