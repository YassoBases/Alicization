"""Persistent recurrent core (RSSM variant) — drop-in alternative to GRUCore,
selected by ``agent.core: rssm`` (default gru).

State layout: the externally visible hidden state is a single flat
``(B, deter + stoch)`` tensor (deterministic GRU path first, Gaussian
stochastic latent second), so the rollout buffer, BPTT replay, and checkpoint
format never need to know about RSSM structure. ``output_dim`` (deter + stoch)
feeds the policy/value heads and (detached) the Ledger, exactly like GRUCore's
output.

``forward`` (the policy/collection path) uses the posterior MEAN for the
stochastic half — deterministic, so PPO's old/new log-prob replay stays exact
and seeded runs stay reproducible. Sampling (reparameterized) happens only
inside ``observe_sequence``/``world_model_loss``, the world-model training
path (Core trains only on world-prediction + task loss — CLAUDE.md).

World-model loss = KL-balanced prior/posterior KL (free-nats floor) +
reconstruction of the egocentric grid + intero vector (+ optional reward
head, used by the sleep trainer). An ensemble of K small dynamics heads
predicts the next observation embedding from (deter, stoch, action):
disagreement (variance of head means) is the epistemic signal; each head's
predicted variance is the aleatoric signal.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, out_dim))


class RSSMCore(nn.Module):
    """Deterministic GRU path + Gaussian stochastic latent, with decoder,
    reward head, and a K-head dynamics ensemble for uncertainty estimates."""

    def __init__(
        self,
        cfg: dict[str, Any],
        input_dim: int,
        grid_shape: tuple[int, int, int],
        intero_dim: int,
        num_actions: int,
    ) -> None:
        """``cfg`` is the top-level ``rssm`` config section."""
        super().__init__()
        self.deter: int = cfg["deter"]
        self.stoch: int = cfg["stoch"]
        self.input_dim = input_dim
        self.hidden_dim: int = self.deter + self.stoch  # flat state size
        self.output_dim: int = self.deter + self.stoch
        self.min_std: float = cfg.get("min_std", 0.1)
        self.free_nats: float = cfg.get("free_nats", 1.0)
        self.kl_balance: float = cfg.get("kl_balance", 0.8)
        self.kl_scale: float = cfg.get("kl_scale", 1.0)
        self.recon_scale: float = cfg.get("recon_scale", 1.0)
        self.ensemble_scale: float = cfg.get("ensemble_scale", 1.0)
        self.reward_scale: float = cfg.get("reward_scale", 1.0)
        self.grid_shape = grid_shape
        self.intero_dim = intero_dim
        self.num_actions = num_actions
        hidden = cfg.get("mlp_hidden", cfg["embed"])

        self._pre = nn.Linear(self.stoch + input_dim, self.deter)
        self.gru = nn.GRUCell(self.deter, self.deter)
        self.prior_net = _mlp(self.deter, hidden, 2 * self.stoch)
        self.post_net = _mlp(self.deter + input_dim, hidden, 2 * self.stoch)

        grid_flat = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.decoder_grid = _mlp(self.output_dim, hidden, grid_flat)
        self.decoder_intero = _mlp(self.output_dim, hidden, intero_dim)
        self.reward_head = _mlp(self.output_dim, hidden, 1)

        k = cfg.get("ensemble_k", 4)
        self.ensemble = nn.ModuleList(
            _mlp(self.output_dim + num_actions, hidden, 2 * input_dim) for _ in range(k)
        )

    # ------------------------------------------------------------- plumbing

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero state of shape (B, deter + stoch)."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _split(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return h[:, : self.deter], h[:, self.deter :]

    def _stats(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw net output -> (mean, std) with a softplus floor (NaN safety)."""
        mean, raw_std = raw.chunk(2, dim=-1)
        std = self.min_std + F.softplus(raw_std)
        return mean, std

    def _step_deter(self, embed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        deter_prev, stoch_prev = self._split(h)
        x = F.elu(self._pre(torch.cat([stoch_prev, embed], dim=-1)))
        return self.gru(x, deter_prev)

    # ------------------------------------------------- GRUCore-compatible API

    def forward(
        self, embed: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: (B, input_dim), (B, deter+stoch) -> (out, next state).

        Posterior mean path (deterministic); sampling lives only in the
        world-model training methods below.
        """
        deter = self._step_deter(embed, h)
        post_mean, _ = self._stats(self.post_net(torch.cat([deter, embed], dim=-1)))
        h_next = torch.cat([deter, post_mean], dim=-1)
        return h_next, h_next

    # ----------------------------------------------------------- world model

    def observe_sequence(
        self, embeds: torch.Tensor, h0: torch.Tensor, dones: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Roll the posterior (reparameterized sampling) over a segment.

        ``embeds`` (T, B, E), ``h0`` (B, deter+stoch) already masked for the
        first step, ``dones`` (T, B) — state zeroed after a done, mirroring
        training.ppo.replay_core. Returns features (T, B, deter+stoch) and
        per-step prior/posterior stats.
        """
        horizon = embeds.shape[0]
        h = h0
        feats, pr_m, pr_s, po_m, po_s = [], [], [], [], []
        for t in range(horizon):
            deter = self._step_deter(embeds[t], h)
            prior_mean, prior_std = self._stats(self.prior_net(deter))
            post_mean, post_std = self._stats(
                self.post_net(torch.cat([deter, embeds[t]], dim=-1))
            )
            stoch = post_mean + post_std * torch.randn_like(post_std)
            h = torch.cat([deter, stoch], dim=-1)
            feats.append(h)
            pr_m.append(prior_mean); pr_s.append(prior_std)
            po_m.append(post_mean); po_s.append(post_std)
            h = h * (1.0 - dones[t]).unsqueeze(-1)
        return {
            "features": torch.stack(feats),
            "prior_mean": torch.stack(pr_m), "prior_std": torch.stack(pr_s),
            "post_mean": torch.stack(po_m), "post_std": torch.stack(po_s),
        }

    @staticmethod
    def _kl_diag_gauss(
        m1: torch.Tensor, s1: torch.Tensor, m2: torch.Tensor, s2: torch.Tensor
    ) -> torch.Tensor:
        """KL(N(m1,s1) || N(m2,s2)) for diagonal Gaussians, summed over dims."""
        return (
            torch.log(s2 / s1) + (s1.pow(2) + (m1 - m2).pow(2)) / (2.0 * s2.pow(2)) - 0.5
        ).sum(dim=-1)

    def kl_loss(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        """KL-balanced loss with a free-nats floor (applied to each side)."""
        detach = lambda x: x.detach()  # noqa: E731
        kl_prior = self._kl_diag_gauss(  # trains the prior toward the posterior
            detach(seq["post_mean"]), detach(seq["post_std"]),
            seq["prior_mean"], seq["prior_std"],
        ).mean()
        kl_post = self._kl_diag_gauss(  # regularizes the posterior toward the prior
            seq["post_mean"], seq["post_std"],
            detach(seq["prior_mean"]), detach(seq["prior_std"]),
        ).mean()
        free = torch.tensor(self.free_nats, device=kl_prior.device)
        return self.kl_balance * torch.maximum(kl_prior, free) + (
            1.0 - self.kl_balance
        ) * torch.maximum(kl_post, free)

    def ensemble_stats(
        self, features: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, F), (B, A) -> (means (K, B, E), epistemic (B,), aleatoric (B,)).

        Epistemic = variance across head means (averaged over embed dims);
        aleatoric = mean predicted variance across heads and dims.
        """
        x = torch.cat([features, action_onehot], dim=-1)
        means, variances = [], []
        for head in self.ensemble:
            mean, raw_std = head(x).chunk(2, dim=-1)
            std = self.min_std + F.softplus(raw_std)
            means.append(mean)
            variances.append(std.pow(2))
        means_t = torch.stack(means)  # (K, B, E)
        epistemic = means_t.var(dim=0, unbiased=False).mean(dim=-1)
        aleatoric = torch.stack(variances).mean(dim=(0, 2))
        return means_t, epistemic, aleatoric

    def world_model_loss(
        self,
        embeds: torch.Tensor,
        h0: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
        grid_target: torch.Tensor,
        intero_target: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full world-prediction loss over a (T, B) segment.

        ``grid_target`` (T, B, C, W, W), ``intero_target`` (T, B, D),
        ``actions`` (T, B) long, ``rewards`` (T, B) optional. Ensemble targets
        are the next step's embedding (detached; masked where a done breaks
        the transition).
        """
        horizon, batch = embeds.shape[0], embeds.shape[1]
        seq = self.observe_sequence(embeds, h0, dones)
        feats = seq["features"]

        kl = self.kl_loss(seq)
        flat = feats.reshape(horizon * batch, -1)
        recon_grid = self.decoder_grid(flat).reshape(horizon, batch, *self.grid_shape)
        recon_intero = self.decoder_intero(flat).reshape(horizon, batch, -1)
        loss_grid = F.mse_loss(recon_grid, grid_target)
        loss_intero = F.mse_loss(recon_intero, intero_target)

        loss_reward = torch.zeros((), device=embeds.device)
        if rewards is not None:
            pred_r = self.reward_head(flat).reshape(horizon, batch)
            loss_reward = F.mse_loss(pred_r, rewards)

        # Ensemble NLL: state_t + action_t -> embed_{t+1}; a done at t breaks it.
        loss_ens = torch.zeros((), device=embeds.device)
        if horizon > 1:
            src = feats[:-1].reshape((horizon - 1) * batch, -1).detach()
            act = F.one_hot(actions[:-1].reshape(-1), self.num_actions).float()
            target = embeds[1:].reshape((horizon - 1) * batch, -1).detach()
            valid = (1.0 - dones[:-1].reshape(-1))
            x = torch.cat([src, act], dim=-1)
            nlls = []
            for head in self.ensemble:
                mean, raw_std = head(x).chunk(2, dim=-1)
                std = self.min_std + F.softplus(raw_std)
                nll = 0.5 * (((target - mean) / std).pow(2) + 2.0 * torch.log(std)).sum(-1)
                nlls.append(nll)
            denom = valid.sum().clamp(min=1.0)
            loss_ens = (torch.stack(nlls).mean(dim=0) * valid).sum() / denom

        total = (
            self.kl_scale * kl
            + self.recon_scale * (loss_grid + loss_intero)
            + self.ensemble_scale * loss_ens
            + self.reward_scale * loss_reward
        )
        return {
            "total": total, "kl": kl, "recon_grid": loss_grid,
            "recon_intero": loss_intero, "ensemble_nll": loss_ens,
            "reward_mse": loss_reward, "features": feats,
        }

    # ------------------------------------------------------------ imagination

    def imagine(
        self, features: torch.Tensor, policy_fn: Any, horizon: int
    ) -> dict[str, torch.Tensor]:
        """Roll the PRIOR forward ``horizon`` steps from (detached) start
        features, sampling actions from the policy on imagined features.

        ``policy_fn`` is a callable ``(core_features) -> (dist, value)`` — the
        sleep trainer passes a wrapper that applies build_policy_features
        first, so the actor sees the same input layout as in wake. Returns
        imagined features (H, B, F), sampled actions + log-probs + entropies
        (H, B), and predicted rewards (H, B). The dynamics input embedding for
        the next step is approximated by the mean of the ensemble heads'
        predictions.
        """
        h = features.detach()
        feats, acts, logps, ents, rews = [], [], [], [], []
        for _ in range(horizon):
            dist, _ = policy_fn(h)
            action = dist.sample()
            onehot = F.one_hot(action, self.num_actions).float()
            means_t, _, _ = self.ensemble_stats(h, onehot)
            embed_pred = means_t.mean(dim=0)
            deter = self._step_deter(embed_pred, h)
            prior_mean, prior_std = self._stats(self.prior_net(deter))
            stoch = prior_mean + prior_std * torch.randn_like(prior_std)
            h = torch.cat([deter, stoch], dim=-1)
            feats.append(h)
            acts.append(action)
            logps.append(dist.log_prob(action))
            ents.append(dist.entropy())
            rews.append(self.reward_head(h).squeeze(-1))
        return {
            "features": torch.stack(feats), "action": torch.stack(acts),
            "logp": torch.stack(logps), "entropy": torch.stack(ents),
            "reward": torch.stack(rews),
        }
