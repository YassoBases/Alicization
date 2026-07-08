"""RSSM-backed ModelAdapter for the EIG ranker (researcher/eig.py).

This module is the torch side of the adapter seam: researcher/ may import
neither torch nor agent/training (structural rule, enforced by
tests/test_proposals_no_execution.py), so it defines a duck-typed
``ModelAdapter`` protocol and THIS module implements it against a live
CircadianTrainer. The battery/harness builds the adapter and injects it.

APPROXIMATIONS (documented per the stage-8c spec; see also the
researcher/eig.py module docstring):

``region_disagreement(region)`` — the mean of the trainer's position-
bucketed epistemic map (running mean of ensemble disagreement at visited
cells) over the region's block. Cells never visited contribute 0.

``imagined_visit_reduction(region, steps)`` — Plan2Explore-style: start
from replay states whose (normalized) positions fall inside the region,
roll the RSSM prior ``steps`` steps under a uniform random policy, and
measure ensemble epistemic variance vs predicted aleatoric variance along
the imagined trajectory. The expected fractional reduction from actually
visiting is approximated by the REDUCIBLE FRACTION

    epistemic / (epistemic + aleatoric)

— disagreement between heads is what more data can shrink; predicted
observation noise is what it cannot. An irreducibly random region (noisy
TV) has high aleatoric variance and scores near 0 no matter how large its
raw disagreement. This does not simulate the future training update
itself (that would need imagined gradient steps); it is a first-order
learnability estimate, and predicted_gain is logged per executed item so
the researcher-value battery can score it against realized reductions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from training.sleep import CircadianTrainer


class RSSMAdapter:
    """Implements the researcher/eig.py ModelAdapter protocol."""

    def __init__(
        self,
        trainer: CircadianTrainer,
        region_size: int = 8,
        context_len: int = 8,
        max_starts: int = 64,
        seed: int = 0,
    ) -> None:
        self.trainer = trainer
        self.region_size = region_size
        self.context_len = context_len
        self.max_starts = max_starts
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ protocol

    def region_disagreement(self, region: tuple[int, int]) -> float:
        emap = self.trainer._inner.epistemic_map  # (size, size), [y, x]
        r, c = int(region[0]), int(region[1])
        s = self.region_size
        block = emap[r * s:(r + 1) * s, c * s:(c + 1) * s]
        return float(block.mean()) if block.size else 0.0

    def imagined_visit_reduction(self, region: tuple[int, int],
                                 steps: int) -> float:
        feats = self._region_start_features(region)
        if feats is None:
            return 0.0  # nothing known about the region: no basis to predict
        core = self.trainer.model.core
        num_actions = core.num_actions
        epi_sum, ale_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            h = feats
            for _ in range(int(steps)):
                # Uniform random policy: visit dynamics, not reward-seeking.
                action = torch.randint(
                    num_actions, (h.shape[0],), device=h.device,
                    generator=self._torch_gen(h.device),
                )
                onehot = F.one_hot(action, num_actions).float()
                means_t, epistemic, aleatoric = core.ensemble_stats(h, onehot)
                epi_sum += float(epistemic.mean())
                ale_sum += float(aleatoric.mean())
                n += 1
                # Advance the prior exactly as core.imagine does.
                embed_pred = means_t.mean(dim=0)
                deter = core._step_deter(embed_pred, h)
                prior_mean, prior_std = core._stats(core.prior_net(deter))
                stoch = prior_mean + prior_std * torch.randn_like(prior_std)
                h = torch.cat([deter, stoch], dim=-1)
        if n == 0:
            return 0.0
        epi, ale = epi_sum / n, ale_sum / n
        return float(epi / (epi + ale + 1e-12))

    # ------------------------------------------------------------ internals

    def _torch_gen(self, device: torch.device) -> torch.Generator:
        if not hasattr(self, "_gen"):
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(int(self.rng.integers(2**31)))
        return self._gen

    def _region_start_features(self, region: tuple[int, int]) -> torch.Tensor | None:
        """Encode replay contexts ending inside the region -> start features
        (B, deter+stoch) for imagination. None if the region is unvisited in
        replay (or replay is still too small for a context)."""
        t = self.trainer
        replay, ctx = t.replay, self.context_len
        if replay._filled <= ctx:
            return None
        size = t._inner.epistemic_map.shape[0]
        r, c = int(region[0]), int(region[1])
        s = self.region_size

        # Candidate context END indices per env whose position is in-region.
        # replay.position is normalized (x, y); region indexing is (row=y//s,
        # col=x//s), matching the epistemic map and questions.py.
        pos = replay.position[:, :replay._filled]           # (E, T, 2)
        cells = np.clip((pos * size).astype(int), 0, size - 1)
        in_region = (cells[..., 1] // s == r) & (cells[..., 0] // s == c)
        envs, ends = np.nonzero(in_region)
        valid = ends >= ctx
        envs, ends = envs[valid], ends[valid]
        if len(envs) == 0:
            return None
        if len(envs) > self.max_starts:
            pick = self.rng.choice(len(envs), self.max_starts, replace=False)
            envs, ends = envs[pick], ends[pick]

        # Gather (ctx, B, ...) windows and roll the posterior over them.
        idx = ends[None, :] - np.arange(ctx - 1, -1, -1)[:, None]  # (ctx, B)
        grid = torch.as_tensor(replay.grid[envs[None, :], idx],
                               dtype=torch.float32, device=t.device)
        intero = torch.as_tensor(replay.intero[envs[None, :], idx],
                                 dtype=torch.float32, device=t.device)
        done = torch.as_tensor(replay.done[envs[None, :], idx],
                               dtype=torch.float32, device=t.device)
        with torch.no_grad():
            flat = grid.reshape(ctx * len(envs), *t.vec.grid_shape)
            flat_i = intero.reshape(ctx * len(envs), -1)
            embeds = t.model.encoder(flat, flat_i).reshape(ctx, len(envs), -1)
            core = t.model.core
            h0 = core.initial_state(len(envs), t.device)
            seq = core.observe_sequence(embeds, h0, done)
        return seq["features"][-1]
