"""
models_general.py — a GENERAL physics predictor: any bodies, any motion, any world.

The leaf model took a flat state vector of fixed length, so it could only ever
predict the world it was built for. This one consumes the universal entity view
(`action/entities.py`): a scene is a *set* of bodies, each described by the same 13
world-frame numbers plus its static properties.

Two ideas make it general:

1. **Shared per-entity weights + permutation equivariance.** Every body is processed
   by the *same* weights, and the model never uses a body's index as an identity. So
   a scene with 1 body and a scene with 40 use the same parameters, and adding a body
   is just adding a token — no retraining, no reshaping.

2. **Interactions are attention between entities.** A bounce redirecting a ball, a
   joint dragging a link, mutual gravitation — physically these are all "bodies
   influence other bodies." Attention across the entity axis learns whatever coupling
   the world has, instead of it being hardcoded in a state layout.

Architecture (factorised space-time attention, which keeps cost at
O(T*N^2 + N*T^2) instead of O((T*N)^2)):

    history (T,N,13) + attrs (N,6)
      -> embed
      -> [ temporal attention within each body | entity attention within each frame ] xL
      -> attention-pool over time
      -> per-entity memory z (N,C)

    decode: shared GRU cell per entity, with one entity-attention layer each step so
    bodies keep influencing each other during the rollout -> per-entity delta (N,13).

Positions are fed relative to the scene centroid (translation invariance) and
quaternions are renormalised every step so rotations stay on the unit sphere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from action.entities import ENTITY_DIM, ATTR_DIM


class _Attn(nn.Module):
    """Pre-norm multi-head self-attention + MLP, applied over one chosen axis."""

    def __init__(self, d, nhead, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.att(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class EntityEncoder(nn.Module):
    """(B,T,N,13) history + (B,N,6) attrs -> (B,N,C) per-entity memory."""

    def __init__(self, d_model=96, nhead=4, layers=3, context_dim=96,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(ENTITY_DIM + ATTR_DIM, d_model)
        self.tpos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.time_blocks = nn.ModuleList([_Attn(d_model, nhead, dropout) for _ in range(layers)])
        self.ent_blocks = nn.ModuleList([_Attn(d_model, nhead, dropout) for _ in range(layers)])
        self.pool_score = nn.Linear(d_model, 1)
        self.to_ctx = nn.Linear(d_model, context_dim)
        self.context_dim = context_dim

    def forward(self, hist, attrs, time_mask=None):
        """hist:(B,T,N,13) normalized, attrs:(B,N,6), time_mask:(B,T) True=pad."""
        B, T, N, _ = hist.shape
        x = torch.cat([hist, attrs.unsqueeze(1).expand(B, T, N, ATTR_DIM)], dim=-1)
        x = self.inp(x) + self.tpos[:, :T].unsqueeze(2)          # (B,T,N,d)

        for tb, eb in zip(self.time_blocks, self.ent_blocks):
            # --- attention over TIME, independently per entity ---
            xt = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            tm = None if time_mask is None else \
                time_mask.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
            xt = tb(xt, key_padding_mask=tm)
            x = xt.reshape(B, N, T, -1).permute(0, 2, 1, 3)
            # --- attention over ENTITIES, independently per frame (interactions) ---
            xe = x.reshape(B * T, N, -1)
            xe = eb(xe)
            x = xe.reshape(B, T, N, -1)

        # attention-pool over time -> one memory vector per entity
        s = self.pool_score(x).squeeze(-1)                        # (B,T,N)
        if time_mask is not None:
            s = s.masked_fill(time_mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)                 # (B,T,N,1)
        return self.to_ctx((x * w).sum(dim=1))                    # (B,N,C)


class GeneralPredictor(nn.Module):
    """Entity memory encoder + interaction-aware autoregressive rollout decoder."""

    def __init__(self, d_model=96, nhead=4, enc_layers=3, context_dim=96,
                 dec_hidden=192, n_time_freq=12, dropout=0.1):
        super().__init__()
        self.encoder = EntityEncoder(d_model, nhead, enc_layers, context_dim, dropout=dropout)
        self.h0 = nn.Linear(context_dim, dec_hidden)
        self.n_time_freq = n_time_freq
        freqs = torch.exp(torch.linspace(math.log(0.008), math.log(0.6), n_time_freq))
        self.register_buffer("time_freqs", freqs)
        in_dim = ENTITY_DIM + ATTR_DIM + context_dim + 2 * n_time_freq
        self.cell = nn.GRUCell(in_dim, dec_hidden)
        self.interact = _Attn(dec_hidden, nhead, dropout)      # bodies talk each step
        self.out = nn.Linear(dec_hidden, ENTITY_DIM)
        self.context_dim, self.dec_hidden = context_dim, dec_hidden

    def encode(self, hist, attrs, time_mask=None):
        return self.encoder(hist, attrs, time_mask)

    def _step(self, h, state, z, attrs, j, ref, fmean, fstd, dmean, dstd):
        B, N, _ = state.shape
        rel = torch.cat([state[..., 0:3] - ref.unsqueeze(1), state[..., 3:]], dim=-1)
        feat = (rel - fmean) / fstd
        ang = self.time_freqs * float(j)
        tf = torch.cat([torch.sin(ang), torch.cos(ang)])
        tf = tf.view(1, 1, -1).expand(B, N, -1)
        inp = torch.cat([feat, attrs, z, tf], dim=-1).reshape(B * N, -1)
        h = self.cell(inp, h)
        h = self.interact(h.view(B, N, -1)).reshape(B * N, -1)   # entities interact
        d = self.out(h).view(B, N, ENTITY_DIM)
        state = state + d * dstd + dmean
        q = state[..., 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        state = torch.cat([state[..., 0:3], q, state[..., 7:]], dim=-1)
        return h, state, d

    def decode(self, z, cur, attrs, n_steps, fmean, fstd, dmean, dstd, ckpt_chunk=0):
        """z:(B,N,C) cur:(B,N,13) attrs:(B,N,6) -> deltas (B,S,N,13), states (B,S,N,13)."""
        B, N, _ = cur.shape
        h = torch.tanh(self.h0(z)).reshape(B * N, -1)
        ref = cur[..., 0:3].mean(dim=1)                          # scene centroid
        state = cur

        def run(h, state, z, attrs, start, count, ref):
            ds, ss = [], []
            for j in range(start, start + count):
                h, state, d = self._step(h, state, z, attrs, j, ref,
                                         fmean, fstd, dmean, dstd)
                ds.append(d); ss.append(state)
            return h, state, torch.stack(ds, 1), torch.stack(ss, 1)

        if not (ckpt_chunk and self.training):
            _, _, d, s = run(h, state, z, attrs, 0, n_steps, ref)
            return d, s
        import torch.utils.checkpoint as cp
        das, sas, j = [], [], 0
        while j < n_steps:
            c = min(ckpt_chunk, n_steps - j)
            h, state, dc, sc = cp.checkpoint(run, h, state, z, attrs, j, c, ref,
                                             use_reentrant=False)
            das.append(dc); sas.append(sc); j += c
        return torch.cat(das, 1), torch.cat(sas, 1)
