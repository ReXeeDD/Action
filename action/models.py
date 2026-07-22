"""
models.py — the predictor networks.

Phase 2 (now): a plain deterministic MLP that maps a history window to the
next-step state delta. This is the baseline the fancier nets must beat.

Phases 3-4 (later, stubbed in spirit): a probabilistic head for the
"cone of futures", and a Lagrangian/Hamiltonian net that learns the *action*.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from action.leaf_world import STATE_DIM  # 13


class MLPPredictor(nn.Module):
    """history features -> normalized next-step delta (13-dim)."""

    def __init__(self, history: int = 6, hidden=(256, 256, 256), state_dim: int = STATE_DIM):
        super().__init__()
        in_dim = history * state_dim
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, state_dim)]
        self.net = nn.Sequential(*layers)
        self.history = history
        self.state_dim = state_dim

    def forward(self, x):
        return self.net(x)


class GaussianMLP(nn.Module):
    """history features -> a Gaussian over the next-step delta: (mean, log_std).

    This is what turns one predicted line into a *distribution* of futures. The
    network learns not just where the leaf goes next, but how uncertain that is —
    and sampling from it, step after step, is how the cone of futures grows.
    """

    LOGSTD_MIN, LOGSTD_MAX = -7.0, 3.0

    def __init__(self, history: int = 6, hidden=(256, 256, 256), state_dim: int = STATE_DIM):
        super().__init__()
        in_dim = history * state_dim
        trunk, d = [], in_dim
        for h in hidden:
            trunk += [nn.Linear(d, h), nn.GELU()]
            d = h
        self.trunk = nn.Sequential(*trunk)
        self.head_mean = nn.Linear(d, state_dim)
        self.head_logstd = nn.Linear(d, state_dim)
        self.history = history
        self.state_dim = state_dim

    def forward(self, x):
        z = self.trunk(x)
        mean = self.head_mean(z)
        log_std = torch.clamp(self.head_logstd(z), self.LOGSTD_MIN, self.LOGSTD_MAX)
        return mean, log_std

    def nll(self, x, target):
        """Gaussian negative log-likelihood of `target` (normalized delta space)."""
        mean, log_std = self(x)
        inv_var = torch.exp(-2.0 * log_std)
        return (0.5 * ((target - mean) ** 2 * inv_var) + log_std).sum(-1).mean()


# ---------------------------------------------------------------------------
# Phase 5 — the MEMORY model.
#
# The user's idea: don't feed the net the raw physics (mass, wind, temp). Instead
# let it *watch* a long stretch of the fall and distill its own compact "memory"
# of how THIS object interacts with THIS environment — the effective mass, drag,
# wind, and wobble character — then read that memory at every step of a long
# prediction. A steel ball's memory would say "falls straight"; a leaf's says
# "wobbles like this in air like that."
#
# TrajContextEncoder = attention over the history -> a context vector z (the memory).
# SeqPredictor       = z + a GRU decoder that rolls the whole future forward,
#                      reading z at every step. Trained with a multi-step rollout
#                      loss, which also removes the single-step compounding blow-up.
# ---------------------------------------------------------------------------
class TrajContextEncoder(nn.Module):
    """A long history of frames -> a context vector z (the 'memory').

    A Transformer encoder attends over the whole observed history, so it can pick
    out slow structure (the ~1.6 s sway cycle, a steady wind) that a 48 ms flat
    window could never see, and compress it into z.
    """

    def __init__(self, state_dim: int = STATE_DIM, d_model: int = 96, nhead: int = 4,
                 layers: int = 3, context_dim: int = 96, max_len: int = 256):
        super().__init__()
        self.inp = nn.Linear(state_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        # attention pooling: a learned query scores each frame, so the context keeps
        # the phase-carrying frames instead of averaging them away (mean-pool did).
        self.pool_score = nn.Linear(d_model, 1)
        self.to_ctx = nn.Linear(d_model, context_dim)
        self.context_dim = context_dim

    def forward(self, hist_feat, key_padding_mask=None):
        """hist_feat: (B, L, state_dim) normalized. key_padding_mask: (B, L) bool,
        True = padded/ignore. Variable-length (streaming) histories are masked out of
        both the attention and the pooling."""
        h = self.inp(hist_feat) + self.pos[:, : hist_feat.size(1)]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        scores = self.pool_score(h).squeeze(-1)               # (B, L)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)        # (B, L, 1)
        pooled = (h * w).sum(dim=1)                           # (B, d_model)
        return self.to_ctx(pooled)                            # (B, context_dim)


class SeqPredictor(nn.Module):
    """Memory encoder + GRU decoder that predicts a whole future trajectory.

    Encode the observed history into z, then autoregressively roll the future
    forward with a GRU cell whose input is [current-state-features, z] — so the
    memory is re-read at *every* step. Because the loss is over the whole rolled-out
    horizon (not one step), the network is forced to make z genuinely informative
    about the environment, and the rollout is trained to stay stable.
    """

    def __init__(self, state_dim: int = STATE_DIM, context_dim: int = 96,
                 dec_hidden: int = 256, d_model: int = 96, nhead: int = 4,
                 enc_layers: int = 3, n_time_freq: int = 12):
        super().__init__()
        self.encoder = TrajContextEncoder(state_dim, d_model, nhead, enc_layers, context_dim)
        self.h0 = nn.Linear(context_dim, dec_hidden)
        # Fourier time-clock: the sway force is a periodic function of the step
        # counter, so we hand the decoder a bank of sin/cos of the elapsed rollout
        # step. Combined with the frequency implied by z, it can lock onto the
        # oscillation's phase instead of trying to count steps implicitly.
        # The band is matched to the leaf's ACTUAL forcing: leaf_world drives sway
        # at angular freqs f*{0.7..2.7} with f in [0.025,0.045] -> ~0.017..0.12
        # rad/step. We span 0.008..0.25 to cover the fundamental + harmonics with
        # margin (the old 0.05..1.2 band was tuned to the wrong frequencies).
        self.n_time_freq = n_time_freq
        time_dim = 2 * n_time_freq
        freqs = torch.exp(torch.linspace(math.log(0.008), math.log(0.25), n_time_freq))
        self.register_buffer("time_freqs", freqs)          # (F,) angular rad/step
        self.cell = nn.GRUCell(state_dim + context_dim + time_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, state_dim)
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.dec_hidden = dec_hidden

    def encode(self, hist_feat, key_padding_mask=None):
        return self.encoder(hist_feat, key_padding_mask)

    def decode(self, z, cur_state, anchor_pos, n_steps,
               feat_mean, feat_std, delta_mean, delta_std, quat_norm: bool = True):
        """Differentiable autoregressive rollout in absolute state space.

        z:(B,C)  cur_state:(B,13) absolute  anchor_pos:(B,3)
        feat_*/delta_*: (13,) normalization tensors. Returns predicted NORMALIZED
        deltas (B, n_steps, 13); also returns the absolute states it passed through.
        """
        h = torch.tanh(self.h0(z))
        state = cur_state
        B = z.size(0)
        deltas_norm, states = [], []
        for j in range(n_steps):
            feat_rel = torch.cat([state[:, 0:3] - anchor_pos, state[:, 3:]], dim=-1)
            feat = (feat_rel - feat_mean) / feat_std
            ang = self.time_freqs * float(j)                 # (F,)
            tf = torch.cat([torch.sin(ang), torch.cos(ang)]).unsqueeze(0).expand(B, -1)
            h = self.cell(torch.cat([feat, z, tf], dim=-1), h)
            d_norm = self.out(h)                             # (B,13)
            deltas_norm.append(d_norm)
            delta = d_norm * delta_std + delta_mean
            state = state + delta
            if quat_norm:
                q = state[:, 3:7]
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
                state = torch.cat([state[:, 0:3], q, state[:, 7:]], dim=-1)
            states.append(state)
        return torch.stack(deltas_norm, dim=1), torch.stack(states, dim=1)

