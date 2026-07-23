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

    def _run_steps(self, h, state, z, start, count, anchor_pos,
                   feat_mean, feat_std, delta_mean, delta_std, quat_norm):
        """Roll `count` autoregressive steps starting at step index `start`.
        Returns (h, state, chunk_deltas_norm (B,count,13), chunk_states (B,count,13))."""
        B = z.size(0)
        dn_list, st_list = [], []
        for j in range(start, start + count):
            feat_rel = torch.cat([state[:, 0:3] - anchor_pos, state[:, 3:]], dim=-1)
            feat = (feat_rel - feat_mean) / feat_std
            ang = self.time_freqs * float(j)                 # (F,)
            tf = torch.cat([torch.sin(ang), torch.cos(ang)]).unsqueeze(0).expand(B, -1)
            h = self.cell(torch.cat([feat, z, tf], dim=-1), h)
            d_norm = self.out(h)                             # (B,13)
            delta = d_norm * delta_std + delta_mean
            state = state + delta
            if quat_norm:
                q = state[:, 3:7]
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
                state = torch.cat([state[:, 0:3], q, state[:, 7:]], dim=-1)
            dn_list.append(d_norm)
            st_list.append(state)
        return h, state, torch.stack(dn_list, dim=1), torch.stack(st_list, dim=1)

    def decode(self, z, cur_state, anchor_pos, n_steps,
               feat_mean, feat_std, delta_mean, delta_std, quat_norm: bool = True,
               ckpt_chunk: int = 0):
        """Differentiable autoregressive rollout in absolute state space.

        z:(B,C)  cur_state:(B,13) absolute  anchor_pos:(B,3)
        feat_*/delta_*: (13,) normalization tensors. Returns predicted NORMALIZED
        deltas (B, n_steps, 13); also the absolute states it passed through.

        ckpt_chunk>0 enables gradient checkpointing: the rollout is run in chunks of
        that many steps whose activations are recomputed in backward instead of
        stored — cutting rollout memory ~chunk-fold for ~30% more compute, so the big
        model can train at a big batch. Off (0) by default and during eval.
        """
        h = torch.tanh(self.h0(z))
        state = cur_state
        if not (ckpt_chunk and self.training):
            _, _, dn, st = self._run_steps(h, state, z, 0, n_steps, anchor_pos,
                                           feat_mean, feat_std, delta_mean, delta_std, quat_norm)
            return dn, st
        import torch.utils.checkpoint as cp
        dn_all, st_all, j = [], [], 0
        while j < n_steps:
            c = min(ckpt_chunk, n_steps - j)
            h, state, dn_c, st_c = cp.checkpoint(
                self._run_steps, h, state, z, j, c, anchor_pos,
                feat_mean, feat_std, delta_mean, delta_std, quat_norm,
                use_reentrant=False)
            dn_all.append(dn_c); st_all.append(st_c)
            j += c
        return torch.cat(dn_all, dim=1), torch.cat(st_all, dim=1)


# ---------------------------------------------------------------------------
# The FLOW-MAP predictor — same job as SeqPredictor, but not autoregressive.
#
# SeqPredictor integrates the future one step at a time. That is 240 *sequential*
# GRU cells per training batch, each a tiny matmul: the GPU spends its life waiting,
# gradient checkpointing is forced on to fit memory, and errors compound along the
# chain. The sequential depth — not the arithmetic — is what makes training slow.
#
# But we do not actually need a recurrence. For a deterministic system the future is
# a *function* of (initial state, environment, elapsed time):
#
#       x(t0 + k*dt) = Phi(k ; x0, theta)          <- the flow map of the ODE
#
# The encoder already infers theta as the context z, so we can learn Phi directly and
# evaluate every horizon k = 1..T **in parallel** — one big batched MLP instead of a
# 240-deep chain. This is the classical solution-operator view (DeepONet / neural
# operators) rather than the numerical-integrator view.
#
# Consequences:
#   * ~sequential depth 240 -> 1, so the GPU is actually saturated.
#   * no compounding error: horizon 240 is supervised directly, not through 239
#     earlier predictions.
#   * no gradient checkpointing needed.
#   * any horizon can be queried directly, including ones between training samples.
# The cost is that smoothness across k is learned rather than structural, which the
# time embedding below is designed to supply.
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 2 * d)
        self.f2 = nn.Linear(2 * d, d)

    def forward(self, x):
        return x + self.f2(nn.functional.gelu(self.f1(self.n1(x))))


class DirectTrajPredictor(nn.Module):
    """Context encoder + parallel flow-map decoder.

    Shares `TrajContextEncoder` with `SeqPredictor`, so the "watch the motion and
    infer the environment" half is unchanged; only the rollout is replaced.
    """

    LOGSTD_MIN, LOGSTD_MAX = -7.0, 3.0

    def __init__(self, state_dim: int = STATE_DIM, context_dim: int = 96,
                 hidden: int = 512, blocks: int = 4, d_model: int = 96, nhead: int = 4,
                 enc_layers: int = 3, n_phys_freq: int = 12, n_oct_freq: int = 10,
                 max_horizon: int = 320, probabilistic: bool = False):
        super().__init__()
        self.encoder = TrajContextEncoder(state_dim, d_model, nhead, enc_layers, context_dim)
        self.max_horizon = max_horizon
        self.probabilistic = probabilistic
        # Two complementary time bases, because horizon has two kinds of structure:
        #  * physical band  — the same 0.008..0.25 rad/step band the GRU clock used,
        #    matched to the leaf's sway and the pendulums' swing, so OSCILLATION is
        #    representable at the right frequencies.
        #  * octave band on tau = k/max_horizon — NeRF-style, resolves the smooth
        #    monotone GROWTH of displacement with elapsed time at every scale.
        self.n_phys_freq, self.n_oct_freq = n_phys_freq, n_oct_freq
        self.register_buffer(
            "phys_freqs",
            torch.exp(torch.linspace(math.log(0.008), math.log(0.25), n_phys_freq)))
        self.register_buffer(
            "oct_freqs", math.pi * torch.pow(2.0, torch.arange(n_oct_freq).float()))
        time_dim = 1 + 2 * n_phys_freq + 2 * n_oct_freq
        self.inp = nn.Linear(context_dim + state_dim + time_dim, hidden)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        # Probabilistic mode emits mean AND log-std per horizon per dimension.
        self.out = nn.Linear(hidden, state_dim * (2 if probabilistic else 1))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # -> mean starts at "no motion"; log_std starts at 0, i.e. sigma = 1 in
        #    normalized-displacement units, which is O(1) by construction of hstd.
        self.state_dim, self.context_dim = state_dim, context_dim

    def encode(self, hist_feat, key_padding_mask=None):
        return self.encoder(hist_feat, key_padding_mask)

    def time_embed(self, k):
        """k: (T,) float horizon indices (1-based) -> (T, time_dim)."""
        tau = (k / self.max_horizon).unsqueeze(-1)                     # (T,1)
        ph = k.unsqueeze(-1) * self.phys_freqs                         # (T,F)
        oc = tau * self.oct_freqs                                      # (T,F)
        return torch.cat([tau, torch.sin(ph), torch.cos(ph),
                          torch.sin(oc), torch.cos(oc)], dim=-1)

    def decode(self, z, cur_state, anchor_pos, n_steps, feat_mean, feat_std,
               horizon_std, quat_norm: bool = True):
        """z:(B,C)  cur_state:(B,13)  horizon_std:(H,13) per-horizon displacement scale.

        Returns (normalized displacement (B,T,13), absolute states (B,T,13)).

        Every horizon is computed at once — the T axis is pure batch, with no
        dependency between steps.
        """
        B = z.size(0)
        dev = z.device
        feat_rel = torch.cat([cur_state[:, 0:3] - anchor_pos, cur_state[:, 3:]], dim=-1)
        feat = (feat_rel - feat_mean) / feat_std                       # (B,13)
        k = torch.arange(1, n_steps + 1, device=dev, dtype=z.dtype)
        te = self.time_embed(k)                                        # (T,time_dim)

        x = torch.cat([
            z.unsqueeze(1).expand(B, n_steps, -1),
            feat.unsqueeze(1).expand(B, n_steps, -1),
            te.unsqueeze(0).expand(B, n_steps, -1)], dim=-1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        raw = self.out(self.norm(h))                                   # (B,T,13 or 26)
        if self.probabilistic:
            disp_n, log_std_n = raw.chunk(2, dim=-1)
            log_std_n = torch.clamp(log_std_n, self.LOGSTD_MIN, self.LOGSTD_MAX)
        else:
            disp_n, log_std_n = raw, None

        # horizons past the fitted table reuse the last scale (mild extrapolation)
        idx = torch.clamp(torch.arange(n_steps, device=dev), max=horizon_std.size(0) - 1)
        scale = horizon_std.index_select(0, idx).unsqueeze(0)          # (1,T,13)
        states = cur_state.unsqueeze(1) + disp_n * scale
        if quat_norm:
            q = states[..., 3:7]
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            states = torch.cat([states[..., 0:3], q, states[..., 7:]], dim=-1)
        return disp_n, states, log_std_n

