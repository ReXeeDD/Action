"""
cone.py — draw the cone of possible futures.

We seed the ensemble with the first few real frames of a held-out fall, then sample
MANY futures: each sample picks a random ensemble member and, at every step, draws the
next delta from that member's Gaussian instead of taking the mean. Small differences
get amplified by the chaotic dynamics, and the spray of paths fans out into a cone.

The honest test isn't "does one line match" — it's **does reality land inside the
cone, and does the cone widen at the right rate?** We measure both.

    python -m action.cone --data data/leaf --ckpt runs/leaf_ensemble.pt --samples 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from action.dataset import load_episodes, make_features, Normalizer
from action.models import GaussianMLP
from action.leaf_world import QUAT


def load_ensemble(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    models = []
    for state in ck["members"]:
        m = GaussianMLP(history=ck["history"]).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    xn = Normalizer(**ck["x_norm"])
    yn = Normalizer(**ck["y_norm"])
    return models, xn, yn, ck["history"]


@torch.no_grad()
def sample_futures(models, xn, yn, history, seed_window, n_steps, n_samples,
                   device="cpu", rng=None, temperature=1.0):
    """Return (n_samples, n_steps, 13) sampled future states.

    Vectorized over samples: all n_samples roll forward together as one batch.
    Each sample is assigned one ensemble member for its whole rollout (epistemic
    spread); within a step we run each member on its own slice of the batch.
    """
    rng = rng or np.random.default_rng(0)
    M = len(models)
    xmean = torch.tensor(xn.mean, device=device)
    xstd = torch.tensor(xn.std, device=device)
    ymean = torch.tensor(yn.mean, device=device)
    ystd = torch.tensor(yn.std, device=device)
    qi = torch.arange(QUAT.start, QUAT.stop, device=device)

    # physical bounds so a diverging sample can't blow past the world
    lo = torch.tensor([-20, -20, 0.0, -1, -1, -1, -1, -25, -25, -25, -60, -60, -60],
                      device=device)
    hi = torch.tensor([20, 20, 12, 1, 1, 1, 1, 25, 25, 25, 60, 60, 60], device=device)

    member = torch.as_tensor(rng.integers(0, M, size=n_samples), device=device)
    seed = torch.tensor(np.asarray(seed_window[-history:]), device=device)   # (H,13)
    win = seed.unsqueeze(0).repeat(n_samples, 1, 1).clone()                   # (S,H,13)
    landed = torch.zeros(n_samples, dtype=torch.bool, device=device)

    out = torch.empty((n_samples, n_steps, 13), device=device)
    for k in range(n_steps):
        cur_pos = win[:, -1, 0:3].clone()                     # (S,3)
        feat = win.clone()
        feat[:, :, 0:3] = feat[:, :, 0:3] - cur_pos.unsqueeze(1)
        xin = (feat.reshape(n_samples, -1) - xmean) / xstd    # (S, H*13)

        mean = torch.empty((n_samples, 13), device=device)
        logstd = torch.empty((n_samples, 13), device=device)
        for m in range(M):                                    # per-member batched forward
            sel = member == m
            if sel.any():
                mu, ls = models[m](xin[sel])
                mean[sel], logstd[sel] = mu, ls

        std = torch.exp(logstd) * temperature
        delta = (mean + std * torch.randn_like(std)) * ystd + ymean
        nxt = torch.clamp(win[:, -1, :] + delta, lo, hi)      # (S,13), bounded
        q = nxt[:, qi]
        nrm = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
        nxt[:, qi] = q / nrm
        # once a sample touches ground it stays put (physically, it landed)
        nxt = torch.where(landed.unsqueeze(1), win[:, -1, :], nxt)
        landed = landed | (nxt[:, 2] <= 0.03)
        out[:, k, :] = nxt
        win = torch.cat([win[:, 1:, :], nxt.unsqueeze(1)], dim=1)

    return out.cpu().numpy()


def coverage_report(samples, truth):
    """How well does the cone bracket reality, horizon by horizon?"""
    n = min(samples.shape[1], len(truth))
    sp = samples[:, :n, 0:3]                    # (S, n, 3)
    tr = truth[:n, 0:3]                         # (n, 3)
    mean_path = sp.mean(0)                       # (n,3)
    spread = np.linalg.norm(sp - mean_path[None], axis=2).mean(0)   # cone radius(t)
    err_mean = np.linalg.norm(mean_path - tr, axis=1)               # |mean - truth|
    # nearest sampled path to truth at each step (is truth *somewhere* in the cone?)
    nearest = np.linalg.norm(sp - tr[None], axis=2).min(0)
    return mean_path, spread, err_mean, nearest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--ckpt", type=str, default="runs/leaf_ensemble.pt")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--plot", type=str, default="runs/cone.png")
    args = ap.parse_args()

    models, xn, yn, history = load_ensemble(args.ckpt)
    eps = load_episodes(args.data)
    traj = eps[args.episode]
    seed = traj[:history]
    truth = traj[history:]

    samples = sample_futures(models, xn, yn, history, seed,
                             n_steps=len(truth), n_samples=args.samples,
                             temperature=args.temperature)

    mean_path, spread, err_mean, nearest = coverage_report(samples, truth)
    dt = 0.008
    for sec in (0.5, 1.0, 2.0, 3.0):
        i = min(int(sec / dt), len(spread) - 1)
        print(f"@ {sec:.1f}s   cone radius {spread[i]:.2f} m | "
              f"mean-vs-truth {err_mean[i]:.2f} m | "
              f"closest path to truth {nearest[i]:.2f} m")
    inside = nearest < spread                    # truth within the spray
    print(f"\ntruth stays inside the cone {100*inside.mean():.0f}% of the horizon")

    try:
        from action.viz import plot_cone
        plot_cone(seed, truth, samples, out=args.plot)
        print(f"plot -> {args.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
