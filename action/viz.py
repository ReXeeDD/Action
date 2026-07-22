"""
viz.py — see the future the network drew.

plot_true_vs_pred: the real fall (from MuJoCo) vs the predicted projection, in
3D, plus how position error grows with the prediction horizon. That error curve
is the honest picture of chaos: small at first, blossoming later.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_true_vs_pred(seed, truth, pred, err, out="runs/rollout.png"):
    fig = plt.figure(figsize=(13, 5.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s = np.asarray(seed)
    t = np.asarray(truth)
    p = np.asarray(pred)
    ax.plot(*s[:, 0:3].T, color="0.4", lw=2, label="seed (given)")
    ax.plot(*t[:, 0:3].T, color="#1f8f3a", lw=2, label="true fall (MuJoCo)")
    ax.plot(*p[:, 0:3].T, color="#d9531e", lw=2, ls="--", label="predicted future")
    ax.scatter(*s[-1, 0:3], color="k", s=30)
    ax.set_title("Leaf fall: predicted vs real")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    dt = 0.008  # timestep * substeps
    ax2.plot(np.arange(len(err)) * dt, err, color="#d9531e")
    ax2.set_title("Prediction error grows with horizon (chaos)")
    ax2.set_xlabel("seconds into the future")
    ax2.set_ylabel("position error (m)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_cone(seed, truth, samples, out="runs/cone.png", max_lines=120):
    """Many predicted futures -> the 'cone of possibility'.

    Left: the 3D spray of sampled futures with the real outcome overlaid.
    Right: cone radius vs the mean-path error over the horizon — the picture of
    uncertainty growing exactly as fast as chaos demands.
    """
    s = np.asarray(seed)
    t = np.asarray(truth)
    S = np.asarray(samples)                       # (n_samples, n_steps, 13)
    mean_path = S[:, :, 0:3].mean(0)

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*s[:, 0:3].T, color="0.35", lw=2.5, label="seed (given)")
    shown = S[np.random.default_rng(0).choice(len(S), min(max_lines, len(S)), replace=False)]
    for i, p in enumerate(shown):
        ax.plot(*p[:, 0:3].T, color="#d9531e", lw=0.5, alpha=0.16,
                label="possible futures" if i == 0 else None)
    ax.plot(*mean_path.T, color="#8a2b0a", lw=1.6, ls=":", label="mean future")
    ax.plot(*t[:, 0:3].T, color="#1f8f3a", lw=2.2, label="what actually happened")
    ax.scatter(*s[-1, 0:3], color="k", s=25)
    ax.set_title("Cone of possible futures")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    dt = 0.008
    n = min(S.shape[1], len(t))
    time = np.arange(n) * dt
    spread = np.linalg.norm(S[:, :n, 0:3] - mean_path[None, :n], axis=2).mean(0)
    err = np.linalg.norm(mean_path[:n] - t[:n, 0:3], axis=1)
    ax2.fill_between(time, 0, spread, color="#d9531e", alpha=0.25, label="cone radius (uncertainty)")
    ax2.plot(time, err, color="#1f8f3a", lw=1.8, label="mean-future error vs reality")
    ax2.set_title("Uncertainty grows with the horizon")
    ax2.set_xlabel("seconds into the future")
    ax2.set_ylabel("meters")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
