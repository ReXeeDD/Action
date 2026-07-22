"""
rollout.py — project the future by feeding the model's own predictions back in.

Given the first few frames of a *held-out* leaf fall (an episode the network
never trained on), we predict every subsequent frame autoregressively and
compare the predicted flight path to what MuJoCo actually did.

    python -m action.rollout --data data/leaf --ckpt runs/leaf_mlp.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from action.dataset import load_episodes, make_features, Normalizer
from action.models import MLPPredictor
from action.leaf_world import QUAT


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPPredictor(history=ck["history"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    xn = Normalizer(**ck["x_norm"])
    yn = Normalizer(**ck["y_norm"])
    return model, xn, yn, ck["history"]


@torch.no_grad()
def predict_future(model, xn, yn, history, seed_window: np.ndarray, n_steps: int,
                   device: str = "cpu") -> np.ndarray:
    """seed_window: (H,13) absolute states. Returns (n_steps, 13) predicted states."""
    window = [s.copy() for s in seed_window]
    preds = []
    for _ in range(n_steps):
        feat = make_features(np.asarray(window[-history:]))
        x = torch.from_numpy(xn(feat)).float().unsqueeze(0).to(device)
        delta_norm = model(x).cpu().numpy()[0]
        delta = yn.inv(delta_norm)
        nxt = window[-1] + delta
        q = nxt[QUAT]                      # keep the orientation a unit quaternion
        n = np.linalg.norm(q)
        nxt[QUAT] = q / n if n > 1e-8 else np.array([1, 0, 0, 0], np.float32)
        window.append(nxt.astype(np.float32))
        preds.append(nxt.astype(np.float32))
    return np.asarray(preds, dtype=np.float32)


def position_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-step Euclidean position error (m). Aligns on the shorter length."""
    n = min(len(pred), len(truth))
    return np.linalg.norm(pred[:n, 0:3] - truth[:n, 0:3], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--ckpt", type=str, default="runs/leaf_mlp.pt")
    ap.add_argument("--episode", type=int, default=-1, help="which episode index; -1 = last")
    ap.add_argument("--plot", type=str, default="runs/rollout.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, xn, yn, history = load_model(args.ckpt, device)

    eps = load_episodes(args.data)
    traj = eps[args.episode]
    seed = traj[:history]
    truth_future = traj[history:]
    pred_future = predict_future(model, xn, yn, history, seed,
                                 n_steps=len(truth_future), device=device)

    err = position_error(pred_future, truth_future)
    print(f"episode length {len(traj)} steps")
    print(f"position error:  @0.5s ~ {err[min(60,len(err)-1)]:.3f} m   "
          f"final {err[-1]:.3f} m   mean {err.mean():.3f} m")

    try:
        from action.viz import plot_true_vs_pred
        plot_true_vs_pred(seed, truth_future, pred_future, err, out=args.plot)
        print(f"plot -> {args.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
