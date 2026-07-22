"""
dataset.py — turn raw leaf trajectories into supervised (history -> next-step) data.

Key design choices, and why:

* **History window (H frames), not a single frame.** The aero force depends on
  velocity-relative-to-wind and on orientation (both in the state) BUT the wind
  and air density are *hidden* and constant within an episode. A window lets the
  network infer those hidden context variables from how the leaf has been moving.
  (This is the honest version of "watch the motion, then predict the future.")

* **Translation invariance.** The physics doesn't care where in x/y/z the leaf
  is, so we feed positions *relative to the current frame*. The network never
  sees an absolute coordinate and therefore generalizes across the whole sky.

* **Predict deltas, not absolute next states.** Easier target, and it makes the
  identity map (nothing moves) the trivial baseline the net must beat.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from action.leaf_world import STATE_DIM  # 13


def make_features(window: np.ndarray) -> np.ndarray:
    """(H, 13) absolute states -> (H*13,) translation-invariant feature vector.

    Position is expressed relative to the *current* (last) frame; everything
    else (quat, lin/ang velocity) is kept as-is.
    """
    cur_pos = window[-1, 0:3]
    feats = window.copy()
    feats[:, 0:3] = feats[:, 0:3] - cur_pos
    return feats.reshape(-1)


class Normalizer:
    """Per-feature standardization with save/load. Zero-variance cols pass through."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(0), x.std(0))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean, "std": self.std}


def load_episodes(data_dir: str | Path) -> list[np.ndarray]:
    data_dir = Path(data_dir)
    # exclude the ep_*_wind.npy context files saved alongside each trajectory
    files = sorted(f for f in data_dir.glob("ep_*.npy") if not f.stem.endswith("_wind"))
    if not files:
        raise FileNotFoundError(f"no episodes in {data_dir} — run generate_data.py first")
    return [np.load(f) for f in files]


def build_supervised(episodes: list[np.ndarray], history: int = 6):
    """Return X (N, H*13) features, Y (N, 13) raw next-step deltas."""
    X, Y = [], []
    for traj in episodes:
        if len(traj) <= history:
            continue
        for t in range(history - 1, len(traj) - 1):
            window = traj[t - history + 1 : t + 1]          # (H, 13)
            delta = traj[t + 1] - traj[t]                    # (13,)
            X.append(make_features(window))
            Y.append(delta)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


if __name__ == "__main__":
    eps = load_episodes("data/leaf")
    X, Y = build_supervised(eps, history=6)
    print(f"episodes={len(eps)}  samples={len(X)}  X={X.shape}  Y={Y.shape}")
    print("target-delta std per dim:", np.round(Y.std(0), 4))
