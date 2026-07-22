"""
generate_data.py — roll out many randomized leaf falls and save them, in parallel.

Each episode gets its own wind, air density, leaf mass, and initial spin, so the
dataset covers a whole family of falls rather than one memorized path. MuJoCo runs
on CPU, so we fan out across processes to use every core.

    python -m action.generate_data --episodes 1500 --out data/leaf --workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from action.leaf_world import LeafWorld


def _gen_one(task):
    """Top-level so it is picklable on Windows (spawn). Returns trajectory length."""
    idx, out_str, substeps, max_steps, seed_base = task
    out = Path(out_str)
    w = LeafWorld(seed=seed_base + idx)
    w.reset(randomize=True)
    traj = w.rollout(max_steps=max_steps, n_substeps=substeps)
    np.save(out / f"ep_{idx:05d}.npy", traj)
    np.save(out / f"ep_{idx:05d}_wind.npy", w.wind.astype(np.float32))
    return len(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", type=str, default="data/leaf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(i, str(out), args.substeps, args.max_steps, args.seed)
             for i in range(args.episodes)]

    lengths = []
    if args.workers <= 1:
        for t in tqdm(tasks, desc="falling leaves"):
            lengths.append(_gen_one(t))
    else:
        with Pool(processes=args.workers) as pool:
            for L in tqdm(pool.imap_unordered(_gen_one, tasks, chunksize=8),
                          total=len(tasks), desc="falling leaves"):
                lengths.append(L)

    lengths = np.array(lengths)
    print(f"\nsaved {args.episodes} episodes to {out}  (workers={args.workers})")
    print(f"trajectory length: mean={lengths.mean():.0f}  min={lengths.min()}  max={lengths.max()}")
    print(f"total transitions ~ {lengths.sum():,}")


if __name__ == "__main__":
    main()
