"""
generate_data.py — roll out many randomized episodes of ANY world, in parallel.

Every episode randomizes that world's hidden physics (the leaf's wind and air
density; the ball's mass, bounciness and friction; each pendulum link's length and
mass; the N-body masses and orbits), so a dataset covers a whole *family* of systems
rather than one memorized trajectory. Those hidden constants are exactly what the
memory model has to infer from motion alone.

MuJoCo physics is CPU-bound, so we fan out across processes.

    python -m action.generate_data --world leaf      --episodes 1500 --out data/leaf
    python -m action.generate_data --world pendulum3 --episodes 1500 --out data/pendulum3
    python -m action.generate_data --world nbody4    --episodes 1500 --out data/nbody4
    python -m action.generate_data --world ball      --episodes 1500 --out data/ball

A `meta.json` is written alongside the episodes recording the state layout
(state_dim, which indices are 3D positions, which are quaternions), so the training
pipeline stays physics-agnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from action.worlds import make_world, WORLD_NAMES


def _gen_one(task):
    """Top-level so it is picklable on Windows (spawn). Returns trajectory length."""
    idx, world_name, out_str, substeps, max_steps, seed_base, tag = task
    out = Path(out_str)
    w = make_world(world_name, seed=seed_base + idx)
    w.reset(randomize=True)
    # Universal single-target format: (T, 13) for the ONE object we predict, in
    # world coordinates, identical in every world. This is what lets one model
    # cover a leaf, a ball, a pendulum tip or an orbiting mass — and why a brand
    # new object type needs no retraining.
    if world_name == "leaf":                      # leaf keeps its own step loop
        from action.entities import entity_state
        traj = [entity_state(w.model, w.data)[0]]
        for _ in range(max_steps):
            w.step(substeps)
            traj.append(entity_state(w.model, w.data)[0])
            if w.on_ground:
                break
        traj = np.asarray(traj, dtype=np.float32)
        np.save(out / f"ep_{tag}_{idx:05d}_wind.npy", w.wind.astype(np.float32))
    else:
        traj = w.target_rollout(max_steps=max_steps)
    np.save(out / f"ep_{tag}_{idx:05d}.npy", traj)
    return len(traj)


def _write_meta(world_name: str, out: Path, seed: int):
    """Describe the UNIVERSAL single-target format that is actually saved.

    Every world writes (T, 13) world-frame states for one target body, so the layout
    is the same everywhere — that identical layout is exactly what makes one model
    work across worlds and on objects it has never seen.
    """
    from action.entities import ENTITY_DIM
    w = make_world(world_name, seed=seed)
    w.reset(randomize=True)
    n_ent = 1 if world_name == "leaf" else w.n_entities
    meta = {"world": world_name, "format": "universal_entity_target",
            "state_dim": ENTITY_DIM,                     # always 13
            "pos_groups": [[0, 1, 2]], "quat_groups": [[3, 4, 5, 6]],
            "n_entities_in_scene": int(n_ent),
            "target_index": 0 if world_name == "leaf" else int(w.target_index),
            "dt": 0.008}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=str, default="leaf",
                    help=f"one of: {', '.join(WORLD_NAMES)}")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", type=str, default=None,
                    help="default: data/<world>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", type=str, default=None,
                    help="filename prefix (default: world name). Lets several worlds "
                         "share one --out directory so you can train ONE general model.")
    args = ap.parse_args()

    out = Path(args.out or f"data/{args.world}")
    out.mkdir(parents=True, exist_ok=True)
    meta = _write_meta(args.world, out, args.seed)
    json.dump(meta, open(out / f"meta_{args.tag or args.world}.json", "w"), indent=2)
    print(f"world={meta['world']}  state_dim={meta['state_dim']}  dt={meta['dt']}")

    tasks = [(i, args.world, str(out), args.substeps, args.max_steps, args.seed, args.tag or args.world)
             for i in range(args.episodes)]

    lengths = []
    if args.workers <= 1:
        for t in tqdm(tasks, desc=args.world):
            lengths.append(_gen_one(t))
    else:
        with Pool(processes=args.workers) as pool:
            for L in tqdm(pool.imap_unordered(_gen_one, tasks, chunksize=8),
                          total=len(tasks), desc=args.world):
                lengths.append(L)

    lengths = np.array(lengths)
    print(f"\nsaved {args.episodes} episodes to {out}  (workers={args.workers})")
    print(f"trajectory length: mean={lengths.mean():.0f}  min={lengths.min()}  max={lengths.max()}")
    print(f"total transitions ~ {lengths.sum():,}")


if __name__ == "__main__":
    main()
