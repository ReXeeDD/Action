"""
diagnose_worlds.py — per-world autopsy of a trained general model.

The point: "the pendulum is trash" can mean two very different things, and they have
opposite fixes.

  (a) PHYSICAL LIMIT. pendulum3/4 are genuinely chaotic (we measured 679,000x and
      1.65M x separation growth). Past the Lyapunov horizon a single-line prediction
      is *guaranteed* wrong, and no model can fix it — only a probabilistic cone can.

  (b) MODEL / SETUP LIMIT. Something about the pipeline is wrong for pendulums
      specifically (normalizer scale, training horizon, target choice, data mix).

The decisive test is **pendulum1**: one link is INTEGRABLE — perfectly predictable
(we measured 28x, i.e. no chaos). If the model is also bad on pendulum1, chaos is
not the explanation and (b) is the culprit.

Metrics are scale-aware, because raw centimetres are meaningless across worlds: a
ball travels 10 m, a pendulum tip lives inside a 1 m sphere. We report error
relative to two baselines the model MUST beat to be worth anything:

    freeze   — predict the object never moves again
    const-v  — predict it keeps its current velocity forever

skill = 1 - err_model / err_freeze.   1.0 = perfect, 0.0 = no better than a rock,
negative = actively worse than assuming it stopped.

    python -m action.diagnose_worlds --ckpt runs/general.pt
    python -m action.diagnose_worlds --ckpt runs/general.pt --worlds pendulum1,pendulum2
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from action.worlds import make_world
from action.train_memory import load_mem

DT = 0.008


def rollout_episodes(world_name: str, n: int, seed: int, max_steps: int):
    """Fresh episodes in the universal (T,13) target format."""
    eps = []
    for i in range(n):
        w = make_world(world_name, seed=seed + i)
        w.reset(randomize=True)
        traj = w.target_rollout(max_steps=max_steps)
        if len(traj) > 40:
            eps.append(traj)
    return eps


def _baselines(traj, t0, H):
    """freeze and constant-velocity predictions of traj[t0 : t0+H] positions."""
    cur = traj[t0 - 1]
    k = np.arange(1, H + 1)[:, None]
    freeze = np.repeat(cur[None, 0:3], H, axis=0)
    constv = cur[0:3][None, :] + cur[7:10][None, :] * k * DT
    return freeze, constv


def evaluate(world_name, mem, n_eps, seed, max_steps, horizons, watch_frac):
    eps = rollout_episodes(world_name, n_eps, seed, max_steps)
    if not eps:
        return None
    Hmax = max(horizons)
    rows = {h: {"model": [], "freeze": [], "constv": []} for h in horizons}
    speeds, spans, lens = [], [], []

    for traj in eps:
        lens.append(len(traj))
        speeds.append(np.linalg.norm(traj[:, 7:10], axis=1).mean())
        spans.append(np.linalg.norm(traj[:, 0:3].max(0) - traj[:, 0:3].min(0)))
        t0 = max(12, int(watch_frac * len(traj)))
        if t0 + min(horizons) >= len(traj):
            continue
        H = min(Hmax, len(traj) - t0)
        pred = mem.predict(traj[:t0], traj[t0 - 1], H)
        truth = traj[t0:t0 + H, 0:3]
        fz, cv = _baselines(traj, t0, H)
        for h in horizons:
            if h > H:
                continue
            i = h - 1
            rows[h]["model"].append(np.linalg.norm(pred[i, 0:3] - truth[i]))
            rows[h]["freeze"].append(np.linalg.norm(fz[i] - truth[i]))
            rows[h]["constv"].append(np.linalg.norm(cv[i] - truth[i]))

    out = {"world": world_name, "n_eps": len(eps),
           "mean_speed": float(np.mean(speeds)),
           "mean_span": float(np.mean(spans)),
           "mean_len": float(np.mean(lens)), "rows": {}}
    for h in horizons:
        if not rows[h]["model"]:
            continue
        m = float(np.mean(rows[h]["model"]))
        f = float(np.mean(rows[h]["freeze"]))
        c = float(np.mean(rows[h]["constv"]))
        out["rows"][h] = {"model": m, "freeze": f, "constv": c,
                          "skill": 1.0 - m / max(f, 1e-9)}
    return out


def delta_stats(world_name, n_eps, seed, max_steps):
    """Per-world statistics of the quantities the GLOBAL normalizer standardizes.

    If one world's deltas are far outside the global scale, its gradients are either
    drowned out or blown up — a plain training bug, not physics."""
    eps = rollout_episodes(world_name, n_eps, seed, max_steps)
    d = np.concatenate([e[1:] - e[:-1] for e in eps], axis=0)
    s = np.concatenate(eps, axis=0)
    return {"world": world_name,
            "d_pos_rms": float(np.sqrt((d[:, 0:3] ** 2).sum(1).mean())),
            "d_vel_rms": float(np.sqrt((d[:, 7:10] ** 2).sum(1).mean())),
            "spd_rms": float(np.sqrt((s[:, 7:10] ** 2).sum(1).mean())),
            "spin_rms": float(np.sqrt((s[:, 10:13] ** 2).sum(1).mean())),
            "abs_pos_rms": float(np.sqrt((s[:, 0:3] ** 2).sum(1).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/general.pt")
    ap.add_argument("--worlds", type=str,
                    default="object,ball,pendulum1,pendulum2,pendulum3,nbody3")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=90000)   # far from any training seed
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--watch", type=float, default=0.35, help="fraction watched first")
    ap.add_argument("--horizons", type=str, default="12,30,60,120,240")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    worlds = [w.strip() for w in args.worlds.split(",")]
    mem = load_mem(args.ckpt, args.device)
    dsd, fs = mem.dsd, mem.fs

    print(f"ckpt={args.ckpt}  arch={mem.arch}  hist_cap={mem.hist_cap}  "
          f"watched={args.watch:.0%} of episode")
    print("global normalizer the model was trained with:")
    print(f"  delta_std pos  = {np.round(dsd.cpu().numpy()[0:3], 5)}")
    print(f"  delta_std vel  = {np.round(dsd.cpu().numpy()[7:10], 5)}")
    print(f"  feat_std  pos  = {np.round(fs.cpu().numpy()[0:3], 4)}   "
          f"vel = {np.round(fs.cpu().numpy()[7:10], 4)}")

    print("\n" + "=" * 92)
    print("PER-WORLD SKILL  (skill = 1 - err_model/err_freeze;  0 = no better than "
          "'it stopped')")
    print("=" * 92)
    hdr = "world        " + "".join(f"{h*DT:>7.2f}s" for h in horizons)
    print(hdr + "     (each cell: model_cm / freeze_cm -> skill)")

    results = []
    for w in worlds:
        r = evaluate(w, mem, args.episodes, args.seed, args.max_steps, horizons,
                     args.watch)
        if r is None:
            continue
        results.append(r)
        line_m = f"{w:<12}"
        line_s = f"{'':<12}"
        for h in horizons:
            if h not in r["rows"]:
                line_m += f"{'--':>8}"; line_s += f"{'':>8}"
                continue
            e = r["rows"][h]
            line_m += f"{e['model']*100:7.1f}c"
            line_s += f"{e['skill']:+7.2f} "
        print(line_m)
        print(line_s + "   <- skill")
        fr = "  freeze:   " + "".join(
            f"{r['rows'][h]['freeze']*100:7.1f}c" if h in r["rows"] else f"{'--':>8}"
            for h in horizons)
        cv = "  const-v:  " + "".join(
            f"{r['rows'][h]['constv']*100:7.1f}c" if h in r["rows"] else f"{'--':>8}"
            for h in horizons)
        print(fr); print(cv)
        print(f"  scale: mean speed {r['mean_speed']:.2f} m/s   path span "
              f"{r['mean_span']:.2f} m   episode {r['mean_len']:.0f} frames "
              f"({r['mean_len']*DT:.1f}s)   n={r['n_eps']}\n")

    print("=" * 92)
    print("PER-WORLD SCALE vs THE ONE GLOBAL NORMALIZER")
    print("=" * 92)
    print(f"{'world':<12}{'|dpos|rms':>11}{'|dvel|rms':>11}{'speed rms':>11}"
          f"{'spin rms':>11}{'|pos| rms':>11}")
    for w in worlds:
        st = delta_stats(w, min(args.episodes, 12), args.seed + 5000, args.max_steps)
        print(f"{w:<12}{st['d_pos_rms']:>11.5f}{st['d_vel_rms']:>11.5f}"
              f"{st['spd_rms']:>11.3f}{st['spin_rms']:>11.3f}{st['abs_pos_rms']:>11.3f}")
    g = dsd.cpu().numpy()
    print(f"{'GLOBAL std':<12}{np.linalg.norm(g[0:3]):>11.5f}"
          f"{np.linalg.norm(g[7:10]):>11.5f}")


if __name__ == "__main__":
    main()
