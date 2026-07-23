"""
audit_randomness.py — is the world actually random, and is it random in a way the
model can learn from?

Two very different questions hide inside "is the world purely random":

  (1) Is there randomness INSIDE an episode?  If the simulator injected noise at every
      step, the future would be irreducibly unpredictable and no model could ever win.
      This is exactly the wall we measured on the leaf (CLAUDE.md 6c): its prediction
      window was 100% self-inflicted by a per-step `rng.normal`, and deleting it made
      the world learnable in principle.

  (2) Is the randomness ACROSS episodes broad and unbiased?  The hidden constants are
      drawn once per episode and then held fixed. That is the good kind of randomness:
      it is what the memory model infers by watching. But if the draws are biased or
      too narrow, the model memorises a corner of the space instead of learning the
      physics -- and if they are too wide/degenerate, episodes stop being informative.

This script measures both, plus whether the hidden parameters are IDENTIFIABLE at all
(if two very different environments produce identical motion, no amount of memory can
tell them apart).

    python -m action.audit_randomness
    python -m action.audit_randomness --worlds ball,pendulum2 --episodes 60
"""
from __future__ import annotations

import argparse
import numpy as np

from action.worlds import make_world
from action.entities import entity_state

DT = 0.008


def rollout(name, seed, steps=400):
    w = make_world(name, seed=seed)
    w.reset(randomize=True)
    ti = 0 if name == "leaf" else w.target_index
    tr = [entity_state(w.model, w.data)[ti]]
    for _ in range(steps):
        w.step(2) if name == "leaf" else w.step()
        tr.append(entity_state(w.model, w.data)[ti])
        if name != "leaf" and w.done:
            break
    return np.asarray(tr, dtype=np.float64)


def check_determinism(name):
    """Same seed twice -> byte-identical? If not, there is randomness INSIDE the
    episode, which is an information-theoretic wall no model can cross."""
    a, b = rollout(name, 12345), rollout(name, 12345)
    n = min(len(a), len(b))
    same = np.array_equal(a[:n], b[:n]) and len(a) == len(b)
    c = rollout(name, 12346)
    m = min(len(a), len(c))
    differs = not np.array_equal(a[:m], c[:m])
    return same, differs


def heading_bias(name, n_eps, seed0):
    """Are trajectories omnidirectional, or do they prefer a compass direction?

    |R| is the resultant length of the unit heading vectors: 0 = perfectly uniform,
    1 = every episode goes the same way. A biased world teaches the model a preferred
    direction that will not hold for a new object."""
    hs = []
    for s in range(n_eps):
        t = rollout(name, seed0 + s, steps=250)
        d = t[-1, 0:3] - t[0, 0:3]
        d[2] = 0.0                                   # horizontal heading only
        if np.linalg.norm(d) > 1e-6:
            hs.append(d / np.linalg.norm(d))
    if len(hs) < 5:
        return None
    return float(np.linalg.norm(np.mean(hs, axis=0)))


def diversity(name, n_eps, seed0):
    """How different are episodes from each other, and is any pair a near-duplicate?

    Reported as the spread of a few coarse trajectory descriptors. If the nearest pair
    is far closer than typical, the sampler is producing repeats."""
    feats, paths = [], []
    for s in range(n_eps):
        t = rollout(name, seed0 + s, steps=250)
        sp = np.linalg.norm(t[:, 7:10], axis=1)
        feats.append([np.linalg.norm(t[-1, 0:3] - t[0, 0:3]),   # net displacement
                      sp.mean(), sp.max(),
                      np.linalg.norm(t[:, 10:13], axis=1).mean(),
                      len(t) * DT])
        paths.append(t[:200, 0:3])
    F = np.asarray(feats)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    d = np.linalg.norm(Fz[:, None, :] - Fz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    L = min(len(p) for p in paths)
    P = np.stack([p[:L].ravel() for p in paths])
    pd = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(pd, np.inf)
    return {"cv": float(np.mean(F.std(0) / (np.abs(F.mean(0)) + 1e-9))),
            "min_feat_dist": float(d.min()), "med_feat_dist": float(np.median(d)),
            "min_path_dist_m": float(pd.min()), "med_path_dist_m": float(np.median(pd))}


def identifiability(name, n_eps, seed0, watch=60):
    """Can the environment be told apart by WATCHING?

    Correlate 'how different two episodes look in their first `watch` frames' against
    'how different their futures are'. If early motion carries no information about the
    future beyond the current state, the memory model has nothing to infer and the
    whole premise fails. High correlation = the fingerprint is really there."""
    early, late = [], []
    for s in range(n_eps):
        t = rollout(name, seed0 + s, steps=400)
        if len(t) < watch + 150:
            continue
        e = t[:watch]
        # translation-invariant early descriptor: motion relative to the start
        early.append(np.concatenate([(e[:, 0:3] - e[0, 0:3]).ravel(),
                                     e[:, 7:10].ravel()]))
        late.append((t[watch:watch + 150, 0:3] - t[watch, 0:3]).ravel())
    if len(early) < 8:
        return None
    E = np.asarray(early); L = np.asarray(late)
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    de = np.linalg.norm(E[:, None] - E[None, :], axis=-1)
    dl = np.linalg.norm(L[:, None] - L[None, :], axis=-1)
    iu = np.triu_indices(len(E), 1)
    return float(np.corrcoef(de[iu], dl[iu])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=str,
                    default="leaf,ball,object,pendulum1,pendulum2,pendulum3,nbody2,nbody3")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=70000)
    args = ap.parse_args()
    worlds = [w.strip() for w in args.worlds.split(",")]

    print("=" * 88)
    print("1. IS THERE RANDOMNESS *INSIDE* AN EPISODE?  (the unlearnable kind)")
    print("=" * 88)
    print(f"{'world':<12}{'same seed -> identical':>24}{'diff seed -> differs':>22}   verdict")
    for w in worlds:
        same, diff = check_determinism(w)
        ok = same and diff
        print(f"{w:<12}{str(same):>24}{str(diff):>22}   "
              f"{'deterministic + varied (LEARNABLE)' if ok else 'PROBLEM'}")

    print("\n" + "=" * 88)
    print("2. IS THE RANDOMNESS *ACROSS* EPISODES BROAD AND UNBIASED?")
    print("=" * 88)
    print(f"{'world':<12}{'|R| heading bias':>18}{'spread(cv)':>12}"
          f"{'closest pair (m)':>18}{'median pair (m)':>17}   verdict")
    for w in worlds:
        b = heading_bias(w, args.episodes, args.seed)
        d = diversity(w, args.episodes, args.seed)
        bs = "n/a" if b is None else f"{b:.3f}"
        flag = "ok"
        if b is not None and b > 0.35:
            flag = "DIRECTION BIAS"
        if d["min_path_dist_m"] < 0.02 * max(d["med_path_dist_m"], 1e-9):
            flag = "NEAR-DUPLICATE EPISODES"
        print(f"{w:<12}{bs:>18}{d['cv']:>12.2f}{d['min_path_dist_m']:>18.2f}"
              f"{d['med_path_dist_m']:>17.2f}   {flag}")

    print("\n" + "=" * 88)
    print("3. IS THE ENVIRONMENT IDENTIFIABLE BY WATCHING?  (the project's premise)")
    print("=" * 88)
    print("corr( difference in the first 0.48s , difference in the next 1.2s ).")
    print("~0 = watching tells you nothing;  high = the motion is a fingerprint.\n")
    print(f"{'world':<12}{'corr(early, late)':>20}   verdict")
    for w in worlds:
        c = identifiability(w, args.episodes, args.seed)
        if c is None:
            print(f"{w:<12}{'n/a (episodes too short)':>20}")
            continue
        v = ("strong fingerprint" if c > 0.6 else
             "usable" if c > 0.3 else "WEAK - little to infer")
        print(f"{w:<12}{c:>20.3f}   {v}")


if __name__ == "__main__":
    main()
