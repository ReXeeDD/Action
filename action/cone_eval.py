"""
cone_eval.py — is the cone HONEST?

A probabilistic predictor is not judged by accuracy. A model that says "somewhere
within 10 metres" is never wrong and never useful; a model that says "within 2 cm" and
is right 40% of the time is confidently lying. The question is **calibration**: when it
claims 90% confidence, does the truth actually land inside 90% of the time?

Three numbers, per world, per horizon:

  coverage(p)  fraction of episodes where the truth lies inside the model's p% band.
               Well calibrated  => coverage(p) ~ p at EVERY horizon.
               coverage < p     => overconfident (the dangerous failure)
               coverage > p     => underconfident (safe but useless)

  sharpness    the median width of the band, in cm. Among calibrated models, smaller
               is better. Reported alongside coverage because either alone is gameable.

  CRPS         continuous ranked probability score (m). A proper scoring rule: it is
               minimised only by the true predictive distribution, so it cannot be
               gamed by inflating or shrinking the spread. This is the single number to
               compare models on.

The interesting result to look for: on a CHAOTIC world the band must *widen* with
horizon (there is genuinely more to be unsure about), while on an integrable one it
should stay tight. If a model shows the same spread on pendulum1 and pendulum3, it has
learned an average uncertainty rather than the physics of predictability.

    python -m action.cone_eval --ckpt runs/general4.pt
    python -m action.cone_eval --ckpt runs/general4.pt --worlds pendulum1,pendulum3
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.stats import norm as _norm  # only for the analytic CRPS of a Gaussian

from action.worlds import make_world
from action.train_memory import load_mem
from action.diagnose_worlds import rollout_episodes

DT = 0.008
LEVELS = (0.5, 0.9)


def crps_gaussian(mu, sigma, y):
    """Closed-form CRPS for a Gaussian forecast. Lower is better; units of y."""
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * _norm.cdf(z) - 1)
                                  + 2 * _norm.pdf(z) - 1 / np.sqrt(np.pi))))


def evaluate(world, mem, n_eps, seed, max_steps, horizons, watch):
    eps = rollout_episodes(world, n_eps, seed, max_steps)
    if not eps:
        return None
    Hmax = max(horizons)
    acc = {h: {"z": [], "err": [], "sig": [], "mu": [], "y": []} for h in horizons}
    for traj in eps:
        t0 = max(12, int(watch * len(traj)))
        if t0 + min(horizons) >= len(traj):
            continue
        H = min(Hmax, len(traj) - t0)
        mean, sigma = mem.predict_dist(traj[:t0], traj[t0 - 1], H)
        if sigma is None:
            return "deterministic"
        truth = traj[t0:t0 + H, 0:3]
        for h in horizons:
            if h > H:
                continue
            i = h - 1
            mu, sg, y = mean[i, 0:3], np.maximum(sigma[i, 0:3], 1e-9), truth[i]
            acc[h]["z"].append((y - mu) / sg)          # per-axis standardized error
            acc[h]["err"].append(np.linalg.norm(y - mu))
            acc[h]["sig"].append(sg)
            acc[h]["mu"].append(mu); acc[h]["y"].append(y)

    out = {}
    for h in horizons:
        if not acc[h]["z"]:
            continue
        z = np.asarray(acc[h]["z"])                    # (N,3)
        sg = np.asarray(acc[h]["sig"])
        row = {"err_cm": float(np.mean(acc[h]["err"]) * 100),
               "sharp_cm": float(np.median(1.96 * 2 * sg.mean(1)) * 100),
               "crps_cm": crps_gaussian(np.asarray(acc[h]["mu"]).ravel(),
                                        sg.ravel(),
                                        np.asarray(acc[h]["y"]).ravel()) * 100}
        for p in LEVELS:
            k = _norm.ppf(0.5 + p / 2)                 # two-sided z for level p
            row[f"cov{int(p*100)}"] = float(np.mean(np.abs(z) <= k))
        out[h] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/general4.pt")
    ap.add_argument("--worlds", type=str,
                    default="object,ball,pendulum1,pendulum2,pendulum3,nbody2,nbody3")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--watch", type=float, default=0.35)
    ap.add_argument("--horizons", type=str, default="12,30,60,120,240")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    mem = load_mem(args.ckpt, args.device)
    if not mem.probabilistic:
        raise SystemExit(f"{args.ckpt} is a deterministic checkpoint — it has no cone.\n"
                         f"Train with --prob to get one.")

    print(f"ckpt={args.ckpt}  arch={mem.arch}  watched={args.watch:.0%} of episode")
    print("\nCALIBRATION — coverage should MATCH the nominal level at every horizon.")
    print("  cov50 -> 0.50 and cov90 -> 0.90 is perfect.  Below = overconfident.\n")
    hdr = f"{'world':<11}{'horizon':>9}{'cov50':>8}{'cov90':>8}{'err(cm)':>10}" \
          f"{'band(cm)':>10}{'CRPS(cm)':>10}   calibration"
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for w in [x.strip() for x in args.worlds.split(",")]:
        r = evaluate(w, mem, args.episodes, args.seed, args.max_steps, horizons, args.watch)
        if r in (None, "deterministic"):
            continue
        for h in horizons:
            if h not in r:
                continue
            e = r[h]
            d90 = e["cov90"] - 0.90
            verdict = ("well calibrated" if abs(d90) <= 0.07 else
                       "OVERCONFIDENT" if d90 < 0 else "underconfident")
            print(f"{w:<11}{h*DT:>8.2f}s{e['cov50']:>8.2f}{e['cov90']:>8.2f}"
                  f"{e['err_cm']:>10.1f}{e['sharp_cm']:>10.1f}{e['crps_cm']:>10.2f}"
                  f"   {verdict}")
            summary.append((w, h, e))
        print()

    print("=" * 78)
    print("DOES THE CONE KNOW WHICH WORLDS ARE PREDICTABLE?")
    print("=" * 78)
    print("Band width at the longest horizon, relative to the shortest. A chaotic world")
    print("must FAN OUT far more than an integrable one; equal growth means the model")
    print("learned an average uncertainty instead of the physics.\n")
    print(f"{'world':<11}{'band @'+str(horizons[0]):>12}{'band @'+str(horizons[-1]):>13}"
          f"{'growth':>10}")
    for w in sorted({s[0] for s in summary}):
        rows = {h: e for ww, h, e in summary if ww == w}
        if horizons[0] in rows and horizons[-1] in rows:
            a, b = rows[horizons[0]]["sharp_cm"], rows[horizons[-1]]["sharp_cm"]
            print(f"{w:<11}{a:>12.1f}{b:>13.1f}{b/max(a,1e-9):>9.1f}x")


if __name__ == "__main__":
    main()
