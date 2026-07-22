# Action — learning to predict the future motion of physical objects

A hobby project inspired by the **principle of least action** and the path-integral
picture of nature: an object like a falling leaf doesn't have *one* future, it has a
**cloud of possible futures**. We train networks to project that future in 3D from a
physically real simulation (MuJoCo), and we render the uncertainty honestly.

> The honest truth about chaos: a fluttering leaf, a 3-body system, a bouncing ball —
> these are **chaotic**. No model predicts them to a single point far ahead; that's a
> law of nature, not a limit of neural nets. So the goal isn't a single line — it's a
> *sharp* short-horizon prediction that **blossoms into a cone** as chaos takes over.

## The world is real, not faked

The leaf falls through MuJoCo's **ellipsoid fluid model**: genuine drag, lift, and
Magnus forces. A thin plate tumbles and flutters on its own. Each episode randomizes
wind, air density, leaf mass, and initial spin — the hidden context the network must
infer from motion alone.

State (13-dim) = free-joint `qpos`(pos + quaternion) ++ `qvel`(linear + angular).

## The staircase

1. **World + data** — `leaf_world.py`, `generate_data.py`  ✅ done
2. **Baseline predictor** — MLP, history window → next-step delta, autoregressive
   3D rollout. `models.py`, `train.py`, `rollout.py`, `viz.py`  ✅ done
3. **Cone of futures** — probabilistic / ensemble head → many sampled futures  ⬜ next
4. **The "Action" net** — a Lagrangian / Hamiltonian network that learns the action
   itself, for physically-consistent long rollouts  ⬜ planned

## Run it

```bash
python -m action.generate_data --episodes 400 --out data/leaf   # make falls
python -m action.train        --data data/leaf --epochs 40      # fit predictor
python -m action.rollout      --data data/leaf --ckpt runs/leaf_mlp.pt  # project + plot
```

## Where the baseline stands

One-step prediction is ~70× better than "the leaf doesn't move." Autoregressive
rollout is accurate for ~1 second, then diverges — partly true chaos, partly
compounding single-step error (a known effect fixed by multi-step training loss and
the probabilistic cone in step 3). The error-vs-horizon curve is the honest picture.

## Design notes

- **Translation invariance:** positions fed relative to the current frame, so the net
  never sees an absolute coordinate and generalizes across the whole sky.
- **Predict deltas:** the identity map (no motion) is the trivial baseline to beat.
- **History window:** lets the net infer hidden wind/density from recent motion.
- **Episode-level train/val split:** validation falls are genuinely unseen.
