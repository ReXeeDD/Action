# CLAUDE.md — Action project: full build log & guidance

This file is the running record of **what we built, what we changed, and *why*** — start
to present. It is meant for future sessions (and future-you) to pick up with full context.
For the short pitch, see `README.md`. For durable cross-session notes, see the memory dir
at `C:\Users\albin\.claude\projects\D--zt-Action\memory\`.

---

## 1. What this project is

`D:\zt\Action` is a hobby project inspired by the **principle of least action** and the
**path-integral** picture (from a Veritasium video): a physical object doesn't have *one*
future, it has a **cloud of possible futures**. Goal: train neural nets to **predict the
future motion** of objects (starting with a falling leaf) inside a physically real MuJoCo
simulation, and render the uncertainty **honestly** as a **cone of futures** — sharp near
term, blossoming as chaos/uncertainty takes over. Later: generalize to N-body, thrown/
bouncing ball, ball rolling downhill.

**Not reinforcement learning.** This is *supervised learning of a dynamics / world model*
(predict what *will* happen), not a policy (what to *do*). PPO — the sibling project at
`D:\zt\drone-ping-pong` — is the opposite thing. The user correctly intuited PPO can't do
this.

**Prior art (told honestly):** not novel science. PETS (2018) ≈ our ensemble cone; World
Models / PlaNet / Dreamer = latent world models; LNN/HNN/Neural-ODE = the planned "Action"
net; GNS = graph sims. The value here is craft, learning, and the honest framing.

---

## 2. The big pivot (2026-07-22)

The user's *original dream*: a **webcam** "mirage of the future" — drop a real leaf, see its
predicted path/landing painted over live video. After an honest reality check — **three
walls**: (1) monocular depth ambiguity, (2) chaos forbids precise landing prediction of a
fluttering leaf, (3) sim-to-real gap — the user chose to **drop the webcam idea** and put the
camera **entirely inside the MuJoCo simulation**. A virtual camera watches the leaf; we
project the predicted future into that rendered view. This removes all three walls. See
`action/camera.py` (3D→pixel projection, validated 0.8 px) and `action/mirage.py` (the video).

---

## 3. File map

| File | Role |
|------|------|
| `action/leaf_world.py` | The MuJoCo world (MJCF template + `LeafWorld` episode generator). Leaf physics, cameras, the imposed sway forcing. |
| `action/generate_data.py` | Parallel data generation (`multiprocessing.Pool`, 8 workers). Saves `ep_XXXXX.npy` + `ep_XXXXX_wind.npy`. |
| `action/dataset.py` | `make_features` (translation-invariant), `Normalizer`, `load_episodes`, `build_supervised` (history→next-step delta). |
| `action/models.py` | `MLPPredictor` (deterministic), `GaussianMLP` (mean+log_std head, `nll`). |
| `action/train.py` | Train the deterministic MLP → `runs/leaf_mlp.pt`. |
| `action/train_ensemble.py` | Train the deep ensemble of GaussianMLPs → `runs/leaf_ensemble.pt`. Bootstrap + input-noise aug. |
| `action/rollout.py` | `load_model`, `predict_future` (autoregressive single line). |
| `action/cone.py` | `load_ensemble`, `sample_futures` (vectorized, physical guard), `coverage_report`. |
| `action/camera.py` | `cam_id`, `project` (world→pixel), validated 0.8 px. |
| `action/mirage.py` | The "ghost of the future" video: truth (green) + prediction/cone (orange) over rendered frames. |
| `action/viz.py` | `plot_true_vs_pred`, `plot_cone`. |

**State (13-dim)** = free-joint `qpos`(pos 3 + quat 4) ++ `qvel`(lin 3 + ang 3). Indices in
`leaf_world.py`: `POS=0:3, QUAT=3:7, LINVEL=7:10, ANGVEL=10:13`. Timestep 0.004 s, 2 substeps
→ effective **dt = 0.008 s** per stored frame.

---

## 4. The staircase (status)

1. **World + data generator** — ✅ done. Randomized wind/density/mass/spin; imposed sway.
2. **Baseline MLP predictor** — ✅ done. History window → next-step delta, translation-
   invariant, episode-level split. ~125× better than "no motion" at 1 step; autoregressive
   rollout is the weak link (see §7).
3. **Cone of futures** — ✅ done. Deep ensemble (5 GaussianMLPs) + vectorized sampler +
   physical rollout guard (clamp to bounds, freeze-at-ground). The honest long-range answer.
4. **Camera mirage (in-sim)** — ✅ done. Side cam = watch the whole descent; chase cam =
   leaf-centered. Best demo = **side cam**.
5. **Memory / attention model** — 🔜 NEXT (see §7, §8). Sequence model over a long lookback.
6. **Lagrangian/Hamiltonian "Action" net** — ⬜ planned. Dissipative (drag) least-action net
   is the under-explored novel corner.

---

## 5. The physics, and the two reworks

### 5a. Original leaf
Thin **box** geom in MuJoCo's **ellipsoid fluid model** (real drag/lift/Magnus). Problem:
MuJoCo's fluid model is *quasi-steady* — it **cannot flutter/tumble**; it stably parachutes a
flat plate and damps spin. So a bare leaf just glides.

### 5b. Rework #1 — "make it look and fall like a leaf" (user: "looks like a ball, too slow, diagonal")
- **Shape:** box → **flat ellipsoid** geom (`type="ellipsoid"`, looks like a leaf oval).
- **Mass/drop:** heavier (~2.2 g) + lower drop (2.5–3.5 m). **Key physics insight told to
  user:** terminal velocity ≈ √(2mg/(ρ·Cd·A)), so **lighter = SLOWER** (mass in numerator);
  the user's "made it lighter, still slow" was correct physics, not a bug.
- **Flutter:** since the fluid model won't flutter, we **impose** a side-to-side sway as a
  horizontal oscillating force (randomized amplitude/freq/phase per episode) + (originally)
  turbulent noise → `_apply_forcing()`.
- **Result:** fall time mean/median **1.7 s** for ~3 m drop (leaf-like), drift ~1.4 m, flutter
  ratio ~1.30. The green truth path shows a visible **S-shaped sway**.
- **Marker:** big yellow dot → subtle **green ring**.
- **Cameras:** added fixed `side` cam and `chase` (track-mode) cam; `<global offwidth/offheight>`
  in MJCF so >640 px offscreen render works on Windows.

### 5c. Rework #2 — determinism (the pivotal change)
**Motivation:** see §6 (Lyapunov finding). We **removed the per-step `rng.normal`** from
`_apply_forcing` and replaced it with a **second incommensurate sinusoid + deterministic
torque**. The forcing is now a pure function of the episode's fixed params and step counter —
no per-step randomness. Per-episode variety still comes from the randomized amplitude/freq/
phase set once in `reset()`. **Why:** the injected RNG was an information-theoretic wall no
model can beat; making it deterministic makes the future *learnable in principle*.

---

## 6. Key findings (measured, not assumed)

### 6a. CPU vs GPU (benchmarked on the user's machine)
- **MuJoCo physics / data gen:** CPU only (GPU idle) → use multiprocessing.
- **PPO (drone project):** CPU faster (env-in-loop, tiny batches). Valid *for PPO*.
- **Our supervised ensemble training:** **GPU ~5× FASTER** (CPU 14.9 ms/step vs GPU 2.5 ms).
  The earlier "train on CPU" assumption was **wrong for offline supervised training** —
  corrected. 4 GB VRAM is plenty. Scripts accept `--device`.

### 6b. Prediction window (user-measured)
Single-line prediction is accurate for only **~0.4 m of a ~3 m drop** (~13 %, ≈0.2 s) — a
constant-**time** window that *slides* with the leaf (same at 1 m or 10 cm up). This is the
number to beat.

### 6c. THE Lyapunov finding — the window was self-inflicted
After making the forcing deterministic (§5c), we measured chaos directly: run two falls
perturbed by ε in x, **same** deterministic forcing, watch separation grow.
**Result across all seeds & ε (1e-6 … 1e-2): separation factor = EXACTLY 1.00 — no
divergence. Lyapunov exponent ≈ 0. This sim is NOT chaotic.**
→ The entire 0.4 m window was **100 % self-inflicted by the injected RNG** (Wall 1). There is
essentially **zero deterministic chaos** (Wall 2), because MuJoCo's quasi-steady fluid model +
rigid body is neutrally stable.
**Honest caveat told to user:** *real* leaves ARE chaotic; this sim is not. For a genuinely
hard-but-learnable problem, drive the flutter with a deterministic **chaotic** system
(Lorenz/Rössler/double-pendulum) → positive Lyapunov, real finite horizon, still a
deterministic function of an inferable hidden state (where Koopman / reservoir / latent-memory
methods earn their keep).

### 6d. Determinism was necessary but NOT sufficient (measured after retrain)
Regenerated deterministic data + retrained. The window **did not open**:
- Plain-MLP autoregressive rollout now **explodes** (→1e18) — pure compounding error, no guard.
- Guarded ensemble mean-path: still ~**0.29 m** window (26 cm error at 0.25 s).
**Diagnosis (the important part):** 1-step accuracy is excellent (val 0.0082), but the sway
force has period ~1.6 s and the model sees only a **6-frame / ~48 ms** window — **too short to
contain even 3 % of one sway cycle.** The model can't infer *where in its sway* the leaf is, so
it rolls forward blind to the forcing. The info to predict the whole fall now **exists**
(deterministic world) but is **not reachable from 48 ms of history.**
→ This is the measured, concrete motivation for the **memory model** (§7).

---

## 7. Why the memory model is the next step (the user's idea, validated)

The user proposed: give the model **memory / attention over the past** — "look at past, look at
present, predict the future" — like a transformer attending over history. This is exactly right,
and §6d turns it from a nice idea into a *measured requirement*:

- Our current model *does* peek at 6 frames, but it **flattens** them into one vector and feeds
  an MLP — the weakest possible memory, only ~48 ms, no sense of sequence.
- **Takens' theorem:** for a deterministic system you can reconstruct the hidden state from a
  long-enough time-delay embedding. The sway phase/frequency + wind/density are recoverable
  from a **long** lookback — but not from 48 ms.
- **System identification / observability:** watching how the leaf *has been* moving lets the
  model back out the hidden environment (like reading wind off a flag), then predict far better.
  Expect the classic **filter-converging** behavior: fuzzy at first, tightening as the leaf
  falls and the model "figures out the environment." We will test this by plotting **error vs
  fall progress** and expect it to drop.

**Honest caveat on the user's "environment is predictable because it changes smoothly":** true
*only* if the environment evolves deterministically (or is constant, as ours is now). Smoothness
bounds the *rate* of change (narrows the cone) but does NOT remove randomness — a random-walk
wind is smooth yet its future is irreducibly uncertain. So: constant wind (current) → fully
predictable once inferred; deterministic-evolving wind → predictable, memory model shines;
stochastic gusts → irreducible floor, cone stays necessary.

**Plan:** add `models.py:SeqPredictor` — a sequence encoder (GRU/LSTM, or a small **Transformer
encoder** = the "attention over everything" version) over a **longer** history, optionally
**seq2seq** (predict the whole future path in one shot → also kills compounding error). Compare
its window head-to-head against the MLP baseline on the same data.

---

## 8. Open threads / next steps

1. **Memory model** (§7) — the current priority.
2. **Widen the fixable part** — multi-step / rollout training (train on the model's own
   multi-step predictions), and/or a physical guard on the plain MLP so it stops exploding.
3. **Optional: a genuinely hard version** — Lorenz-driven deterministic chaotic flutter
   (positive Lyapunov) so advanced methods (Koopman, reservoir computing à la Pathak 2018,
   latent-memory) have something real to beat.
4. **Flip ensemble training to GPU by default** (5× faster; §6a).
5. **Phase 6: dissipative Lagrangian/Hamiltonian "Action" net.**
6. **Deterministic-evolving wind** to showcase the "understands the environment as it falls"
   effect at full strength.

---

## 9. How to run

```bash
# 1. generate falls (parallel, 8 workers)
python -m action.generate_data --episodes 1500 --out data/leaf --workers 8

# 2. train
python -m action.train          --data data/leaf --out runs/leaf_mlp.pt --epochs 40
python -m action.train_ensemble --data data/leaf --out runs/leaf_ensemble.pt --epochs 30

# 3. cone report (does reality stay inside the cone?)
python -m action.cone --data data/leaf --ckpt runs/leaf_ensemble.pt --samples 200

# 4. the mirage video (side cam = watch the whole descent)
python -m action.mirage --episode 25 --ensemble runs/leaf_ensemble.pt \
    --samples 80 --stride 1 --temperature 1.3 --cam side --out runs/mirage_side.mp4
```

Good centered fluttery demo episodes: 25, 53, 252, 144.

---

## 10. Environment & gotchas

- Python 3.11, mujoco 3.5, torch 2.6+cu124, numpy 2.4, gymnasium 1.0. No install needed;
  stack shared with `D:\zt\drone-ping-pong`.
- MuJoCo offscreen render >640 px needs `<global offwidth/offheight>` in MJCF (Windows).
- Renderer teardown throws a harmless `_gl_context` AttributeError at exit — filter with
  `grep -v _gl_context`, ignore.
- `load_episodes` must **exclude** `ep_*_wind.npy` context files (already handled).
- `data/`, `runs/`, `*.mp4`, `*.pt` are git-ignored (regenerable).

---

## 11. Edit log (chronological, why each change)

- **World + pipeline built** — leaf in ellipsoid fluid model; parallel data gen; MLP baseline;
  translation-invariant features; predict deltas; episode-level split.
- **Cone built** — deep ensemble of GaussianMLPs. *Fixed autoregressive blow-up* with (a)
  input-noise augmentation in training, (b) physical rollout guard (clamp to bounds +
  freeze-at-ground) in `cone.py`.
- **Camera + mirage** — `camera.py` projection (validated 0.8 px); `mirage.py` ghost video.
  Switched fixed→chase cam when a high-drift glider left frame; later added `side` cam back for
  the physics-rework demo (leaf now stays framed).
- **`.gitignore` added** — ignore regenerable artifacts.
- **CPU/GPU corrected** — benchmarked; GPU 5× faster for our training (I had wrongly agreed
  with "CPU faster"; owned and corrected it).
- **Physics rework #1** — box→ellipsoid, heavier+lower drop, imposed sway force, green-ring
  marker. Because the fall "looked like a ball, too slow, diagonal."
- **`mirage.py`** — added `--cam {chase,side}` to `main()`.
- **Lyapunov diagnosis** — made forcing **deterministic** (removed per-step RNG), measured
  Lyapunov ≈ 0 → the window was self-inflicted. (§6c)
- **Deterministic regen + retrain** — window did NOT open; diagnosed 48 ms history too short to
  infer the ~1.6 s sway → **memory model is the required next step**. (§6d, §7)
- **This CLAUDE.md written.**
```
Remember: report outcomes faithfully. When a prediction turns out wrong (e.g. "determinism
will open the window"), say so plainly and explain what the measurement actually showed.
```
