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
5. **Memory / attention model** — ✅ DONE and it worked (see §12). Transformer context
   encoder + GRU rollout decoder, trained with a multi-step position loss.
6. **Live demo** — ✅ DONE (`action/live.py`): fresh random drops, blind prediction,
   predicted landing called before it happens.
7. **Multi-world scale-up** — 🔜 IN PROGRESS (see §13): ball, n-link pendulums, n-body.
8. **Lagrangian/Hamiltonian "Action" net** — ⬜ planned. Dissipative (drag) least-action net
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

## 12. RESULTS — the memory model (the main outcome so far)

The user's idea: don't feed the net the physics (mass, wind, temperature) — let it **watch**
and build its own memory of how *this* object interacts with *this* air, then predict from it.
Built as `TrajContextEncoder` (Transformer attends over a long history → a 96-d context vector
`z`) + `SeqPredictor` (GRU decoder that re-reads `z` at **every** rollout step), trained with a
**multi-step position loss** (`action/train_memory.py`).

### What actually moved the needle (in order of impact)

| Change | whole-fall val RMSE | window @50% watched |
|---|---|---|
| v2, Fourier time-clock mistuned to 0.05–1.2 rad/step | plateaued 53 cm | — |
| v2.1, **clock retuned to the leaf's real sway band** (0.008–0.25) | 34.4 cm | 0.40 m |
| v3, **attention pooling + decoder 128→256 + encoder 64→96/3 layers** | **30.6 cm** | **0.79 m** |

Three lessons worth keeping:

1. **Train on the metric you care about.** Per-step *delta* loss let tiny biases integrate into
   huge drift. Adding a cumulative **position loss in metres**, differentiated through the whole
   rollout, was a step change.
2. **Give a periodic system a clock.** The sway force is a function of the step counter. A bare
   GRU cannot hold that phase over 200 autoregressive steps. A Fourier time-clock fixed it — but
   only once its frequency band was **matched to the actual physics** (`leaf_world` drives at
   `f*{0.7..2.7}` with `f∈[0.025,0.045]`). A mistuned clock is worthless; that one retune took
   53 cm → 34 cm.
3. **Measure the right thing.** "Window from 12 frames watched" is **information-limited** —
   12 frames is ~6 % of the 1.6 s sway cycle, so no model can identify the environment yet. It
   sat at ~0.25 m no matter what we did. The honest metrics are **window vs. how much was
   watched** and the **sharpen curve**, both of which improved dramatically.

### The headline
Prediction **sharpens as the object falls**, exactly as the user predicted:

```
watched 20% of fall: WINDOW = 0.45 m ahead
watched 35% of fall: WINDOW = 0.58 m ahead
watched 50% of fall: WINDOW = 0.79 m ahead      (+0.48s error: 29.4 -> 14.6 -> 10.5 cm)
```

### Live demo (`action/live.py`)
Drops **fresh** leaves (random release point over a ±1.5 m zone, random height 2.2–3.8 m, fresh
air/wind/sway — none of it in the training set) and predicts **blind**: at every frame the model
sees only what it has watched, rolls the future forward, and **calls its own landing spot** by
cutting the predicted path at the first predicted ground contact. Real run:

```
drop 1: from (-0.69,-1.38) at 2.23m | landing call after 25%: 55.1cm  50%: 16.9cm  75%:  5.9cm
drop 2: from (+0.94,+1.24) at 3.17m | landing call after 25%: 55.5cm  50%: 44.8cm  75%: 15.7cm
drop 3: from (+0.13,+1.31) at 3.51m | landing call after 25%: 59.4cm  50%: 39.5cm  75%:  8.6cm
drop 4: from (-1.49,+1.07) at 2.25m | landing call after 25%: 41.1cm  50%: 28.9cm  75%:  6.4cm
```
Monotonic on every drop. Generalization to unseen release points/heights comes free from
**translation invariance** — the model never sees absolute coordinates.

### Why it works this well (and the honest caveat)
The environment is only ~8 hidden constants (density, wind×3, mass, sway amp/freq/phase), fixed
per fall; motion is the integral of force, so the trajectory is a **fingerprint** of them — this
is learned system identification. And because we made the world **deterministic** (Lyapunov ≈ 0,
§6c), once those constants are pinned the rest of the fall is *determined*, and a periodic force
is computable arbitrarily far ahead — which is why long-horizon predictions work, not just short
ones. **Caveat, stated plainly: it predicts this well largely because we removed the randomness.
A real leaf, with genuine turbulence, would not be this predictable and no model could make it
so.** That is why the next phase adds genuinely chaotic worlds.

---

## 13. Multi-world scale-up (`action/worlds/`)

Moving beyond the leaf to many systems, with **real** physics (no imposed forcing anywhere).

| File | World | Physics |
|---|---|---|
| `worlds/ball.py` | `ball` | solid ball launched in **any** direction; real contacts, restitution, friction, drag |
| `worlds/pendulum.py` | `pendulum1..N` | n-link chained pendulum; exact rigid-body dynamics under gravity |
| `worlds/nbody.py` | `nbody2..N` | n point masses under **real Newtonian mutual gravitation**, computed pairwise and applied via `xfrc_applied` (MuJoCo has no inter-body gravity), Plummer softening, zero uniform gravity, contacts off |

`worlds/base.py` defines the interface; `worlds/__init__.py` is the registry (`make_world`).
A universal state convention (`state = qpos ++ qvel`) plus two index maps — `pos_groups` (3D
positions, made relative for translation invariance) and `quat_groups` (quaternions to
renormalize each step) — let the pipeline stay physics-agnostic. `generate_data.py --world <name>`
writes a `meta.json` with that layout.

### Measured chaos (separation growth from a 1e-6 perturbation over 7.2 s)

| world | growth | verdict |
|---|---|---|
| ball | 1× | predictable (ballistic) |
| pendulum1 | 28× | predictable (**integrable**) |
| nbody2 | 5× | predictable (**Kepler — integrable**) |
| pendulum2 | 73× | mildly chaotic (mixed phase space) |
| nbody3 | 29× | mostly regular at this timescale |
| pendulum3 | **679,000×** | **CHAOTIC** |
| nbody4 | **4,400×** | **CHAOTIC** |
| pendulum4 | **1,652,000×** | **CHAOTIC** |

This is physically correct — the two *integrable* systems stay predictable and chaos switches on
with the third link / fourth body. **This is the scientific complement to the leaf:** there the
horizon was self-inflicted and removable; here it is a genuine physical wall. Calibration notes:
N-body needed `G=15` (at `G=1` an episode covered less than one orbit), and pendulum damping had
to be cut to ≤0.004 or friction suppressed the very chaos the world exists to show.

### Still to do for the scale-up
The training pipeline is still leaf-shaped (hardcoded `STATE_DIM=13`, positions at `[0:3]`,
quaternion at `[3:7]`). To train these worlds, `dataset.py`, `train_memory.py` and
`models.py::SeqPredictor.decode` must read `pos_groups`/`quat_groups` from `meta.json` instead.
Then: per-world models first, and a **prediction-horizon vs. measured-Lyapunov** comparison
across all worlds — the honest headline result this project has been building toward.

---

## 14. The general predictor — universal representation + universal physics

Two corrections from the user reshaped this phase:

1. *"Limit the prediction to ONE object at a time"* — not multi-body scene prediction. But
   **any new object, in any environment, must work with no retraining**, and an episode runs
   **until the body comes to rest**.
2. *"The physics must be the same for every body — otherwise how can we import a general
   entity and just run it?"* — which caught a real design violation (see below).

### 14a. Universal representation (`action/entities.py`)
Every object in every world is described by the **same 13 numbers in world coordinates**:
position(3) + quaternion(4) + linear velocity(3) + angular velocity(3), read straight from
MuJoCo. Because the model's input/output size never changes, a brand-new object type is just
more data in the same format — **that is what makes "no retraining" true**. `entity_attrs()`
also exposes static identity (mass, inertia, size). `world.target_rollout()` emits `(T,13)` for
the single target body and stops when it comes to rest. `target_index` = tip link for pendulums,
0 otherwise. Every world's `meta.json` therefore reports `state_dim=13`, `pos_groups=[[0,1,2]]`,
`quat_groups=[[3,4,5,6]]` — **always**.

**Key consequence:** the existing `train_memory.py` / `SeqPredictor` (13-dim, pos `[0:3]`, quat
`[3:7]`) works UNCHANGED on every world. No pipeline generalization was needed.

### 14b. UNIVERSAL PHYSICS (the important correction)
I had given the ball and the leaf **hand-written force models** (`-c*v` drag; a bespoke
`_aero()`). That silently broke the whole premise: an imported cube or sheet of paper would get
**no aerodynamics at all**, because nobody had written a model for it. Both were **deleted**.

Now there is one set of laws, applied by the engine to any geometry:
gravity · contacts + friction + restitution · fluid drag/lift · Newtonian mutual gravitation.
`MujocoWorld.physics_options()` emits the shared `<option>` block; `MujocoWorld.apply_forces()`
holds the *only* remaining force law (mutual gravitation, off unless `grav_const>0`). Worlds
differ **only in environment parameters** (air density 1.2 vs vacuum, gravity 9.81 vs 0,
mutual-G 0 vs 15) — legitimate, and exactly what the model must infer.

`worlds/generic.py` proves it: `object` / `object_{sphere,box,plate,capsule,cylinder}` drops an
arbitrary shape with **zero special-case code**. Behaviour differs purely from geometry —
plate travels 1.75 m (glides), box settles in 2.9 s, sphere rolls 13.6 m.

**Honest cost:** without its bespoke aero the leaf now *glides* instead of fluttering. The
flutter was the fake part; this is what universal physics actually produces for a plate.

### 14c. Measured chaos (real physical horizons, unlike the leaf)
Separation growth from a 1e-6 perturbation over 7.2 s: ball 1x, pendulum1 28x, nbody2 5x (both
**integrable → predictable, correct**), pendulum2 73x, nbody3 29x, **pendulum3 679,000x,
nbody4 4,400x, pendulum4 1,652,000x = CHAOTIC**. Chaos correctly switches on with the third
link / fourth body.

### 14d. Physics audit results
Gravity 9.81 m/s² (measured free-fall −9.765 with drag). Pendulum energy conserved to
**0.001–0.018%** (the integrator is faithful). N-body energy drift **0.5–7.6%** after fixes.
All worlds measured **omnidirectional** (direction concentration |R| ≈ 0.01–0.15).

**Caveat that still stands:** translation invariance is *architecturally guaranteed* (positions
fed relative); **direction is not** — the model sees absolute orientation and world-frame
velocities, and gravity singles out −z, so all-heading generalization is *learned from data
coverage*, not enforced. Recommended (not yet implemented): **yaw augmentation** — randomly
rotate each training trajectory about the vertical axis.

### 14e. Errors hit and how each was fixed
| Problem | Cause | Fix |
|---|---|---|
| Wrong body velocities everywhere | `data.cvel` is a **com-based spatial** velocity referenced to the subtree CoM, not a body's own linear velocity | use `mj_objectVelocity(..., flg_local=0)` |
| Spins hit **800 rad/s**, speeds 29 m/s (energy *injected*) | damping torque computed from free-joint `qvel[3:6]` (**body** frame) applied via `xfrc_applied` (**world** frame) | read + apply in world frame |
| Ball never came to rest (rolled 12+ s, 15 m) | MuJoCo's rolling-friction term barely decelerates | explicit rolling resistance (later removed for universality; episodes now cap at `max_steps`) |
| Rest never detected for a rolling ball | a rolling sphere spins `v/r`, so a tight spin bound is unsatisfiable | loosened `rest_spin` to 3.0; linear speed is the real criterion |
| N-body energy drift **255%** | forces held constant across RK4's internal stages | `timestep=0.0005`, `n_substeps=16` |
| N-body drift still 105% | close approaches → enormous forces | `SOFTENING` 0.05→0.10 + end episode at `COLLIDE_R` (they'd have collided) |
| N-body: nothing interesting happened | `G=1` → an episode covered **less than one orbit** | `G=15` |
| `nbody4` episodes ended after 8 steps | bodies spawned already inside the collision radius | resample/expand spawn until separations > `4*COLLIDE_R` |
| N-body **direction bias** (\|R\|=0.60) | `np.sort()` on ring angles always gave body 0 (the target) the smallest angle | removed the sort → \|R\|=0.07 |
| Pendulums strictly **planar** (y-spread 0.0000) | fixed hinge axis `0 1 0` | randomize the swing plane per episode |
| Pendulum chaos suppressed | joint damping too high | damping ≤ 0.004 |
| Pendulum entity count inflated | massless `anchor` wrapper body | chain attached directly to worldbody |
| Leaf aero attempt: fell in 0.86 s, no flutter | pitch-moment **sign wrong** — drove the leaf edge-on where drag is 12× lower | verified sign by experiment (negative = restoring); later removed entirely |
| Leaf aero attempt: spin-up, no flutter | centre-of-pressure offset acted as a **pinwheel** (maple-seed autorotation) | reduced, then whole model removed for universality |
| `live.py` reported landing error of "0.5 cm" | measured at the **last frame**, when the body had already landed and only 1 step remained — trivially accurate and **misleading** | report the call after 25/50/75% watched |
| `live.py` video frozen after generalizing | removed the per-frame pose-set; the universal 13-dim state **cannot** pose a pendulum (its `qpos` is joint angles) | store `qpos/qvel` snapshots alongside the trajectory |
| Kaggle OOM at batch 8192/4096/3072 | **two independent** sources: encoder attention is O(L²·batch), decoder backprop is O(fut_cap·batch·dec_hidden) | gradient checkpointing (`--ckpt-chunk`), lower batch, `expandable_segments:True` |
| Mixed-world dataset filename collisions | every world wrote `ep_00000.npy` | `--tag` prefix (defaults to world name) → many worlds share one `--out` dir |
| **All pendulum data physically impossible** (`d(pos)/dt` disagreed with `linvel` by 79–100%) | position read from `data.xpos` (body **frame origin**) but velocity from `mj_objectVelocity` (**centre of mass**) — different points | `xpos` → `xipos` (§15b) |
| **`pendulum1` target never moved** (path span 0.0000 m) | its body frame origin *is* the fixed pivot | same fix — the CoM does move |
| Pendulum skill collapsed past ~1 s **even for the integrable pendulum1** | trained `--fut-cap 120` (0.96 s) but rendered 260 frames (2.08 s) — an oscillator cannot extrapolate phase past its trained horizon | `--fut-cap 240` (§15c) |
| Orange predicted path frozen as a **single point** on pendulums | `live.py` hardcoded `GROUND_Z = 0.03`, but pendulum ground planes sit at −0.68/−1.03/−1.30 and the bob swings below z=0 → all 260 frames "underground" → path cut to 1 | read the plane out of the model; cut at `ground + 0.03`; no plane ⇒ no cut (§15d) |
| Pendulums ignored by the loss despite being 29% of the data | loss in raw metres² — a 9 m ball outweighs a 0.4 m pendulum ~500× | scale-normalized position loss (§15f) |
| The 35 cm val plateau was uninterpretable | one blended RMSE over nine worlds with wildly different irreducible error | per-world validation; checkpoint on **mean of per-world RMSE** (§15f) |
| Training slow (150 s/epoch) | 240 *sequential* GRU steps per batch — sequential depth, not arithmetic | flow-map decoder, all horizons in parallel → 50 s/epoch (§15e) |
| `predict()` silently corrupted the caller's trajectory | `torch.from_numpy` shares memory and `.to("cpu")` is a no-op, so the in-place anchor subtraction wrote through | `.clone()` |
| `_diagnostics` crashed on any dataset < 1400 episodes | hardcoded `eps[1400:1470]` → empty slice | use the tail, `eps[-70:]` |

### 14f. Commands
```bash
# generate - all worlds into ONE folder (tagged filenames prevent collisions)
python -m action.generate_data --world object    --episodes 2500 --out data/all --workers 8
python -m action.generate_data --world ball      --episodes 1200 --out data/all --workers 8
python -m action.generate_data --world leaf      --episodes 1200 --out data/all --workers 8
python -m action.generate_data --world pendulum2 --episodes 1000 --out data/all --workers 8
python -m action.generate_data --world pendulum3 --episodes 1000 --out data/all --workers 8
python -m action.generate_data --world nbody3    --episodes 1000 --out data/all --workers 8

# train ONE general model  (--stride raises/lowers the window count; big datasets need
# FAR fewer epochs: ~1.46M windows at stride 3 is ~380 batches/epoch = 5-15 min/epoch)
python -m action.train_memory --data data/all --out runs/general.pt \
    --epochs 15 --batch 3840 --fut-cap 120 --hist-cap 160 --ckpt-chunk 24 \
    --stride 10 --device cuda --lr 2e-3

# use it
python -m action.live --memory runs/general.pt --world object --drops 5 --show
python -m action.live --memory runs/general.pt --world ball --drops 5 --out runs/live_ball.mp4
python -m action.train_memory --measure runs/general.pt --data data/all
```
Note: training prints nothing until an **epoch completes** — a long silence after the
`train windows ...` line is normal, not a hang.

**The real test still to run:** hold out an entire world (e.g. `object_capsule`, `pendulum4`),
train without it, and measure on it. That turns "generalizes" from a claim into a result.

### 14g. Superseded
`action/models_general.py` — a multi-entity graph/attention net (permutation-equivariant,
verified on 1–12 bodies). Built before correction #1; kept in case true multi-body scene
prediction is ever wanted, but it is **not** the current direction.

---

## 15. "The pendulum is trash" — autopsy, and what it actually was

The first general model (`runs/general.pt`) handled the ball and free-falling objects
well but was useless on pendulums. The tempting explanation was **chaos** (pendulum3 =
679,000×, §14c). That explanation was **wrong**, and the decisive test proved it:
`pendulum1` is **integrable** — 28×, no chaos, provably predictable — and it failed just
as badly. Three separate bugs, none of them physics.

### 15a. The measurement tool (`action/diagnose_worlds.py`)
Raw centimetres are meaningless across worlds (a ball travels 8.8 m, a pendulum tip lives
inside a 1 m sphere), so this scores against baselines the model must beat:

    skill = 1 - err_model / err_freeze        freeze  = "it never moves again"
                                              const-v = "it keeps its current velocity"

skill 1.0 = perfect, 0.0 = no better than a rock, negative = worse than assuming it
stopped. It also dumps per-world motion scale against the single global normalizer.
**This tool is how every claim below was established.** Note that `freeze` is a *strong*
baseline for a pendulum (bounded orbit), so pendulum skill is harder-won than ball skill.

### 15b. BUG 1 — position and velocity described different points
`entities.entity_state` read position from `data.xpos` (the body **frame origin**) but
velocity from `mj_objectVelocity`, which reports **at the centre of mass**. For a free
body those coincide (the geom is centred on its frame), so leaf/ball/object were fine.
A pendulum link's frame sits at its **hinge**, its CoM half a link away. Measured
`|d(pos)/dt − linvel|`:

| world | mismatch |
|---|---|
| object / ball | 4.3% (contact impulses — fine) |
| leaf | 0.6% |
| **pendulum1** | **100.0%** |
| **pendulum2** | **79.1%** |
| **pendulum3** | **79.5%** |

Worse, `pendulum1`'s target frame origin **is** the fixed pivot: its recorded path span
over 400 frames was **0.0000 m** while the true chain tip swept 0.7009 m. Every
pendulum1 episode was a stationary point annotated with a velocity of 0.78 m/s —
physically impossible data that no model can learn. Fix: `xpos` → `xipos`, one line,
also the physically correct choice since F=ma holds at the CoM. Mismatch → 0.1/0.2/0.5%,
free-body worlds untouched. **All pendulum data had to be regenerated.**

### 15c. BUG 2 — the horizon collapse was a training artifact
Trained with `--fut-cap 120` (0.96 s) but rendered with `n_ahead=260` (2.08 s). Skill
held to 0.96 s and collapsed past it **on every pendulum, including the integrable
one** — exactly at the trained-rollout boundary. Ballistic worlds extrapolate past it
benignly (a parabola stays a parabola); an oscillator does not, because phase must be
maintained. Fix: `--fut-cap 240`. Combined with 15b:

| @1.92 s | before | after |
|---|---|---|
| pendulum1 | +0.01 (15.0 cm) | **+0.38 (8.9 cm)** |
| pendulum2 | +0.05 (36.0 cm) | **+0.40 (20.5 cm)** |
| pendulum3 | **−0.46** (50.7 cm) | **+0.32 (23.6 cm)** |

No negative skill anywhere afterwards; errors roughly halved on every pendulum.

### 15d. BUG 3 — the renderer threw the prediction away
Even after both fixes the video showed a **frozen orange cross** on the pendulum. Cause:
`live.py` hardcoded `GROUND_Z = 0.03` and cut the predicted path at the first frame
below it. But a pendulum's ground plane is at `-(chain length)-0.3` — measured **-0.676
/ -1.031 / -1.298** — and the bob swings *below z=0* for half of every cycle. So all 260
predicted frames counted as "underground", the path was truncated to **1 point**, and the
marker froze on the current position. **The model was predicting correctly the whole
time; the renderer discarded it.** Fix: read the ground plane out of the model
(`live.ground_level`) and cut at `ground + 0.03`; worlds with no plane geom (n-body) are
never cut. Ball/object behaviour is bit-identical (their plane is at z=0). The "predicted
landing" caption is now suppressed for worlds that never land.

### 15e. Faster training — the flow-map decoder (`models.DirectTrajPredictor`)
`SeqPredictor` integrates 240 *sequential* GRU cells per batch, each a tiny matmul: the
GPU idles, gradient checkpointing is forced on to fit memory, and error compounds along
the chain. **The sequential depth, not the arithmetic, was the cost.**

But no recurrence is needed. For a deterministic system the future is a *function* of
(state, environment, elapsed time) — the **flow map** `x(t0+k·dt) = Φ(k; x0, θ)`. The
encoder already infers θ as the context `z`, so Φ can be learned directly and every
horizon evaluated **in parallel**: one wide batched MLP instead of a 240-deep chain.
This is the solution-operator view (DeepONet / neural operators) rather than the
numerical-integrator view. The decoder gets two time bases — the physical 0.008–0.25
rad/step band (oscillation, as the GRU clock had) plus NeRF-style octaves on `k/k_max`
(smooth growth of displacement) — and predicts displacement in units of a fitted
per-horizon scale table so its output stays O(1) from 1 step to 240.

Head-to-head, identical data/settings/batch, curriculum off:

| arch | s/epoch | params | 3-epoch mean-world RMSE |
|---|---|---|---|
| `gru` | 150 / 152 / 212 | 0.70M | 14.7 cm |
| **`direct`** | **50 / 51 / 53** | 4.66M | **14.8 cm** |

**~3× faster at identical accuracy**, with 6.6× more parameters and no checkpointing.
After only 3 epochs on pendulum data it already beat the 12-epoch `general2.pt`:
pendulum1 +0.38 → **+0.66**, pendulum2 +0.40 → **+0.51** at 1.92 s.

### 15f. Accuracy — scale-normalized loss and per-world validation
Two more things were silently wrong for a mixed-world dataset:

* **The loss was in raw metres²**, so a ball travelling 9 m produced ~500× the squared
  error of a pendulum swinging 0.4 m. Pendulums were effectively ignored at 29% of the
  data, and the **chaotic** worlds — largest error, *least* reducible — dominated the
  gradient hardest. Now each window's error is divided by its own RMS displacement
  (floored at `--min-scale` so near-still windows can't blow up), making every world
  contribute comparably. `--no-scale-norm` restores the old behaviour.
* **Validation was one blended number** across nine worlds whose irreducible error
  differs by an order of magnitude — it could not distinguish "stopped learning" from
  "the chaotic worlds hit their physical floor", which is exactly why the 35 cm plateau
  was uninterpretable. Now `load_episodes_tagged` recovers each episode's world from its
  `ep_<world>_<idx>.npy` filename and validation prints **per world**, with the
  checkpoint metric being the **mean of per-world RMSE** so no world can be ignored for
  moving in small numbers.

Also added: AMP (bf16/fp16, auto-on for `direct`), dataloader workers (0 on Windows —
spawn would copy every episode into each worker), and a rollout-length **curriculum**
(`--curriculum 0.35`) that starts short and ramps to full — cheaper early and better
conditioned, since the encoder learns to identify the environment before it is asked to
hold a two-second prediction together.

### 15g. Faster rendering — batched prediction
`live.py` called the model once per frame: ~450 frames × 260 sequential rollout steps, at
batch 1, on CPU. But every frame's prediction depends only on that frame's history —
never on another frame's prediction — and the whole trajectory is simulated *before* the
render loop, so they all batch perfectly. Now one padded `encode` + one `decode` per
chunk (`--pred-batch`), with `--device cuda`. A 446-frame pendulum video renders in
**8.8 s** including simulation and rendering.

`LoadedMemory` (in `train_memory.py`) replaced the old 6-tuple `load_mem` return, wrapping
both architectures behind one `predict` / `predict_batch` API so callers never branch on
`arch`. Old `gru` checkpoints still load (`arch` defaults to `"gru"` when absent).

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
- **Memory model built** — `TrajContextEncoder` + `SeqPredictor` + `train_memory.py`. Killed the
  compounding explosion; confirmed "sharpens as it falls."
- **Streaming memory + train-to-landing** — encoder reads all history so far (padded/masked),
  decoder trained over the full remaining horizon.
- **Regularization** (input-noise aug, denser val stride 20, weight decay 3e-4) — fixed a val
  flatline at 13.2 → 9.7. *But it did not move the window*, which is how we learned the window
  was information-limited, not model-limited.
- **Position loss + Fourier time-clock** — trained on cumulative metres and gave the decoder a
  clock. Then **retuned the clock band to the real sway frequencies** (the mistuned band was
  worthless): 53 cm → 34.4 cm.
- **v3 capacity + attention pooling** — 30.6 cm, window 0.40 → **0.79 m**.
- **Gradient checkpointing** on the rollout (`--ckpt-chunk`) — required to fit v3; two
  independent OOM sources: encoder attention is O(L²·batch), decoder backprop is
  O(fut_cap·batch·dec_hidden).
- **`live.py`** — fresh randomized drops, blind prediction, self-called landing. Its first
  landing-error report was **misleading** (measured at the last frame, when the leaf had already
  landed and only one step remained); fixed to report the call after 25/50/75 % watched.
- **`worlds/` package** — ball, n-link pendulum, n-body under real Newtonian gravitation;
  world-agnostic `generate_data.py --world` + `meta.json`. Measured the chaos spectrum.
- **`entities.py` universal representation** — every object is the same 13 world-frame numbers,
  so a new object type needs no retraining. Single-target `(T,13)` format for all worlds.
- **UNIVERSAL PHYSICS enforced** — deleted the bespoke ball drag and leaf `_aero()` after the
  user correctly pointed out that per-object physics makes importing a new entity impossible.
  Added `worlds/generic.py` (`object_*`) to prove arbitrary shapes work with zero new code.
- **Physics audit** — verified gravity, energy conservation, and omnidirectionality by
  measurement rather than assertion; fixed the velocity-frame, energy-injection, n-body
  integration, pendulum-planarity and n-body direction-bias bugs it exposed (see §14e).
- **`--tag`** on data generation so many worlds share one dataset; **`--world`** on `live.py`.
- **Full error/fix table written up in §14e.**
- **`diagnose_worlds.py`** — per-world skill against freeze/const-v baselines, because raw
  centimetres cannot be compared across worlds. Every §15 claim rests on it.
- **Pendulum autopsy (§15)** — chased "the pendulum is trash" to *three* bugs, none of them
  chaos. The decisive test was `pendulum1`: it is **integrable**, so its failure ruled chaos
  out immediately. Fixed the CoM/frame-origin data corruption, the trained-horizon collapse,
  and the renderer's hardcoded ground plane.
- **Flow-map decoder** (`DirectTrajPredictor`) — dropped the autoregressive rollout for a
  parallel solution operator. 3× faster training at equal accuracy; also removes compounding
  error and the need for gradient checkpointing.
- **Scale-normalized loss + per-world validation** — stopped the big/chaotic worlds from
  owning the gradient, and made the plateau diagnosable instead of a single blended number.
- **Batched prediction in `live.py`** + `--device cuda` — per-frame predictions are mutually
  independent, so a whole video is a handful of batched forward passes.
```
Remember: report outcomes faithfully. When a prediction turns out wrong (e.g. "determinism
will open the window"), say so plainly and explain what the measurement actually showed.
```
