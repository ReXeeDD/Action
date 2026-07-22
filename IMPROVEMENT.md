# IMPROVEMENT.md — how to take this project to the next level

A prioritised roadmap with a concrete **algorithm** for every item, written so you can
implement each one without further discussion. Read `CLAUDE.md` first for what exists and
why; this file is only about **what to build next**.

Each entry has: **Why** (grounded in something we measured), **Algorithm** (steps you can
code), **Files**, **Expected gain**, **How to verify**, **Effort**.

---

## 0. Where we stand (the honest baseline to beat)

| Quantity | Value | Source |
|---|---|---|
| Whole-fall position RMSE (leaf) | **30.6 cm** | v3 memory model, 30 epochs |
| Prediction window @50% watched | **0.79 m** | `window vs how much it watched` |
| +0.48 s error @50% watched | **10.5 cm** | sharpen curve |
| Live landing call (25/50/75% watched) | **~55 / 17 / 6 cm** | `live.py`, fresh unseen drops |
| Leaf chaos | Lyapunov ≈ 0 (**not** chaotic) | perturbation test |
| Chaotic worlds | pendulum3 679,000×, pendulum4 1.65M×, nbody4 4,400× | perturbation test |

**Three known gaps**, each addressed below:
1. Generalization to a genuinely unseen world is **claimed, not proven** (§0.1).
2. Direction (yaw) invariance is **learned, not enforced** (§0.2).
3. The model is **deterministic** — it outputs one line even for provably chaotic systems,
   where a single line is *guaranteed wrong* past the horizon (§1.1).

---

# TIER 0 — do these first (cheap, high certainty, high information)

## 0.1 Held-out-world evaluation — the missing proof

**Why.** The entire premise is "any new object works with no retraining." We have never
tested it. Everything else is worth less until this number exists.

**Algorithm.**
```
1. Build two dataset folders:
     data/train_worlds/  <- object_sphere, object_box, ball, leaf, pendulum2, nbody3
     data/held_out/      <- object_capsule, object_cylinder, pendulum4, nbody4
   (generate_data.py --world X --out data/train_worlds ... use --tag to share a folder)
2. Train ONE model on data/train_worlds only.
3. Evaluate on data/held_out WITHOUT any fine-tuning:
     for each held-out world w:
         err_w(h) = mean over episodes of ||pred(h) - truth(h)||   for horizons h
         window_w = first h where err_w(h) > 0.10 m
4. Report a table: world | seen? | window | RMSE | measured Lyapunov exponent.
5. Control: also train a model WITH the held-out world included. The gap between
   held-out and included performance is the true generalization cost.
```
**Files.** New `action/eval_holdout.py` (reuse `_diagnostics` from `train_memory.py`).
**Expected gain.** No accuracy gain — this is *information*. It tells you whether to invest
in generalization or in capacity.
**Verify.** Held-out window within ~2× of the included-world window = strong generalization.
**Effort.** Half a day.

---

## 0.2 Yaw augmentation — enforce the symmetry we currently only hope for

**Why.** Gravity singles out −z, so the physics is **exactly invariant to rotation about the
vertical axis**. Our model doesn't know that: it sees absolute quaternions and world-frame
velocities, and only handles all headings because the data happens to cover them. Free
accuracy and robustness are being left on the table.

**Algorithm.** Apply a random yaw `θ ~ U(0, 2π)` to every training window, consistently:
```
Rz(θ)  = [[cosθ, -sinθ, 0], [sinθ, cosθ, 0], [0, 0, 1]]
q_yaw  = (cos(θ/2), 0, 0, sin(θ/2))          # quaternion for the same rotation

for each state s = [pos(3), quat(4), linvel(3), angvel(3)]:
    pos'    = Rz @ pos
    quat'   = quat_multiply(q_yaw, quat)     # rotate the orientation
    linvel' = Rz @ linvel
    angvel' = Rz @ angvel                    # angvel is a pseudovector; Rz is a proper
                                             # rotation (det=+1) so this is correct
```
Apply the **same θ to the whole window** (history + future), sampled fresh per sample.
Do it on GPU inside the training step — it is a few matmuls, effectively free.

**Sanity check before trusting it:** a yaw-rotated trajectory must still satisfy the physics.
Take a real trajectory, yaw-rotate it, and confirm the implied accelerations are unchanged in
magnitude. Also confirm the `quat_multiply` argument order — a wrong order silently corrupts
orientation while still "looking" plausible.

**Files.** `action/train_memory.py` (inside `run()`), helper in `action/entities.py`.
**Expected gain.** Better direction generalization; effectively free extra data. Most visible
on worlds with strong directional motion (`ball`, `object`).
**Verify.** Evaluate on trajectories rotated by yaws not seen in training; error should be flat
vs θ. Before augmentation it will show ripple.
**Effort.** 1–2 hours.

---

## 0.3 Lyapunov-normalised horizon — the physically meaningful metric

**Why.** "0.79 m of window" is not comparable across worlds. A double pendulum and a Kepler
orbit have utterly different intrinsic predictability. The field-standard unit is the
**Lyapunov time** — how long before a small error grows e-fold. Reporting horizon in Lyapunov
times tells you whether the model is near the *physical* limit or just weak.

**Algorithm (Benettin).**
```
# 1. Largest Lyapunov exponent per world
lam_sum = 0; d0 = 1e-8
x  = initial_state
x' = x + d0 * random_unit_vector
for k in 1..N:
    x  = step(x);  x' = step(x')
    d  = ||x' - x||
    lam_sum += log(d / d0)
    x' = x + (d0/d) * (x' - x)        # renormalise the separation
lambda_max = lam_sum / (N * dt)
T_lyap     = 1 / lambda_max

# 2. Model horizon in the same units
H = first horizon where model error exceeds a threshold
horizon_in_lyapunov_times = H / T_lyap

# 3. Theoretical ceiling for a state estimate with error eps:
H_max = (1/lambda_max) * log(tolerance / eps)
```
**Files.** New `action/lyapunov.py`; extend the diagnostics table.
**Expected gain.** Turns every future result into a claim about *physics*, not just cm.
**Verify.** pendulum1/nbody2 should give λ ≈ 0 (T_lyap → ∞); pendulum3/4 clearly positive.
**Effort.** Half a day. **This is the metric your final write-up should lead with.**

---

# TIER 1 — high value, moderate effort

## 1.1 Bring back the cone — probabilistic prediction, done properly

**Why.** For pendulum3/4 and nbody4 we *measured* positive Lyapunov exponents. Past the
horizon a single trajectory is **provably** wrong; only a distribution is honest. The v3
memory model outputs one line. This is the single biggest correctness gap.

**Algorithm — three options, increasing power.**

**(a) Gaussian head + deep ensemble** (easiest, already proven here in Phase 3):
```
- Two decoder heads: mean(13) and log_std(13), clamp log_std to [-7, 3].
- Train with Gaussian NLL instead of MSE:
      nll = sum_dims( 0.5*((y-mu)^2)*exp(-2*logstd) + logstd )
- Train K=5 models with different seeds + bootstrap resamples (epistemic spread).
- Sample futures: pick a random member per sample; at each step draw
      delta ~ N(mu, (temperature*std)^2)
  Keep the physical guard: clamp to world bounds, freeze a sample once it lands.
```

**(b) Quantile / distributional head** — predict quantiles of future position with the pinball
loss. No Gaussian assumption; captures skew (a ball that either clears an edge or doesn't).

**(c) Diffusion / flow-matching over whole trajectories** (most powerful, multimodal):
```
- Represent the future as a fixed-length tensor Y (H x 13).
- Train a conditional denoiser eps_theta(Y_t, t, z) with the DDPM / flow-matching
  objective, conditioned on the memory vector z.
- Sample N futures by running the reverse process N times -> a genuine cone that can
  represent "it bounces LEFT or RIGHT", which a Gaussian fundamentally cannot.
```
**Files.** `action/models.py` (heads), `action/train_memory.py` (loss), `action/cone.py`
(reuse the vectorised sampler + physical guard).
**Expected gain.** Correct behaviour on chaotic worlds; honest long-horizon output.
**Verify.** **Calibration, not accuracy**: reliability diagram, and
`coverage(h) = fraction of episodes where truth lies inside the p% band at horizon h`.
Well-calibrated ⇒ coverage ≈ p at every horizon. Also report **CRPS**.
**Effort.** (a) 1–2 days · (c) 1–2 weeks.

---

## 1.2 Grey-box residual physics — stop making the net relearn gravity

**Why.** We *know* gravity, rigid-body kinematics and quaternion integration exactly. The
network currently spends capacity rediscovering them, and its rollouts drift off the physical
manifold. Learning only the **unknown force/torque** is far more data-efficient and keeps long
rollouts physical. Highest-leverage change that is not research-risky.

**Algorithm.**
```
Known (hard-coded, exact):
    a_known = g                                    # gravity
    dq/dt   = 0.5 * quat_mult(q, [0, omega])       # quaternion kinematics
    integrate with semi-implicit Euler or RK4

Learned (small net, conditioned on the memory z):
    (F_res, tau_res) = f_theta(state_features, z)  # unknown aero / contact residual

Step:
    a     = a_known + F_res / m
    alpha = I^{-1} ( tau_res - omega x (I omega) ) # exact Euler rigid-body equation
    v    += a*dt      ; p += v*dt
    omega+= alpha*dt  ; q = normalize(q + dq/dt * dt)
```
Feed `m`, `I`, size from `entity_attrs()` (already extracted), or let the memory infer them —
**run both and compare**; "inferred" is the more general claim.

**Files.** New `action/models_greybox.py`; reuse the encoder from `models.py`.
**Expected gain.** Large. Expect much better long-horizon behaviour and far better
extrapolation to unseen objects, because the invariant part of the physics is exact.
**Verify.** Held-out-world window (§0.1) should improve most. Also check energy plausibility.
**Effort.** 3–5 days.

---

## 1.3 Multiple shooting + horizon curriculum

**Why.** We train one long rollout per window; gradients through 120 sequential steps are
ill-conditioned, and the model only ever starts from *real* states. Multiple shooting is the
standard system-identification fix and usually buys a lot of long-horizon accuracy.

**Algorithm.**
```
Split the future into S segments of length L (H = S*L).
For each segment s:
    start from the TRUE state at the segment start (not the model's own prediction)
    roll the model L steps -> y_hat_s
    L_fit  += MSE(y_hat_s, truth_s)
Continuity: penalise mismatch between the end of segment s and the start of s+1
    L_cont += || y_hat_s[-1] - truth_{s+1}[0] ||^2
Total = L_fit + beta * L_cont

Curriculum: start with L small (e.g. 10) and anneal L -> H over training, so early
training is well-conditioned and late training matches the deployment rollout.
```
**Files.** `action/train_memory.py`.
**Expected gain.** Moderate-to-large at long horizons; faster, more stable convergence.
**Verify.** The error-vs-horizon curve should flatten at long h without hurting short h.
**Effort.** 1–2 days.

---

## 1.4 Contact/event-aware prediction — handle bounces properly

**Why.** A bounce is a **discontinuity**. A smooth GRU rollout blurs it, and the `ball` /
`object` worlds are mostly bounce-and-roll. Treating contacts as ordinary steps is a
systematic error source.

**Algorithm (hybrid dynamics with event heads).**
```
Add two heads to the decoder:
    p_contact(h) : probability of a contact within the next h steps      (BCE loss)
    impulse(h)   : the velocity jump (delta_v, delta_omega) IF it occurs

Rollout:
    if p_contact > tau:
        predict time-to-contact t_c
        integrate smoothly to t_c
        apply the predicted impulse (a discontinuous jump)
        continue the smooth rollout
    else:
        ordinary smooth step

Labels are free: log data.ncon and the velocity jump each step during data
generation and save them alongside the trajectory.
```
Also consider a **mode-switching mixture of experts** (flight / impact / rolling) with a
learned gate — each regime genuinely has different dynamics.

**Files.** `action/generate_data.py` (log contacts), `models.py`, `train_memory.py`.
**Expected gain.** Large for `ball` and `object`; little for pendulum / n-body.
**Verify.** Break error down by phase: pre-first-bounce / at bounce / post-bounce. The
at-bounce error should drop sharply.
**Effort.** 3–5 days.

---

# TIER 2 — architecture upgrades

## 2.1 SO(2)-equivariant network — build the symmetry in, not learn it

**Why.** §0.2 augments; this makes yaw invariance **exact by construction**, which is strictly
stronger and needs no extra data. Gravity breaks full SO(3), so the correct group is rotation
about the vertical axis (plus translation, which we already have).

**Algorithm.**
```
Canonical-frame approach (simplest, no exotic layers):
  1. From the current state build a yaw frame:
         e1 = normalize(horizontal component of velocity)   # fallback: world x-axis
         e2 = z_hat x e1 ;  e3 = z_hat
         R_c = [e1 e2 e3]
  2. Express EVERY vector in that frame: v_local = R_c^T v, etc.
     Express orientation as R_c^T * R_body.
  3. Predict the delta in the local frame, map back: delta_world = R_c @ delta_local.
  -> the network never sees absolute yaw, so it CANNOT depend on it.

Equivariant-layer approach (stronger, more work):
  vector-neuron / EGNN-style layers where scalar features are invariants
  (norms, dot products) and vector features rotate with the frame.
```
**Caution.** The canonical frame is undefined when horizontal velocity ≈ 0. Blend smoothly to a
fixed frame as |v_horiz| → 0, or the frame flips discontinuously and injects noise. Test it.

**Files.** `action/entities.py` (frame construction), `models.py`.
**Expected gain.** Better sample efficiency; exact directional generalization.
**Verify.** Error must be *identical to float precision* for a trajectory and its yaw-rotated
copy. A hard pass/fail test — it either holds or the implementation is wrong.
**Effort.** 3–5 days.

---

## 2.2 Longer context without quadratic cost — state-space models

**Why.** We measured that identifying the environment needs a *long* look-back (12 frames is
~6% of a sway cycle and hopeless). But encoder attention costs O(L²·batch) — one of our two OOM
sources. State-space models (S4/S5/Mamba) are O(L) and handle very long sequences.

**Algorithm.**
```
Replace the Transformer encoder with a bidirectional SSM stack:
    h = SSM_block(x)  xN   # linear recurrence, learned diagonal transition + gating
Pool over time as now (attention pooling) to get z.
Keep everything else identical so the comparison is clean.
```
**Expected gain.** Enables `hist_cap` of 500–1000 frames at the same memory, directly attacking
the "needs to watch longer" limit. Also faster.
**Verify.** Sweep `hist_cap ∈ {80, 160, 320, 640}` and plot window vs history length. If it
keeps improving, context length was a real bottleneck.
**Effort.** 2–4 days (use an existing Mamba implementation).

---

## 2.3 Koopman latent linearization — long rollouts without compounding

**Why.** Autoregressive rollout compounds error (ours once blew up to 1e18). Koopman theory
says a nonlinear system becomes **linear** in a lifted observable space; an H-step prediction is
then a single matrix power, with no step-by-step compounding.

**Algorithm.**
```
1. Encoder: g = phi(state, z)                  # lift to latent, dim d
2. Linear latent dynamics: g_{k+1} = K g_k     # K a learned d x d matrix
3. Decoder: state_hat = psi(g)
Losses:
   reconstruction : || psi(phi(x)) - x ||^2
   one-step latent: || phi(x_{k+1}) - K phi(x_k) ||^2
   multi-step     : || psi(K^h phi(x_0)) - x_h ||^2   for h = 1..H   <- the key term
Optionally force stability (|eigenvalues| <= 1) via a parameterisation such as
   K = exp(A - A^T + diag(-softplus(s)))
so rollouts cannot blow up by construction.
```
**Honest caveat.** Koopman shines for near-linear/periodic systems; for strongly chaotic ones a
finite-dimensional K cannot hold up for long. Expect big wins on leaf/ball/pendulum1/nbody2,
modest on pendulum3/4.
**Effort.** ~1 week.

---

## 2.4 Reservoir computing — a cheap, strong chaotic baseline

**Why.** Pathak et al. (PRL 2018) showed echo-state networks predict chaotic systems for
several Lyapunov times, often beating deep nets, and they train in **seconds** (ridge
regression, no backprop). It is the right yardstick for our chaotic worlds — if a reservoir
beats the transformer on pendulum4, that is important information.

**Algorithm.**
```
1. Fixed random reservoir:  r_{k+1} = tanh(W r_k + W_in u_k)
     scale W so spectral radius rho(W) ~ 0.9-1.2 (edge of chaos), sparse (1-5% density)
2. Drive with the observed trajectory for a washout period (discard those states).
3. Train ONLY the readout by ridge regression:
     W_out = Y R^T (R R^T + beta I)^{-1}
4. Predict autonomously by feeding the readout output back as the next input.
```
Tune `rho`, leak rate, reservoir size (1000–5000), ridge `beta`.
**Effort.** ~1 day. **Best effort-to-insight ratio in this document.**

---

# TIER 3 — the original dream and genuine research

## 3.1 Dissipative Lagrangian / Hamiltonian "Action" net (Phase 4)

**Why.** The idea the project was named for — the principle of least action. LNN/HNN assume
energy conservation; **our systems dissipate** (drag, friction, contacts). A *dissipative*
least-action network is genuinely under-explored, and is the most novel thing available here.

**Algorithm.**
```
Learn a Lagrangian L_theta(q, qdot) and a Rayleigh dissipation function D_theta(q, qdot) >= 0.
Euler-Lagrange with dissipation:
    d/dt (dL/dqdot) - dL/dq + dD/dqdot = 0
Solve for the acceleration (this is what makes it integrable as a network):
    qddot = (d^2L/dqdot^2)^{-1} [ dL/dq - (d^2L/dqdot dq) qdot - dD/dqdot ]
Take derivatives with autograd; invert the (small) mass matrix with torch.linalg.solve.
Integrate qddot with RK4. Train on trajectory MSE exactly as now.
Enforce D >= 0 by construction, e.g. D = 0.5 * qdot^T (A^T A) qdot.
```
**Why it should help.** The learned dynamics are *structurally* a physical system, so energy
behaves sensibly and long rollouts stay on the manifold instead of drifting into nonsense.
**Verify.** Compare the learned dissipation against the true energy curve: the model should
lose energy only where the real system does (contacts, drag) and **never gain** it.
**Effort.** 1–2 weeks. **Highest novelty.**

---

## 3.2 Vision → state (the original webcam dream, done honestly)

**Why.** The project began wanting a webcam "mirage of the future." We deferred it for three
good reasons (monocular depth, chaos, sim-to-real). Two are now tractable: the worlds are
calibrated and we can render unlimited labelled data.

**Algorithm.**
```
Stage 1 (sim only, supervised):
  render each episode from a known camera (projection already exists in action/camera.py)
  train a CNN/ViT:  image(s) -> 13-dim entity state
  use 2+ frames so velocity is observable; multi-view or a known ground plane fixes scale
Stage 2:
  freeze the estimator and feed its output into the existing memory model
Stage 3 (real world):
  domain randomization at render time (lighting, texture, camera pose, motion blur)
  calibrate the real camera; use a known-size reference object to fix scale
```
**Honest caveat.** Monocular depth/scale is genuinely ambiguous. Fix it with a known object
size, a known ground plane, or stereo. Do not pretend one uncalibrated webcam solves it.
**Effort.** 2–4 weeks.

---

## 3.3 A genuinely fluttering leaf — unsteady aerodynamics

**Why.** We failed at this honestly (CLAUDE.md §14e). Quasi-steady models *cannot* flutter:
real flutter needs the wake's feedback delay. Note the trade-off — success makes the leaf
**chaotic**, so its long-horizon predictability gets *worse*. Do it for realism, not accuracy.

**Algorithm (Wagner/Theodorsen-style lagged circulation).**
```
Add a circulation state Gamma with its own dynamics:
    Gamma_steady = 0.5 * rho * c * |v| * C_L(alpha)
    dGamma/dt    = (Gamma_steady - Gamma) / tau      # tau ~ c/|v|, the wake timescale
Use the LAGGED Gamma in the lift:   F_lift = rho * Gamma * (v x span_axis)
Add added-mass terms:               F_am = -m_added * dv/dt
                                    tau_am ~ (m_perp - m_par) * sin(2*alpha)
The phase lag between attitude and force is what destabilises the glide into oscillation.
```
Then re-measure the Lyapunov exponent — success looks like λ > 0 **and** flutter ratio > 1.3.
**Effort.** 1–2 weeks, uncertain payoff.

---

## 3.4 Multi-body scenes (revive `models_general.py`)

Already built and verified permutation-equivariant on 1–12 bodies; parked when the scope
narrowed to one object at a time. Algorithm: factorised space-time attention (attend over time
per entity, then over entities per frame), shared weights, per-entity decoder with an
interaction layer each step. Collisions, joints and gravity are all learned as attention
between entities.

---

# ENGINEERING & SPEED

| Item | Algorithm / action | Gain |
|---|---|---|
| **Mixed precision** | wrap the step in `torch.autocast('cuda', torch.float16)` + `GradScaler` | ~1.3–1.5× |
| **Multi-GPU** | move the per-batch loss into `forward()` (return per-sample loss), wrap in `DistributedDataParallel` (DDP > DataParallel) | ~1.8× on 2×T4 |
| **`torch.compile` / CUDA graphs** | the sequential decoder is launch-latency bound; CUDA graphs remove per-step launch overhead | 1.2–2× |
| **Cache windows** | precompute + memory-map the (history, future) index instead of rebuilding each epoch | faster startup |
| **Experiment tracking** | log window, RMSE, coverage and per-world breakdown to CSV/W&B | stops guesswork |

**Memory rules of thumb we measured:** encoder attention ∝ `L² · batch`; decoder backprop ∝
`fut_cap · batch · dec_hidden`. These are **independent** OOM sources — check both before
blaming one. `--ckpt-chunk` fixes only the decoder.

---

# EXPERIMENT PROTOCOL (apply to every change above)

1. **Fix the split.** Same held-out episodes every time; never tune on the test set.
2. **Always report five numbers:** whole-fall RMSE · window @50% watched · sharpen curve ·
   held-out-world window · **horizon in Lyapunov times**.
3. **One change at a time**, ≥3 seeds. Several of our "improvements" were within noise until
   we used a denser validation set.
4. **Ablate.** If you add three things and it improves, you don't know which one worked.
5. **Beware the metric.** We wasted real effort on a "window from 12 frames watched" that was
   *information-limited* and could never improve. Ask of every metric: *is this measuring the
   model, or a limit of the setup?*

---

# SUGGESTED ORDER

```
Week 1    0.1 held-out eval  +  0.2 yaw augmentation  +  0.3 Lyapunov metric
          -> you now know what to optimise, in honest units

Week 2    1.1(a) Gaussian ensemble cone     -> correctness on chaotic worlds
          2.4  reservoir baseline (1 day)   -> cheap, strong yardstick

Week 3-4  1.2 grey-box residual physics     -> biggest expected accuracy win
          1.3 multiple shooting + curriculum

Week 5+   pick by evidence:
          contacts dominate error?  -> 1.4 event-aware prediction
          context length helps?     -> 2.2 state-space encoder
          want novelty?             -> 3.1 dissipative Lagrangian net
          want the original dream?  -> 3.2 vision -> state
```

**If you only do three things:** §0.1 (held-out proof), §1.2 (grey-box physics), §1.1
(probabilistic cone). Those give, respectively, the **evidence**, the **accuracy**, and the
**correctness** the project currently lacks.
