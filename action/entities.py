"""
entities.py — the universal, world-agnostic description of a physical scene.

This is the key to a *general* predictor. Instead of a flat state vector whose
meaning and length depend on the world (13 for a free body, 2n for a pendulum,
6n for n-body), every scene is described as a **set of entities**, where every
entity — a leaf, a ball, a pendulum link, a star — carries the exact same 13
numbers in world coordinates:

    [0:3]   position           x, y, z
    [3:7]   orientation        qw, qx, qy, qz   (unit quaternion)
    [7:10]  linear velocity    vx, vy, vz
    [10:13] angular velocity   wx, wy, wz

plus a small vector of *static* properties (mass, inertia, size) that never change
during an episode but differ between bodies — the physical identity of the object.

Why this matters:

* **Any number of bodies.** A scene is a variable-length set, so a model with shared
  per-entity weights handles 1 body or 40 without changing shape.
* **Any new body works.** Nothing is hardcoded per world; we read the state straight
  out of MuJoCo, so a world you write tomorrow is described correctly today.
* **Every kind of motion is the same problem.** Free fall, a bounce that redirects a
  ball, a joint dragging a link, mutual gravity — all of them are just bodies
  influencing other bodies. Interactions become attention between entities rather
  than something baked into the state layout.
* **Maximal (Cartesian) coordinates.** We deliberately record each body's *world*
  pose rather than joint angles. A hinge constraint is then something the model
  observes and learns, instead of a different state layout it must be rebuilt for.

Reference frame: `relative_to` re-expresses positions relative to the scene centroid
(and optionally scales), which is what gives translation invariance — a system
orbiting 50 m away looks identical to one at the origin.
"""
from __future__ import annotations

import numpy as np
import mujoco

ENTITY_DIM = 13          # pos(3) + quat(4) + linvel(3) + angvel(3)
ATTR_DIM = 6             # mass, log-mass, inertia diag(3), bounding radius

POS = slice(0, 3)
QUAT = slice(3, 7)
LINVEL = slice(7, 10)
ANGVEL = slice(10, 13)


def n_entities(model: mujoco.MjModel) -> int:
    """Number of real bodies (body 0 is MuJoCo's worldbody, which we skip)."""
    return int(model.nbody - 1)


def entity_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """(N, 13) world-frame state of every body. Works for ANY MuJoCo world.

    Velocities come from `mj_objectVelocity`, NOT from `data.cvel`. `cvel` is a
    com-based *spatial* velocity referenced to the subtree centre of mass, so for a
    body that is spinning or far from that reference it is not the body's own linear
    velocity — using it silently produces wrong speeds.
    """
    n = n_entities(model)
    out = np.zeros((n, ENTITY_DIM), dtype=np.float32)
    res = np.zeros(6, dtype=np.float64)
    for i in range(n):
        b = i + 1                                   # skip worldbody
        out[i, POS] = data.xpos[b]
        out[i, QUAT] = data.xquat[b]
        # world-frame velocity of this body: res = [angular(3), linear(3)]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
        out[i, ANGVEL] = res[0:3]
        out[i, LINVEL] = res[3:6]
    return out


def entity_attrs(model: mujoco.MjModel) -> np.ndarray:
    """(N, 6) static physical identity per body: mass, log-mass, inertia, size.

    These are constant within an episode but differ between bodies and between
    episodes — exactly the hidden properties the memory model must exploit (a steel
    ball and a leaf differ here, and that is *why* they move differently)."""
    n = n_entities(model)
    out = np.zeros((n, ATTR_DIM), dtype=np.float32)
    for i in range(n):
        b = i + 1
        m = float(model.body_mass[b])
        out[i, 0] = m
        out[i, 1] = np.log(max(m, 1e-8))
        out[i, 2:5] = model.body_inertia[b]
        # rbound of the body's first geom is a decent scalar "size"
        gi = int(model.body_geomadr[b])
        out[i, 5] = float(model.geom_rbound[gi]) if gi >= 0 and model.body_geomnum[b] > 0 else 0.0
    return out


def relative_to(states: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Make positions relative to a reference point -> translation invariance.

    states: (..., N, 13); ref: (..., 3) broadcast over entities.
    Orientation and velocities are already frame-independent, so only positions move.
    """
    out = states.copy()
    out[..., 0:3] = out[..., 0:3] - ref[..., None, :]
    return out


def scene_centroid(state: np.ndarray) -> np.ndarray:
    """(..., 3) mean body position — a stable, permutation-invariant origin."""
    return state[..., 0:3].mean(axis=-2)


def normalize_quats(states: np.ndarray) -> np.ndarray:
    """Renormalize every entity's quaternion (rollouts drift off the unit sphere)."""
    out = states.copy()
    q = out[..., 3:7]
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    out[..., 3:7] = np.where(n > 1e-8, q / np.maximum(n, 1e-8), q)
    return out
