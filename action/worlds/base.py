"""
base.py — the common interface every physical world implements.

Everything downstream (data generation, the memory model, the renderer) talks to a
world only through this interface, so adding a new physical system means writing one
small class, not touching the pipeline.

State convention (universal): `state = qpos ++ qvel`, so its dimension is whatever
that world's degrees of freedom require — 13 for a free rigid body, 2n for an n-link
pendulum, 6n for n gravitating point masses.

Two index maps let the pipeline stay physics-agnostic while still being
physics-correct:

* `pos_groups` — index triples in the state vector that are 3D **positions**. The
  feature builder makes these relative to a reference point, which is what gives us
  translation invariance (a fall/orbit at the origin looks identical to one 5 m away).
* `quat_groups` — index quadruples that are unit **quaternions**, which must be
  renormalized after every predicted step or the rotation drifts off the manifold.

Worlds with neither (a pendulum's state is angles) simply leave both empty.
"""
from __future__ import annotations

import numpy as np
import mujoco


class MujocoWorld:
    """Base class: builds a randomized MJCF world and rolls it forward."""

    name = "base"
    timestep = 0.004
    n_substeps = 2          # one recorded frame = timestep * n_substeps
    max_steps = 900

    # ------------------------------------------------------------------
    # UNIVERSAL PHYSICS. The same laws act on every body in every world:
    #   * gravity                (MuJoCo)
    #   * contacts + friction    (MuJoCo, any geom)
    #   * fluid drag / lift      (MuJoCo's fluid model — applies to ANY body from
    #                             its own geometry once `density` is non-zero)
    #   * mutual gravitation     (Newton's law, applied uniformly to all bodies)
    #
    # Worlds differ only in the *environment parameters* below (how dense the air
    # is, how strong gravity is), never in the laws themselves. That is what lets
    # you drop in a brand-new object and have it behave correctly with no new code:
    # the engine already knows how to push on any shape.
    # ------------------------------------------------------------------
    gravity = (0.0, 0.0, -9.81)
    air_density = 1.2          # kg/m^3  (0 = vacuum)
    air_viscosity = 1.8e-5     # Pa*s
    grav_const = 0.0           # mutual gravitation strength (0 = off)

    def physics_options(self) -> str:
        gx, gy, gz = self.gravity
        return (f'<option timestep="{self.timestep}" integrator="RK4" '
                f'gravity="{gx} {gy} {gz}" density="{self.air_density}" '
                f'viscosity="{self.air_viscosity}"/>')

    def apply_forces(self) -> None:
        """Universal mutual gravitation — Newton's law, applied to every body
        identically. Off by default (grav_const = 0); the N-body worlds turn it on.
        No other bespoke force exists anywhere: everything else is the engine."""
        if not self.grav_const:
            return
        n = self.n_entities
        p = np.array([self.data.xpos[i + 1] for i in range(n)])
        m = np.array([self.model.body_mass[i + 1] for i in range(n)])
        soft2 = self.softening ** 2
        for i in range(n):
            d = p - p[i]
            r2 = (d ** 2).sum(1) + soft2
            r2[i] = np.inf
            f = (self.grav_const * m[i] * m[:, None] * d / r2[:, None] ** 1.5).sum(0)
            self.data.xfrc_applied[i + 1, 0:3] = f

    softening = 0.10

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.pos_groups: list[list[int]] = []
        self.quat_groups: list[list[int]] = []

    # ---- subclass hooks ---------------------------------------------------
    def build(self, randomize: bool) -> str:
        """Return the MJCF for one episode (randomizing the hidden physics)."""
        raise NotImplementedError

    def init_state(self, randomize: bool) -> None:
        """Set qpos/qvel for the start of the episode."""
        raise NotImplementedError

    def apply_forces(self) -> None:
        """Optional per-step forces (e.g. mutual gravitation in the N-body world)."""

    @property
    def done(self) -> bool:
        """Episode-ending condition (landed, escaped, ...)."""
        return False

    # ---- shared machinery -------------------------------------------------
    def reset(self, randomize: bool = True) -> np.ndarray:
        xml = self.build(randomize)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.init_state(randomize)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def state_dim(self) -> int:
        return int(self.model.nq + self.model.nv)

    @property
    def dt(self) -> float:
        return self.timestep * self.n_substeps

    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    # ---- universal entity view (see action/entities.py) --------------------
    # Every world gets this for free: an (N, 13) world-frame description of its
    # bodies, which is what the GENERAL predictor consumes. Adding a new world or a
    # new body requires no changes here.
    @property
    def n_entities(self) -> int:
        from action.entities import n_entities
        return n_entities(self.model)

    def entity_state(self) -> np.ndarray:
        from action.entities import entity_state
        return entity_state(self.model, self.data)

    def entity_attrs(self) -> np.ndarray:
        from action.entities import entity_attrs
        return entity_attrs(self.model)

    def entity_rollout(self, max_steps: int | None = None) -> np.ndarray:
        """(T, N, 13) — the universal trajectory format."""
        traj = [self.entity_state()]
        for _ in range(max_steps or self.max_steps):
            self.step()
            traj.append(self.entity_state())
            if self.done:
                break
        return np.asarray(traj, dtype=np.float32)

    # ---- the ONE object we predict ----------------------------------------
    # We predict a single target body at a time. Because its state is the same
    # universal 13 numbers in every world, one model covers a leaf, a ball, a
    # pendulum bob or an orbiting mass — and a brand-new object works with no
    # retraining, since the representation never changes shape.
    @property
    def target_index(self) -> int:
        return 0

    # rest detection: an episode runs until the target actually stops moving
    rest_speed = 0.08          # m/s
    # NOTE: a rolling sphere spins at v/r, so a small ball creeping along at the
    # speed threshold still spins several rad/s. Gating rest on a tight spin limit
    # would mean a rolling body could never be declared at rest, so the spin bound
    # is deliberately loose and linear speed is the real criterion.
    rest_spin = 3.0            # rad/s
    rest_frames = 25           # must stay slow this many frames (not just graze zero)

    def target_at_rest(self, recent: list[np.ndarray]) -> bool:
        if len(recent) < self.rest_frames:
            return False
        w = np.asarray(recent[-self.rest_frames:])
        return bool(np.all(np.linalg.norm(w[:, 7:10], axis=1) < self.rest_speed) and
                    np.all(np.linalg.norm(w[:, 10:13], axis=1) < self.rest_spin))

    def target_rollout(self, max_steps: int | None = None,
                       until_rest: bool = True) -> np.ndarray:
        """(T, 13) trajectory of the single target object — the universal format
        the predictor consumes. Runs until the body comes to rest (or the world's
        own `done`, or max_steps)."""
        i = self.target_index
        traj = [self.entity_state()[i]]
        for _ in range(max_steps or self.max_steps):
            self.step()
            traj.append(self.entity_state()[i])
            if self.done:
                break
            if until_rest and self.target_at_rest(traj):
                break
        return np.asarray(traj, dtype=np.float32)

    def step(self, n_substeps: int | None = None) -> np.ndarray:
        for _ in range(n_substeps or self.n_substeps):
            self.apply_forces()
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def rollout(self, max_steps: int | None = None) -> np.ndarray:
        traj = [self.state()]
        for _ in range(max_steps or self.max_steps):
            traj.append(self.step())
            if self.done:
                break
        return np.asarray(traj, dtype=np.float32)

    def meta(self) -> dict:
        """Everything the training pipeline needs to stay physics-agnostic."""
        return {"world": self.name, "state_dim": self.state_dim,
                "nq": int(self.model.nq), "nv": int(self.model.nv),
                "pos_groups": self.pos_groups, "quat_groups": self.quat_groups,
                "dt": self.dt}


VISUAL = """
  <visual>
    <headlight ambient="0.45 0.45 0.45"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
"""
