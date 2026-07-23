"""
nbody.py — n point masses under real Newtonian mutual gravitation.

MuJoCo has no body-to-body gravity, so we compute it ourselves each step straight
from Newton's law,

    F_ij = G * m_i * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^{3/2}

and apply it with `xfrc_applied`. That is not a fake forcing like the leaf's sway —
it *is* the physics, just evaluated by us instead of by the engine. Uniform gravity
and all contacts are switched off, so the only thing acting is mutual attraction.
(`eps` is the standard Plummer softening; without it a close approach produces an
infinite force and the integrator explodes.)

n=2 is the Kepler two-body problem — exactly solvable and perfectly predictable.
n=3 is *the* three-body problem: famously chaotic, no closed-form solution, and the
canonical example of a system whose future is knowable only for a finite time. n=4
is worse. So this world family spans the full range from integrable to violently
chaotic, which makes it the ideal stress test for a prediction horizon.

Each body is three slide joints (a point mass), so
state = qpos(3n positions) ++ qvel(3n velocities) = 6n.
"""
from __future__ import annotations

import numpy as np

from action.worlds.base import MujocoWorld, VISUAL

# Chosen so an orbit takes ~1-2 s of sim time: with R~1.6 and total mass ~3, the
# period is 2*pi*sqrt(R^3/(G*M)), so G~15 puts several orbits inside one episode.
# (At G=1 an episode covered less than a single orbit and nothing interesting -- or
# chaotic -- had time to happen.)
G_CONST = 15.0
SOFTENING = 0.10       # Plummer softening (m) — keeps close approaches finite
ESCAPE_R = 12.0        # if a body wanders this far, the episode is over
# Bodies passing within a few softening lengths produce enormous forces that no
# practical timestep integrates accurately (energy drifted >100%). Physically they
# would have *collided* at that separation, so we end the episode there instead of
# integrating through a regime the simulation cannot represent honestly.
COLLIDE_R = 0.18


class NBodyWorld(MujocoWorld):
    max_steps = 900
    # We apply gravitation through `xfrc_applied`, which MuJoCo holds CONSTANT across
    # RK4's internal stages — so the force lags the state and energy drifts badly
    # (255% over an episode at timestep 4e-3). Stepping finer updates the force more
    # often and restores conservation; dt per recorded frame stays 0.008.
    timestep = 0.0005
    n_substeps = 16
    gravity = (0.0, 0.0, 0.0)   # deep space: no uniform gravity
    air_density = 0.0           # vacuum
    grav_const = G_CONST        # universal mutual gravitation (base class)
    softening = SOFTENING

    # Spawn geometry, exposed so it can be tuned by measurement rather than guesswork.
    # These had never been exercised against real orbital motion: mutual gravitation
    # was silently disabled by a duplicate `apply_forces` in base.py, so every
    # "calibration" of these numbers was performed on bodies drifting in straight
    # lines. spawn_r is the ring radius range, speed_frac the launch speed as a
    # fraction of the local circular speed (1.0 = circular, <1 falls inward).
    spawn_r = (2.2, 4.0)
    speed_frac = (0.90, 1.05)   # 1.0 = circular; escape is sqrt(2)
    collide_r = 0.12            # just above SOFTENING, so the force is still smooth

    def __init__(self, n_bodies: int = 3, seed: int | None = None):
        super().__init__(seed)
        self.n = int(n_bodies)
        self.name = f"nbody{self.n}"

    def build(self, randomize: bool) -> str:
        self.M = (self.rng.uniform(0.6, 1.8, size=self.n) if randomize
                  else np.full(self.n, 1.0))
        self.pos_groups = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(self.n)]
        self.quat_groups = []

        colors = ["0.95 0.75 0.25", "0.45 0.75 0.95", "0.95 0.45 0.55",
                  "0.6 0.9 0.5", "0.8 0.6 0.95", "0.9 0.9 0.9"]
        bodies = ""
        for i in range(self.n):
            rad = 0.06 * float(self.M[i]) ** (1 / 3) + 0.04
            bodies += f"""
    <body name="b{i}" pos="0 0 0">
      <joint name="x{i}" type="slide" axis="1 0 0"/>
      <joint name="y{i}" type="slide" axis="0 1 0"/>
      <joint name="z{i}" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="{rad:.4f}" mass="{self.M[i]:.4f}"
            rgba="{colors[i % len(colors)]} 1" contype="0" conaffinity="0"/>
    </body>"""

        return f"""
<mujoco model="nbody{self.n}">
  {self.physics_options()}
{VISUAL}
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <camera name="wide" pos="0 -14 0" xyaxes="1 0 0 0 0 1" fovy="55"/>
    <camera name="side" pos="0 -9 0" xyaxes="1 0 0 0 0 1" fovy="50"/>{bodies}
  </worldbody>
</mujoco>
""".strip()

    def init_state(self, randomize: bool) -> None:
        n = self.n
        if randomize:
            # place bodies on a jittered ring and give them tangential velocities,
            # which tends to produce bound, interesting orbits rather than instant escape
            R = self.rng.uniform(*self.spawn_r)
            # NOTE: do NOT sort these angles. Sorting always handed body 0 (the
            # prediction target) the smallest angle, which systematically biased its
            # direction of motion and would have taught the model a preferred heading.
            ang = self.rng.uniform(0, 2 * np.pi, size=n)
            pos = np.zeros((n, 3))
            vel = np.zeros((n, 3))
            # Circular speed for a RING of masses, not for a point mass at the centre.
            # For a regular n-gon of mass m at radius R the net inward force is
            #   F = (G m^2 / R^2) * S_n,   S_n = (1/4) sum_{k=1..n-1} 1/sin(pi k/n)
            # giving v_circ = sqrt(G m S_n / R). The old code used
            # sqrt(G * M_total / R), i.e. all the mass concentrated at the origin,
            # which overestimates v_circ by sqrt(n/S_n) — 2.83x at n=2, 2.28x at n=3.
            # Escape is only sqrt(2) x circular, so a "0.75 of circular" launch was
            # really 1.7x circular and the system simply flew apart. Nobody noticed
            # because mutual gravitation was disabled entirely (duplicate
            # `apply_forces` in base.py), so nothing ever orbited to begin with.
            S_n = 0.25 * sum(1.0 / np.sin(np.pi * k / n) for k in range(1, n))
            m_avg = self.M.sum() / n
            for i in range(n):
                r = R * self.rng.uniform(0.7, 1.3)
                pos[i] = [r * np.cos(ang[i]), r * np.sin(ang[i]),
                          self.rng.normal(0, 0.25 * R / 2.0)]
                v_circ = np.sqrt(G_CONST * m_avg * S_n / max(r, 0.3))
                speed = v_circ * self.rng.uniform(*self.speed_frac)
                vel[i] = [-speed * np.sin(ang[i]), speed * np.cos(ang[i]),
                          self.rng.normal(0, 0.05)]
        else:
            ang = np.arange(n) * 2 * np.pi / n
            pos = np.stack([1.6 * np.cos(ang), 1.6 * np.sin(ang), np.zeros(n)], 1)
            vel = np.stack([-0.55 * np.sin(ang), 0.55 * np.cos(ang), np.zeros(n)], 1)

        # do not spawn bodies already inside the collision radius (nbody4 was
        # ending after 8 steps because two masses started essentially on top of
        # each other and immediately tripped the collision test)
        for _ in range(200):
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(pos[i] - pos[j]) < 4 * self.collide_r:
                        ok = False
            if ok:
                break
            pos *= 1.15
        # remove net drift so the system stays centred in frame
        vel -= (self.M[:, None] * vel).sum(0) / self.M.sum()
        pos -= (self.M[:, None] * pos).sum(0) / self.M.sum()
        self.data.qpos[:] = pos.reshape(-1)
        self.data.qvel[:] = vel.reshape(-1)

    @property
    def done(self) -> bool:
        p = self.data.qpos.reshape(self.n, 3)
        if np.linalg.norm(p, axis=1).max() > ESCAPE_R:      # a body escaped
            return True
        for i in range(self.n):                             # a pair collided
            for j in range(i + 1, self.n):
                if np.linalg.norm(p[i] - p[j]) < self.collide_r:
                    return True
        return False
