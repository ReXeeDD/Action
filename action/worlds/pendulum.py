"""
pendulum.py — an n-link chained pendulum. Real physics, real chaos.

n=1 is the classic (integrable, perfectly predictable). n=2 is the textbook **double
pendulum** — one of the cleanest chaotic systems there is. n=3 and n=4 are wilder
still. Nothing is imposed here: MuJoCo integrates the exact rigid-body equations
under gravity, and the chaos is genuine, not injected.

This is the scientific complement to the leaf. We *measured* the leaf world to be
non-chaotic (Lyapunov ~ 0), which is why a single line could predict most of its
fall. Here the Lyapunov exponent is genuinely positive, so there is a real, physical
prediction horizon that no model — however large — can cross. Comparing the two is
the honest heart of the project.

Hidden per-episode physics the model must infer from motion: each link's length and
mass, plus joint damping.

State = qpos(n angles) ++ qvel(n angular velocities) = 2n. No positions or
quaternions, so `pos_groups`/`quat_groups` are empty.
"""
from __future__ import annotations

import numpy as np

from action.worlds.base import MujocoWorld, VISUAL


class ChainPendulum(MujocoWorld):
    max_steps = 900
    air_density = 1.2      # a real pendulum swings in air, not vacuum

    def __init__(self, n_links: int = 2, seed: int | None = None):
        super().__init__(seed)
        self.n = int(n_links)
        self.name = f"pendulum{self.n}"

    def build(self, randomize: bool) -> str:
        if randomize:
            self.L = self.rng.uniform(0.22, 0.42, size=self.n)
            self.M = self.rng.uniform(0.15, 0.6, size=self.n)
            # keep damping tiny: friction bleeds energy and suppresses the very chaos
            # this world exists to exhibit
            damp = float(self.rng.uniform(0.0, 0.004))
            # Randomize which VERTICAL PLANE the chain swings in. With a fixed hinge
            # axis every episode moved in the x-z plane only (measured y-spread was
            # exactly 0), so a model would never see a pendulum swinging in any other
            # direction. Rotating the axis in the horizontal plane fixes that.
            th = float(self.rng.uniform(0, 2 * np.pi))
        else:
            self.L = np.full(self.n, 0.3)
            self.M = np.full(self.n, 0.3)
            damp = 0.005
            th = 0.0
        self.pos_groups, self.quat_groups = [], []      # angles only
        ax = f"{np.cos(th):.5f} {np.sin(th):.5f} 0"     # horizontal hinge axis

        # build the chain from the inside out
        body = ""
        for i in range(self.n):
            L, M = self.L[i], self.M[i]
            pos = "0 0 0" if i == 0 else f"0 0 {-self.L[i-1]:.4f}"
            body += (f'<body name="l{i}" pos="{pos}">'
                     f'<joint name="j{i}" type="hinge" axis="{ax}" pos="0 0 0" '
                     f'damping="{damp}"/>'
                     f'<geom type="capsule" fromto="0 0 0 0 0 {-L:.4f}" size="0.018" '
                     f'mass="{M:.4f}" rgba="{0.3+0.6*i/max(self.n,1):.2f} 0.6 0.85 1"/>')
        body += "</body>" * self.n

        total = float(self.L.sum())
        return f"""
<mujoco model="pendulum{self.n}">
  {self.physics_options()}
{VISUAL}
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 {-total-0.3:.3f}"
          rgba="0.28 0.32 0.3 1"/>
    <camera name="wide" pos="0 -{2.6*total+1.4:.2f} 0" xyaxes="1 0 0 0 0 1" fovy="55"/>
    <camera name="side" pos="0 -{1.9*total+1.0:.2f} 0" xyaxes="1 0 0 0 0 1" fovy="50"/>
    {body}
  </worldbody>
</mujoco>
""".strip()

    @property
    def target_index(self) -> int:
        # The LAST link — the richest, most chaotic motion in the chain. What gets
        # recorded is that link's **centre of mass** (see entities.entity_state), not
        # its free end: the end of the chain is not any body's frame, so tracking it
        # would need pendulum-specific code, and per-object special cases are exactly
        # what this project forbids. The CoM is universal and moves genuinely.
        return self.n - 1

    def init_state(self, randomize: bool) -> None:
        if randomize:
            # start well away from the bottom so it actually swings/tumbles
            self.data.qpos[:] = self.rng.uniform(-np.pi, np.pi, size=self.n)
            self.data.qvel[:] = self.rng.normal(0.0, 1.5, size=self.n)
        else:
            self.data.qpos[:] = np.linspace(2.0, 2.6, self.n)
            self.data.qvel[:] = 0.0
