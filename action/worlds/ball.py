"""
ball.py — a solid ball thrown in an arbitrary direction, with real contacts.

Nothing is faked here: MuJoCo integrates gravity, air drag (via the fluid model),
friction and restitution. The ball is launched from a random point with a random
velocity in *any* direction — up, sideways, down — so it arcs, lands, bounces, and
rolls to rest. Unlike the leaf, the flight phase is smooth and near-ballistic
(highly predictable), while each **bounce** is a sharp, contact-driven event that
amplifies small differences — so the interesting difficulty is concentrated at the
impacts rather than spread through the trajectory.

Hidden per-episode physics the model must infer from motion alone: mass, radius,
restitution, friction, and the air density it is flying through.
"""
from __future__ import annotations

import numpy as np
import mujoco

from action.worlds.base import MujocoWorld, VISUAL


class BallWorld(MujocoWorld):
    name = "ball"
    max_steps = 1600   # give it time to bounce, roll, and actually stop

    def build(self, randomize: bool) -> str:
        r = float(self.rng.uniform(0.03, 0.09)) if randomize else 0.05
        m = float(self.rng.uniform(0.05, 0.6)) if randomize else 0.2
        # solref: (stiffness time-constant, damping ratio). A low damping ratio is a
        # springy, bouncy contact; near 1 is dead. This is the "how bouncy" knob.
        damp = float(self.rng.uniform(0.15, 0.75)) if randomize else 0.35
        fric = float(self.rng.uniform(0.2, 0.9)) if randomize else 0.5
        # rolling resistance matters: with it near zero a sphere rolls forever and
        # never "comes to rest", so an episode could never end naturally.
        roll = float(self.rng.uniform(0.10, 0.35)) if randomize else 0.18
        dens = float(self.rng.uniform(0.8, 1.5)) if randomize else 1.2
        self._r = r
        self._fric = f"{fric} 0.005 {roll}"
        self.pos_groups = [[0, 1, 2]]
        self.quat_groups = [[3, 4, 5, 6]]
        return f"""
<mujoco model="ball_world">
  {self.physics_options()}
{VISUAL}
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="30 30 0.1" rgba="0.3 0.35 0.3 1"
          friction="{self._fric}" solref="0.008 {damp}"/>
    <camera name="wide" pos="0 -12 3.0" xyaxes="1 0 0 0 0 1" fovy="60"/>
    <camera name="side" pos="0 -8 2.0" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <body name="ball" pos="0 0 1.5">
      <freejoint name="root"/>
      <camera name="chase" mode="track" pos="0 -5 1.0" xyaxes="1 0 0 0 0 1" fovy="55"/>
      <geom name="ball" type="sphere" size="{r}" mass="{m}" rgba="0.75 0.76 0.8 1"
            friction="{self._fric}" solref="0.008 {damp}"
            fluidshape="ellipsoid"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    def reset(self, randomize: bool = True):
        s = super().reset(randomize)
        self._bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        return s

    def init_state(self, randomize: bool) -> None:
        if randomize:
            self.data.qpos[0:3] = [self.rng.uniform(-2.0, 2.0),
                                   self.rng.uniform(-2.0, 2.0),
                                   self.rng.uniform(0.6, 3.0)]
            q = self.rng.normal(size=4); q /= np.linalg.norm(q)
            self.data.qpos[3:7] = q
            # launched in ANY direction: full 3D velocity, often upward
            self.data.qvel[0:3] = [self.rng.normal(0, 1.6), self.rng.normal(0, 1.6),
                                   self.rng.uniform(-1.0, 3.0)]
            self.data.qvel[3:6] = self.rng.normal(0, 4.0, size=3)
        else:
            self.data.qpos[0:3] = [0, 0, 2.0]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qvel[0:3] = [1.5, 0.5, 2.0]

    @property
    def done(self) -> bool:
        # stop once it has essentially come to rest on the ground
        v = float(np.linalg.norm(self.data.qvel[0:3]))
        w = float(np.linalg.norm(self.data.qvel[3:6]))
        return bool(self.data.qpos[2] < self._r * 1.4 and v < 0.12 and w < 0.8)
