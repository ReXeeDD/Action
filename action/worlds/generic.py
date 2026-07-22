"""
generic.py — drop in ANY object and it just works.

This world exists to prove the central claim: there is no per-object physics
anywhere. A shape, a size and a mass are all you supply; gravity, aerodynamic
drag/lift, contacts, friction and restitution are then applied by the engine's
universal laws, exactly as they are for the leaf, the ball or a pendulum link.

Shapes are picked at random per episode — sphere, box, thin plate (a sheet of
paper/card), capsule, cylinder — with random dimensions, mass, air density and
surface properties, and the object is launched in an arbitrary direction. A flat
plate will glide and slice differently from a sphere purely because the *same*
fluid model sees a different geometry, not because anyone wrote special code for it.

Adding a genuinely new object later means adding a shape string here. Nothing in
the physics, the data pipeline, or the model has to change.
"""
from __future__ import annotations

import numpy as np
import mujoco

from action.worlds.base import MujocoWorld, VISUAL

SHAPES = ("sphere", "box", "plate", "capsule", "cylinder")


class GenericObject(MujocoWorld):
    name = "object"
    max_steps = 1600

    def __init__(self, shape: str | None = None, seed: int | None = None):
        super().__init__(seed)
        self.shape = shape                     # None = random each episode

    def build(self, randomize: bool) -> str:
        rng = self.rng
        shape = self.shape or (rng.choice(SHAPES) if randomize else "box")
        self._shape = str(shape)
        s = rng.uniform(0.03, 0.10)
        if shape == "sphere":
            gtype, size = "sphere", f"{s}"
        elif shape == "box":
            gtype, size = "box", f"{s} {s*rng.uniform(0.5,1.5):.4f} {s*rng.uniform(0.5,1.5):.4f}"
        elif shape == "plate":                 # a sheet: wide, thin -> glides
            gtype, size = "box", f"{s*rng.uniform(1.5,3.0):.4f} {s*rng.uniform(1.5,3.0):.4f} {s*0.05:.4f}"
        elif shape == "capsule":
            gtype, size = "capsule", f"{s*0.5:.4f} {s*rng.uniform(1.0,2.5):.4f}"
        else:
            gtype, size = "cylinder", f"{s*0.7:.4f} {s*rng.uniform(0.5,2.0):.4f}"

        m = float(rng.uniform(0.01, 0.8))
        fric = float(rng.uniform(0.2, 0.9))
        roll = float(rng.uniform(0.05, 0.3))
        damp = float(rng.uniform(0.15, 0.8))
        self.air_density = float(rng.uniform(0.8, 1.6))
        self._fric = f"{fric} 0.005 {roll}"
        self.pos_groups = [[0, 1, 2]]
        self.quat_groups = [[3, 4, 5, 6]]
        return f"""
<mujoco model="generic_object">
  {self.physics_options()}
{VISUAL}
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="30 30 0.1" rgba="0.3 0.35 0.3 1"
          friction="{self._fric}" solref="0.008 {damp}"/>
    <camera name="wide" pos="0 -12 3.0" xyaxes="1 0 0 0 0 1" fovy="60"/>
    <camera name="side" pos="0 -8 2.0" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <body name="obj" pos="0 0 2.0">
      <freejoint name="root"/>
      <camera name="chase" mode="track" pos="0 -5 1.0" xyaxes="1 0 0 0 0 1" fovy="55"/>
      <geom name="obj" type="{gtype}" size="{size}" mass="{m}"
            friction="{self._fric}" solref="0.008 {damp}"
            fluidshape="ellipsoid" rgba="0.85 0.7 0.35 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    def reset(self, randomize: bool = True):
        st = super().reset(randomize)
        self._bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self._r = float(self.model.geom_rbound[self.model.body_geomadr[self._bid]])
        return st

    def init_state(self, randomize: bool) -> None:
        if randomize:
            self.data.qpos[0:3] = [self.rng.uniform(-2, 2), self.rng.uniform(-2, 2),
                                   self.rng.uniform(0.8, 3.2)]
            q = self.rng.normal(size=4); q /= np.linalg.norm(q)
            self.data.qpos[3:7] = q
            # thrown in ANY direction, not merely dropped
            self.data.qvel[0:3] = [self.rng.normal(0, 1.8), self.rng.normal(0, 1.8),
                                   self.rng.uniform(-1.0, 3.0)]
            self.data.qvel[3:6] = self.rng.normal(0, 4.0, size=3)
        else:
            self.data.qpos[0:3] = [0, 0, 2.0]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qvel[0:3] = [1.2, 0.4, 1.5]

    @property
    def done(self) -> bool:
        v = float(np.linalg.norm(self.data.qvel[0:3]))
        w = float(np.linalg.norm(self.data.qvel[3:6]))
        return bool(self.data.qpos[2] < self._r * 1.6 and v < 0.10 and w < 0.6)
