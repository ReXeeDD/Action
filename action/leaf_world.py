"""
leaf_world.py — a fluttering leaf in a physically real airworld (MuJoCo).

The aerodynamics are real: MuJoCo's *ellipsoid fluid model* gives the flat leaf
genuine drag, lift, and Magnus forces from its shape. On its own a rigid plate
tends to *glide* smoothly (like a falling business card), because MuJoCo's fluid
model is quasi-steady and can't produce the unsteady vortex shedding that makes a
real leaf flutter. We add that missing physics as small random *buffeting torque*
each step — turbulent nudges that tumble the leaf and turn the smooth glide into a
fluttering, zig-zagging fall.

Why the fall speed is what it is: terminal velocity ~ sqrt(2*m*g / (rho*Cd*A)),
so a light, high-drag leaf falls SLOWLY, and making it *lighter* makes it *slower*.
For a lively, natural-looking demo we use a slightly heavier leaf, a smaller drag
area, and a low (hand-height) drop.

State layout (per timestep), 13-dim:
    [0:3]  position   x, y, z            (world frame)
    [3:7]  orientation qw, qx, qy, qz    (unit quaternion)
    [7:10] linear velocity  vx, vy, vz   (world frame)
    [10:13] angular velocity wx, wy, wz  (body frame — MuJoCo free-joint convention)

This is exactly MuJoCo's free-joint qpos(7) ++ qvel(6).
"""

from __future__ import annotations

import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# The world as MJCF (MuJoCo's XML). Kept as a template so we can perturb the
# air/leaf per-episode by rebuilding cheaply.
# ---------------------------------------------------------------------------
def leaf_mjcf(
    density: float = 1.2,          # air density  (kg/m^3), ~sea level
    viscosity: float = 1.8e-5,     # air dynamic viscosity (Pa*s)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    leaf_size: tuple[float, float, float] = (0.055, 0.04, 0.0012),  # ellipsoid semi-axes (m)
    leaf_mass: float = 0.0022,     # ~2.2 g: heavier than a dry leaf so it falls at a lively pace
    start_height: float = 3.0,
) -> str:
    sx, sy, sz = leaf_size
    wx, wy, wz = wind
    return f"""
<mujoco model="leaf_world">
  <option timestep="0.004" integrator="RK4"
          density="{density}" viscosity="{viscosity}"
          wind="{wx} {wy} {wz}"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="20 20 0.1" rgba="0.3 0.35 0.3 1"/>
    <!-- Fixed camera looking along +y at the fall column. Image x = world +x,
         image up = world +z, so the projection math is exact and simple. -->
    <camera name="side" pos="0 -6 1.5" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <body name="leaf" pos="0 0 {start_height}">
      <freejoint name="root"/>
      <!-- A flat oval: looks like a leaf and its ellipsoid fluid interaction is exact. -->
      <geom name="leaf" type="ellipsoid" size="{sx} {sy} {sz}" mass="{leaf_mass}"
            fluidshape="ellipsoid"
            fluidcoef="0.5 0.25 1.5 1.0 1.0"
            rgba="0.45 0.72 0.18 1"/>
      <!-- Chase camera: follows the leaf's position (no rotation with its tumble),
           so any fall stays centered and the predicted cone stays in view. -->
      <camera name="chase" mode="track" pos="0 -3.2 0" xyaxes="1 0 0 0 0 1" fovy="55"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


# 13-dim state indices, exported for the rest of the pipeline.
POS = slice(0, 3)
QUAT = slice(3, 7)
LINVEL = slice(7, 10)
ANGVEL = slice(10, 13)
STATE_DIM = 13


class LeafWorld:
    """One randomizable leaf-fall episode generator."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._bid = -1
        self._k = 0

    # -- build a fresh, randomized world -----------------------------------
    def reset(self, randomize: bool = True) -> np.ndarray:
        if randomize:
            density = float(self.rng.uniform(1.05, 1.35))          # temperature/altitude proxy
            wind = self.rng.normal(0.0, 0.7, size=3)               # gentle ambient wind
            wind[2] *= 0.3                                          # less vertical wind
            leaf_mass = float(self.rng.uniform(0.0015, 0.0028))
            start_h = float(self.rng.uniform(2.5, 3.5))
            # swaying-force parameters — MuJoCo's fluid model can't flutter on its
            # own, so we impose the leaf's characteristic side-to-side sway: an
            # oscillation (its own amplitude/frequency/phase) plus turbulent noise.
            sway_amp = float(self.rng.uniform(0.03, 0.055))
            sway_freq = float(self.rng.uniform(0.025, 0.045))
            sway_phase = self.rng.uniform(0, 2 * np.pi, size=2)
        else:
            density, wind = 1.2, np.zeros(3)
            leaf_mass, start_h = 0.0022, 3.0
            sway_amp, sway_freq, sway_phase = 0.045, 0.03, np.array([0.0, 1.5])

        self.wind = np.asarray(wind, dtype=np.float64)
        self._sway_amp, self._sway_freq, self._sway_phase = sway_amp, sway_freq, sway_phase
        self._k = 0
        xml = leaf_mjcf(density=density, wind=tuple(wind),
                        leaf_mass=leaf_mass, start_height=start_h)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leaf")

        # randomized initial pose + kick so no two falls are alike
        quat = self.rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        self.data.qpos[POS] = [self.rng.uniform(-0.2, 0.2),
                               self.rng.uniform(-0.2, 0.2),
                               start_h]
        self.data.qpos[QUAT] = quat
        self.data.qvel[0:3] = self.rng.normal(0.0, 0.2, size=3)     # linear (qvel layout)
        self.data.qvel[3:6] = self.rng.normal(0.0, 3.0, size=3)     # angular, spin it
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[:7], self.data.qvel[:6]]).astype(np.float32)

    def _apply_forcing(self):
        """The leaf's swaying flutter — now fully DETERMINISTIC (no per-step RNG).

        MuJoCo's fluid model can't flutter, so we impose the sway as a sum of a few
        incommensurate sinusoids (quasi-periodic, irregular-looking, but a pure
        function of the episode's fixed params and the step counter). This removes the
        artificial information-theoretic wall: the future is now a deterministic
        function of state+phase, so it is learnable within the true chaos horizon.
        Per-episode variety still comes from the randomized amplitude/freq/phase set
        once in reset(); it just no longer changes every step."""
        a, f, ph, k = self._sway_amp, self._sway_freq, self._sway_phase, self._k
        side = a * np.array([np.sin(f * k + ph[0]),
                             0.6 * np.cos(0.8 * f * k + ph[1]),
                             0.0])
        # second incommensurate component replaces the old turbulent noise term
        side += 0.3 * a * np.array([np.sin(2.31 * f * k + ph[1]),
                                    np.cos(1.73 * f * k + ph[0]),
                                    0.0])
        self.data.xfrc_applied[self._bid, 0:3] = side
        self.data.xfrc_applied[self._bid, 3:6] = 1.5e-4 * np.array(
            [np.sin(1.9 * f * k + ph[0]), np.cos(2.7 * f * k + ph[1]),
             np.sin(0.7 * f * k)])
        self._k += 1

    def step(self, n_substeps: int = 1) -> np.ndarray:
        for _ in range(n_substeps):
            self._apply_forcing()
            mujoco.mj_step(self.model, self.data)
        return self.state()

    @property
    def on_ground(self) -> bool:
        return float(self.data.qpos[2]) <= 0.05

    def rollout(self, max_steps: int = 900, n_substeps: int = 2) -> np.ndarray:
        """Simulate until it lands (or max_steps). Returns (T, 13) trajectory."""
        traj = [self.state()]
        for _ in range(max_steps):
            traj.append(self.step(n_substeps))
            if self.on_ground:
                break
        return np.asarray(traj, dtype=np.float32)


if __name__ == "__main__":
    # smoke test: does the leaf load, fall at a natural pace, and actually flutter?
    for seed in range(3):
        w = LeafWorld(seed=seed)
        w.reset(randomize=True)
        traj = w.rollout()
        dt = 0.008
        fall_time = len(traj) * dt
        disp = np.linalg.norm(traj[-1, :3] - traj[0, :3])
        path = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1).sum()
        drift = np.linalg.norm(traj[-1, :2] - traj[0, :2])
        print(f"seed {seed}: fall {fall_time:4.1f}s  drop {traj[0,2]:.1f}->{traj[-1,2]:.2f}m  "
              f"drift {drift:.2f}m  flutter_ratio {path/max(disp,1e-6):.2f}  "
              f"(>1.3 = visibly fluttering)")
