"""
camera.py — turn 3D world points into camera pixels, so we can paint the
predicted future straight onto the rendered view (the "mirage of the future").

MuJoCo's fixed camera looks down its local -z axis (OpenGL convention). Given the
camera's world pose (cam_xpos, cam_xmat) and vertical field of view (cam_fovy),
a world point projects to a pixel by:

    p_cam = R^T (p_world - cam_pos)          # into camera frame
    in front of camera  <=>  p_cam.z < 0     # camera looks along -z
    f = (H/2) / tan(fovy/2)                   # focal length in pixels
    u = W/2 + f * ( x_cam / -z_cam )
    v = H/2 - f * ( y_cam / -z_cam )          # image y points down
"""

from __future__ import annotations

import numpy as np
import mujoco


def cam_id(model, name: str = "side") -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def project(model, data, cid: int, pts_world: np.ndarray, W: int, H: int):
    """pts_world: (N,3). Returns (uv (N,2) float, in_front (N,) bool)."""
    pts = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
    cam_pos = data.cam_xpos[cid]                       # (3,)
    R = data.cam_xmat[cid].reshape(3, 3)               # cols = cam x,y,z in world
    fovy = np.deg2rad(model.cam_fovy[cid])
    f = (H / 2.0) / np.tan(fovy / 2.0)

    p_cam = (pts - cam_pos) @ R                         # == R^T (p - c) per row
    z = p_cam[:, 2]
    in_front = z < -1e-6
    denom = np.where(in_front, -z, 1.0)
    u = W / 2.0 + f * (p_cam[:, 0] / denom)
    v = H / 2.0 - f * (p_cam[:, 1] / denom)
    return np.stack([u, v], axis=1), in_front


if __name__ == "__main__":
    # Validate: the projected leaf position must land on the leaf's pixels.
    from action.leaf_world import LeafWorld
    W, H = 640, 480
    w = LeafWorld(seed=3)
    w.reset(randomize=False)
    for _ in range(120):
        w.step(2)
    mujoco.mj_forward(w.model, w.data)
    cid = cam_id(w.model, "side")

    r = mujoco.Renderer(w.model, height=H, width=W)
    r.update_scene(w.data, camera="side")
    img = r.render()

    leaf_pos = w.data.qpos[:3].copy()
    uv, front = project(w.model, w.data, cid, leaf_pos, W, H)
    u, v = uv[0]
    print(f"leaf world pos {np.round(leaf_pos,2)}  -> pixel ({u:.0f},{v:.0f})  in_front={front[0]}")

    # find the leaf in the image by its green-ish color, compare centroids
    r_, g_, b_ = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    mask = (g_ > 90) & (g_ > r_ + 15) & (g_ > b_ + 25)
    ys, xs = np.where(mask)
    if len(xs):
        print(f"green leaf pixels centroid ({xs.mean():.0f},{ys.mean():.0f})  n={len(xs)}")
        err = np.hypot(u - xs.mean(), v - ys.mean())
        print(f"projection error = {err:.1f} px  ({'GOOD' if err < 25 else 'CHECK MATH'})")
    else:
        print("no green pixels found (leaf out of view?)")
