"""
live.py — drop fresh leaves and watch the future get predicted, live.

Unlike `mirage.py` (which replays a recorded dataset episode), this drops BRAND NEW
leaves: each fall gets a random release point, a random height, and fresh randomized
air/wind/sway. Nothing here was ever in the training set.

The prediction is genuinely blind: at every frame the memory model is handed only
what it has *watched so far*, and it rolls the rest of the fall forward — including
where the leaf will land. It does NOT get to see the future. The predicted path is
cut at the first moment it predicts ground contact, and that point is the predicted
landing. Watch the orange line snap onto the leaf as it descends and the model
figures out the air it's falling through.

    python -m action.live --memory runs/leaf_mem.pt --drops 4 --out runs/live.mp4
    python -m action.live --memory runs/leaf_mem.pt --drops 3 --show   # live window

Colours: green = where it has actually been (and, faintly, where it will go),
orange = the model's predicted future + predicted landing X.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
import imageio
import mujoco

from action.worlds import make_world, WORLD_NAMES
from action.camera import cam_id, project
from action.train_memory import load_mem, mem_predict

GREEN, ORANGE, WHITE, DIM = (60, 220, 90), (255, 120, 30), (235, 235, 235), (150, 150, 150)
GROUND_Z = 0.03


def _poly(img, uv, front, color, thickness=2, alpha=1.0):
    pts = uv[front]
    if len(pts) < 2:
        return
    pts = pts.astype(np.int32)
    ov = img.copy()
    cv2.polylines(ov, [pts], False, color, thickness + 3, cv2.LINE_AA)
    cv2.polylines(ov, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _mark(img, uv, front, color, r=8):
    if not front:
        return
    cv2.drawMarker(img, (int(uv[0]), int(uv[1])), color,
                   cv2.MARKER_TILTED_CROSS, r * 2, 2, cv2.LINE_AA)


def one_drop(net, hist_cap, fm, fs, dm, dsd, rng, cam="wide", W=900, H=640,
             n_ahead=260, stride=2, min_watch=12, show=False, show_truth=True,
             spread=1.5, h_lo=2.2, h_hi=3.8, device="cpu", world_name="leaf"):
    """Simulate one fresh episode of ANY world, predicting live."""
    world = make_world(world_name, seed=int(rng.integers(0, 2**31 - 1)))
    world.reset(randomize=True)
    # If this object has a free root (leaf, ball, generic object) release it from a
    # random spot at a random height. Pendulums and n-body have no free root, so we
    # leave their own randomized initial condition alone.
    x0, y0, h0 = 0.0, 0.0, 0.0
    if world.model.nq >= 7:
        x0, y0 = rng.uniform(-spread, spread, size=2)
        h0 = rng.uniform(h_lo, h_hi)
        world.data.qpos[0:3] = [x0, y0, h0]
    mujoco.mj_forward(world.model, world.data)

    model_mj, data = world.model, world.data
    cid = cam_id(model_mj, cam)
    renderer = mujoco.Renderer(model_mj, height=H, width=W)

    # roll the fall out, but render/predict as we go (the model only ever sees the past)
    # Simulate once, keeping BOTH the universal 13-dim target state (what the model
    # consumes) and the engine pose at each frame (what the renderer needs). The
    # universal state cannot pose the scene by itself -- a pendulum's qpos is joint
    # angles, not a position/quaternion -- so we store qpos/qvel snapshots too.
    from action.entities import entity_state
    ti = 0 if world_name == "leaf" else world.target_index
    traj, poses = [entity_state(world.model, world.data)[ti]],                   [(world.data.qpos.copy(), world.data.qvel.copy())]
    for _ in range(world.max_steps if world_name != "leaf" else 900):
        world.step(2) if world_name == "leaf" else world.step()
        traj.append(entity_state(world.model, world.data)[ti])
        poses.append((world.data.qpos.copy(), world.data.qvel.copy()))
        if world_name == "leaf":
            if world.on_ground:
                break
        else:
            if world.done or world.target_at_rest(traj):
                break
    traj = np.asarray(traj, dtype=np.float32)
    true_land = traj[-1, 0:3]

    frames, pred_land, land_track = [], None, []
    for t in range(min_watch, len(traj), stride):
        data.qpos[:], data.qvel[:] = poses[t]      # pose the scene for this frame
        mujoco.mj_forward(model_mj, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()

        # where it has actually been so far
        uv_p, fr_p = project(model_mj, data, cid, traj[:t + 1, 0:3], W, H)
        _poly(img, uv_p, fr_p, GREEN, thickness=1, alpha=0.55)
        if show_truth and t + 1 < len(traj):
            uv_f, fr_f = project(model_mj, data, cid, traj[t:, 0:3], W, H)
            _poly(img, uv_f, fr_f, DIM, thickness=1, alpha=0.35)

        # BLIND prediction: only the watched history goes in
        pred = mem_predict(net, hist_cap, fm, fs, dm, dsd,
                           traj[:t + 1], traj[t], n_ahead, device)
        below = np.where(pred[:, 2] <= GROUND_Z)[0]
        cut = int(below[0]) + 1 if len(below) else len(pred)
        pred = pred[:cut]
        uv_q, fr_q = project(model_mj, data, cid, pred[:, 0:3], W, H)
        _poly(img, uv_q, fr_q, ORANGE, thickness=2, alpha=0.95)
        if len(pred):
            pred_land = pred[-1, 0:3]
            uv_l, fr_l = project(model_mj, data, cid, pred[-1:, 0:3], W, H)
            _mark(img, uv_l[0], fr_l[0], ORANGE, r=9)

        # leaf locator
        uv_c, fr_c = project(model_mj, data, cid, traj[t:t + 1, 0:3], W, H)
        if fr_c[0]:
            cv2.circle(img, (int(uv_c[0, 0]), int(uv_c[0, 1])), 11, (120, 235, 120), 1,
                       cv2.LINE_AA)

        err = np.linalg.norm(pred_land[:2] - true_land[:2]) if pred_land is not None else 0.0
        land_track.append((t / max(len(traj) - 1, 1), err))   # (fraction watched, error)
        cv2.putText(img, f"drop from ({x0:+.2f}, {y0:+.2f}) at {h0:.2f} m", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"t={t*0.008:4.2f}s   watched {t} frames   "
                         f"predicted landing off by {err*100:5.1f} cm", (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "orange = PREDICTED future + landing    green = actual path",
                    (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        frames.append(img)
        if show:
            cv2.imshow("Action - live leaf prediction", img[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def err_at(frac):
        """Landing error using only the frames watched up to `frac` of the fall.
        This is the honest number — measuring at the end is trivial (the leaf has
        already landed, so the model only has to predict one step)."""
        cand = [e for f, e in land_track if f <= frac]
        return cand[-1] * 100 if cand else float("nan")

    return frames, {"x0": x0, "y0": y0, "h0": h0, "fall_s": len(traj) * 0.008,
                    "err25": err_at(0.25), "err50": err_at(0.50), "err75": err_at(0.75)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", type=str, default="runs/leaf_mem.pt")
    ap.add_argument("--drops", type=int, default=4)
    ap.add_argument("--out", type=str, default="runs/live.mp4")
    ap.add_argument("--cam", type=str, default="wide", choices=["wide", "side", "chase"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--spread", type=float, default=1.5, help="drop-zone half-width (m)")
    ap.add_argument("--h-lo", type=float, default=2.2)
    ap.add_argument("--h-hi", type=float, default=3.8)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--show", action="store_true", help="live window while it runs")
    ap.add_argument("--no-truth", action="store_true", help="hide the true future (pure mirage)")
    ap.add_argument("--world", type=str, default="leaf",
                    help=f"any of: {', '.join(WORLD_NAMES)}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net, hist_cap, fm, fs, dm, dsd = load_mem(args.memory, "cpu")
    rng = np.random.default_rng(args.seed)
    all_frames = []
    for i in range(args.drops):
        fr, info = one_drop(net, hist_cap, fm, fs, dm, dsd, rng, cam=args.cam,
                            stride=args.stride, show=args.show,
                            show_truth=not args.no_truth, spread=args.spread,
                            h_lo=args.h_lo, h_hi=args.h_hi, world_name=args.world)
        all_frames += fr
        print(f"drop {i+1}: from ({info['x0']:+.2f},{info['y0']:+.2f}) at {info['h0']:.2f}m, "
              f"fell {info['fall_s']:.2f}s | landing call after watching "
              f"25%: {info['err25']:5.1f}cm   50%: {info['err50']:5.1f}cm   "
              f"75%: {info['err75']:5.1f}cm", flush=True)
    if args.show:
        cv2.destroyAllWindows()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, all_frames, fps=args.fps, macro_block_size=None)
    print(f"wrote {len(all_frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
