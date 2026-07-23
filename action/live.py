"""
live.py — drop fresh objects and watch the future get predicted, live.

Unlike `mirage.py` (which replays a recorded dataset episode), this runs BRAND NEW
episodes: random release point, random height, fresh randomized physics. Nothing here
was ever in the training set.

The prediction is genuinely blind: at every frame the memory model is handed only
what it has *watched so far*, and it rolls the rest of the episode forward. It does
NOT get to see the future.

    python -m action.live --memory runs/general3.pt --world ball --drops 4
    python -m action.live --memory runs/general3.pt --world pendulum2 --show

Colours: green = where it has actually been (and, faintly, where it will go),
orange = the model's predicted future.
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
from action.train_memory import load_mem

GREEN, ORANGE, WHITE, DIM = (60, 220, 90), (255, 120, 30), (235, 235, 235), (150, 150, 150)

# Clearance above the ground plane at which we call it "contact". This is measured
# from the world's OWN ground plane, which is NOT always z=0 — a pendulum's floor sits
# at -(chain length)-0.3, roughly -1.0 m, and the bob swings *below* z=0 for half of
# every cycle. A hardcoded absolute `z <= 0.03` therefore marked all 260 predicted
# frames as "underground", truncated the predicted path to a single point, and froze
# the marker on the object's current position — the model's prediction was computed
# correctly and then thrown away. Worlds with no ground plane at all (n-body) are
# never cut.
GROUND_CLEARANCE = 0.03


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


def ground_level(model) -> float | None:
    """Height of this world's ground plane, or None if it has no ground."""
    zs = [float(model.geom_pos[g][2]) for g in range(model.ngeom)
          if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    return max(zs) if zs else None


def one_drop(mem, rng, cam="wide", W=900, H=640, n_ahead=260, stride=2, min_watch=12,
             show=False, show_truth=True, spread=1.5, h_lo=2.2, h_hi=3.8,
             world_name="leaf", pred_batch=48):
    """Simulate one fresh episode of ANY world, predicting live."""
    world = make_world(world_name, seed=int(rng.integers(0, 2**31 - 1)))
    world.reset(randomize=True)
    # If this world is a SINGLE free body (leaf, ball, generic object) release it from
    # a random spot at a random height. Multi-body worlds have a configured initial
    # arrangement — a pendulum chain, an n-body ring balanced for orbits — that must
    # not be disturbed.
    #
    # The test used to be `model.nq >= 7`, meaning "has a free joint". That silently
    # broke n-body: each mass is THREE SLIDE JOINTS, so nbody3 has nq=9 and nbody4
    # nq=12, both of which pass. Body 0 was teleported to a random point 2.2-3.8 m up
    # while keeping the orbital velocity computed for its original ring position, so
    # it left on a near-straight line and never visibly interacted with anything.
    # (nbody2, with nq=6, happened to escape this.) Counting bodies is the correct
    # test and works for every world.
    from action.entities import n_entities
    x0, y0, h0 = 0.0, 0.0, 0.0
    if world.model.nq >= 7 and n_entities(world.model) == 1:
        x0, y0 = rng.uniform(-spread, spread, size=2)
        h0 = rng.uniform(h_lo, h_hi)
        world.data.qpos[0:3] = [x0, y0, h0]
    mujoco.mj_forward(world.model, world.data)

    model_mj, data = world.model, world.data
    cid = cam_id(model_mj, cam)
    renderer = mujoco.Renderer(model_mj, height=H, width=W)

    # Simulate once, keeping BOTH the universal 13-dim target state (what the model
    # consumes) and the engine pose at each frame (what the renderer needs). The
    # universal state cannot pose the scene by itself -- a pendulum's qpos is joint
    # angles, not a position/quaternion -- so we store qpos/qvel snapshots too.
    from action.entities import entity_state
    ti = 0 if world_name == "leaf" else world.target_index
    traj = [entity_state(world.model, world.data)[ti]]
    poses = [(world.data.qpos.copy(), world.data.qvel.copy())]
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
    true_end = traj[-1, 0:3]

    gz = ground_level(model_mj)
    cut_z = None if gz is None else gz + GROUND_CLEARANCE

    # ---- predict every frame AT ONCE ------------------------------------------
    # Each frame's prediction depends only on that frame's history, never on another
    # frame's prediction, so they batch perfectly. Doing them one at a time made the
    # render cost (frames x rollout) sequential; batching collapses it.
    ts = list(range(min_watch, len(traj), stride))
    preds = []
    for i in range(0, len(ts), pred_batch):
        ch = ts[i:i + pred_batch]
        preds.append(mem.predict_batch([traj[:t + 1] for t in ch], traj[ch], n_ahead))
    preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, n_ahead, 13))

    frames, end_track, any_contact = [], [], False
    for fi, t in enumerate(ts):
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

        pred = preds[fi]
        contact = False
        if cut_z is not None:
            below = np.where(pred[:, 2] <= cut_z)[0]
            if len(below):
                pred, contact = pred[:int(below[0]) + 1], True
                any_contact = True
        uv_q, fr_q = project(model_mj, data, cid, pred[:, 0:3], W, H)
        _poly(img, uv_q, fr_q, ORANGE, thickness=2, alpha=0.95)
        pred_end = pred[-1, 0:3] if len(pred) else traj[t, 0:3]
        if contact:                      # only a real landing deserves the X marker
            uv_l, fr_l = project(model_mj, data, cid, pred[-1:, 0:3], W, H)
            _mark(img, uv_l[0], fr_l[0], ORANGE, r=9)

        uv_c, fr_c = project(model_mj, data, cid, traj[t:t + 1, 0:3], W, H)
        if fr_c[0]:
            cv2.circle(img, (int(uv_c[0, 0]), int(uv_c[0, 1])), 11, (120, 235, 120), 1,
                       cv2.LINE_AA)

        err = float(np.linalg.norm(pred_end[:2] - true_end[:2]))
        end_track.append((t / max(len(traj) - 1, 1), err))
        head = (f"drop from ({x0:+.2f}, {y0:+.2f}) at {h0:.2f} m" if world.model.nq >= 7
                else f"world: {world_name}")
        cv2.putText(img, head, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
                    cv2.LINE_AA)
        # A pendulum never lands, so a "landing error" would be meaningless for it —
        # say what is actually being shown instead.
        second = (f"predicted landing off by {err*100:5.1f} cm" if contact
                  else f"predicting {len(pred)*0.008:4.2f}s ahead")
        cv2.putText(img, f"t={t*0.008:4.2f}s   watched {t} frames   {second}", (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "orange = PREDICTED future    green = actual path",
                    (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        frames.append(img)
        if show:
            cv2.imshow("Action - live prediction", img[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def err_at(frac):
        """Error using only the frames watched up to `frac` of the episode. Measuring
        at the end is trivial — the object has already arrived."""
        cand = [e for f, e in end_track if f <= frac]
        return cand[-1] * 100 if cand else float("nan")

    return frames, {"x0": x0, "y0": y0, "h0": h0, "dur_s": len(traj) * 0.008,
                    "landing": any_contact, "err25": err_at(0.25),
                    "err50": err_at(0.50), "err75": err_at(0.75)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", type=str, default="runs/general.pt")
    ap.add_argument("--drops", type=int, default=4)
    ap.add_argument("--out", type=str, default="runs/live.mp4")
    ap.add_argument("--cam", type=str, default="wide", choices=["wide", "side", "chase"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-ahead", type=int, default=260, help="frames of future to predict")
    ap.add_argument("--pred-batch", type=int, default=48,
                    help="frames predicted per batched forward pass")
    ap.add_argument("--spread", type=float, default=1.5, help="drop-zone half-width (m)")
    ap.add_argument("--h-lo", type=float, default=2.2)
    ap.add_argument("--h-hi", type=float, default=3.8)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--show", action="store_true", help="live window while it runs")
    ap.add_argument("--no-truth", action="store_true", help="hide the true future")
    ap.add_argument("--world", type=str, default="leaf",
                    help=f"any of: {', '.join(WORLD_NAMES)}")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cuda makes the batched prediction pass much faster")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    mem = load_mem(args.memory, dev)
    rng = np.random.default_rng(args.seed)
    all_frames = []
    for i in range(args.drops):
        fr, info = one_drop(mem, rng, cam=args.cam, stride=args.stride,
                            n_ahead=args.n_ahead, show=args.show,
                            show_truth=not args.no_truth, spread=args.spread,
                            h_lo=args.h_lo, h_hi=args.h_hi, world_name=args.world,
                            pred_batch=args.pred_batch)
        all_frames += fr
        if info["landing"]:
            print(f"run {i+1}: {info['dur_s']:.2f}s | landing call after watching "
                  f"25%: {info['err25']:5.1f}cm   50%: {info['err50']:5.1f}cm   "
                  f"75%: {info['err75']:5.1f}cm", flush=True)
        else:
            print(f"run {i+1}: {info['dur_s']:.2f}s | no ground contact predicted "
                  f"(this world does not land); endpoint drift after 50%: "
                  f"{info['err50']:5.1f}cm", flush=True)
    if args.show:
        cv2.destroyAllWindows()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, all_frames, fps=args.fps, macro_block_size=None)
    print(f"wrote {len(all_frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
