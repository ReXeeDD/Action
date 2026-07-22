"""
mirage.py — the "ghost of the future" video.

A fixed camera watches the leaf fall. At every frame we take the last few frames
of motion, ask the network to predict the rest of the fall, project that predicted
path back into the camera image, and paint it as a glowing ghost *ahead* of the
leaf — a mirage of where it's about to go. The true path is drawn faintly so you
can watch the prediction tighten onto reality as the leaf nears the ground.

    python -m action.mirage --episode 0 --out runs/mirage.mp4

Uses the deterministic predictor (runs/leaf_mlp.pt) by default; pass
--ensemble runs/leaf_ensemble.pt to draw the cone of futures instead of one line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
import imageio
import mujoco

from action.leaf_world import LeafWorld
from action.camera import cam_id, project
from action.rollout import load_model, predict_future
from action.dataset import load_episodes


def _set_state(data, state):
    data.qpos[:7] = state[:7]
    data.qvel[:6] = state[7:13]


def _draw_polyline(img, uv, front, color, thickness=2, alpha=1.0):
    """Draw a projected path with a soft glow. uv:(N,2), front:(N,) bool."""
    pts = uv[front]
    if len(pts) < 2:
        return
    pts = pts.astype(np.int32)
    overlay = img.copy()
    cv2.polylines(overlay, [pts], False, color, thickness + 4, cv2.LINE_AA)  # glow
    cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)      # core
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_marker(img, uv, front, color, r=7):
    if not front:
        return
    x, y = int(uv[0]), int(uv[1])
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, r * 2, 2, cv2.LINE_AA)


def make_mirage(episode, ckpt, out, W=720, H=540, stride=3, horizon=None,
                ensemble=None, samples=60, fps=30, cam="chase", temperature=1.0,
                memory=None):
    device = "cpu"
    world = LeafWorld(seed=0)
    world.reset(randomize=False)
    model_mj, data = world.model, world.data
    cid = cam_id(model_mj, cam)
    renderer = mujoco.Renderer(model_mj, height=H, width=W)

    if ensemble:
        from action.cone import load_ensemble, sample_futures
        members, xn, yn, history = load_ensemble(ensemble, device)
    elif memory:
        from action.train_memory import load_mem, mem_predict
        net, hist_cap, fm, fs, dm, dsd = load_mem(memory, device)
        history = 12   # min frames to watch before the memory can say anything
    else:
        net, xn, yn, history = load_model(ckpt, device)

    traj = episode
    frames = []
    ORANGE, GREEN, WHITE = (255, 120, 30), (60, 220, 90), (235, 235, 235)

    for t in range(history, len(traj), stride):
        _set_state(data, traj[t])
        mujoco.mj_forward(model_mj, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()

        # subtle locator ring on the leaf (it vanishes when seen edge-on)
        uv_leaf, fr_leaf = project(model_mj, data, cid, traj[t:t+1, 0:3], W, H)
        if fr_leaf[0]:
            cv2.circle(img, (int(uv_leaf[0, 0]), int(uv_leaf[0, 1])), 10,
                       (120, 235, 120), 1, cv2.LINE_AA)

        seed_window = traj[t - history:t]
        true_future = traj[t:]
        n_steps = len(true_future) if horizon is None else min(horizon, len(true_future))
        if n_steps < 2:
            frames.append(img); continue

        # true path (faint), for honest comparison
        uv_t, fr_t = project(model_mj, data, cid, true_future[:n_steps, 0:3], W, H)
        _draw_polyline(img, uv_t, fr_t, GREEN, thickness=1, alpha=0.5)

        if ensemble:
            samp = sample_futures(members, xn, yn, history, seed_window,
                                  n_steps=n_steps, n_samples=samples, device=device,
                                  rng=np.random.default_rng(t), temperature=temperature)
            for s in samp:
                uv_s, fr_s = project(model_mj, data, cid, s[:, 0:3], W, H)
                _draw_polyline(img, uv_s, fr_s, ORANGE, thickness=1, alpha=0.12)
            land = samp[:, -1, 0:3]
            uv_l, fr_l = project(model_mj, data, cid, land, W, H)
            for p, f in zip(uv_l, fr_l):
                _draw_marker(img, p, f, ORANGE, r=4)
        elif memory:
            # the memory model watches EVERYTHING so far (traj[:t+1]) and predicts
            # the rest of the fall — the line tightens onto truth as it descends.
            m_steps = n_steps - 1
            if m_steps >= 2:
                pred = mem_predict(net, hist_cap, fm, fs, dm, dsd,
                                   traj[:t + 1], traj[t], m_steps, device)
                uv_p, fr_p = project(model_mj, data, cid, pred[:, 0:3], W, H)
                _draw_polyline(img, uv_p, fr_p, ORANGE, thickness=2, alpha=0.95)
                uv_pl, fr_pl = project(model_mj, data, cid, pred[-1:, 0:3], W, H)
                _draw_marker(img, uv_pl[0], fr_pl[0], ORANGE, r=8)
        else:
            pred = predict_future(net, xn, yn, history, seed_window,
                                  n_steps=n_steps, device=device)
            uv_p, fr_p = project(model_mj, data, cid, pred[:, 0:3], W, H)
            _draw_polyline(img, uv_p, fr_p, ORANGE, thickness=2, alpha=0.95)
            uv_pl, fr_pl = project(model_mj, data, cid, pred[-1:, 0:3], W, H)
            _draw_marker(img, uv_pl[0], fr_pl[0], ORANGE, r=8)

        # true landing marker
        uv_tl, fr_tl = project(model_mj, data, cid, true_future[-1:, 0:3], W, H)
        _draw_marker(img, uv_tl[0], fr_tl[0], GREEN, r=8)

        cv2.putText(img, f"t={t*0.008:4.1f}s  predicting {n_steps*0.008:.1f}s ahead",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "orange = predicted future   green = truth",
                    (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        frames.append(img)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
    print(f"wrote {len(frames)} frames -> {out}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="runs/leaf_mlp.pt")
    ap.add_argument("--ensemble", type=str, default=None)
    ap.add_argument("--memory", type=str, default=None,
                    help="seq memory model ckpt (runs/leaf_mem.pt) — single sharpening line")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--out", type=str, default="runs/mirage.mp4")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cam", type=str, default="chase", choices=["chase", "side"])
    args = ap.parse_args()

    eps = load_episodes(args.data)
    make_mirage(eps[args.episode], args.ckpt, args.out, stride=args.stride,
                horizon=args.horizon, ensemble=args.ensemble, samples=args.samples,
                temperature=args.temperature, cam=args.cam, memory=args.memory)


if __name__ == "__main__":
    main()
