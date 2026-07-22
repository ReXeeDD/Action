"""
train_memory.py — the STREAMING memory model, trained all the way to landing.

Upgrade over train_seq.py:
  * The memory grows. Instead of a fixed 48-frame window, the encoder reads ALL of
    the fall observed so far (up to a cap), so it accumulates evidence as the leaf
    descends — "look at everything that happened, then predict the rest."
  * We train the decoder to LANDING (the whole remaining trajectory, variable
    length), not a fixed 80 steps — so long-horizon accuracy is what we optimize.
  * Variable-length histories and futures are padded + masked so they batch cleanly.

At the end it prints the two numbers that matter — the prediction WINDOW and the
"sharpens as it falls" curve — so you capture them before the env closes.

    python -m action.train_memory --data data/leaf --out runs/leaf_mem.pt \
        --epochs 30 --hist-cap 160 --fut-cap 220 --device cuda

    python -m action.train_memory --measure runs/leaf_mem.pt --data data/leaf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from action.dataset import load_episodes
from action.models import SeqPredictor


class StreamWindows(Dataset):
    """Growing-history windows. For a start point t: history = all frames up to t
    (capped), future = the rest of the fall up to t+fut_cap."""

    def __init__(self, episodes, hist_cap=160, fut_cap=220, min_hist=12, stride=3):
        self.eps = [torch.from_numpy(e).float() for e in episodes]
        self.hist_cap, self.fut_cap, self.min_hist = hist_cap, fut_cap, min_hist
        self.index = []
        for i, e in enumerate(self.eps):
            for t in range(min_hist, len(e) - 2, stride):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.eps[i]
        anchor = e[t - 1, 0:3]
        h0 = max(0, t - self.hist_cap)
        hist = e[h0:t].clone()
        hist[:, 0:3] = hist[:, 0:3] - anchor            # translation-invariant
        cur = e[t - 1].clone()
        fut = e[t:min(len(e), t + self.fut_cap)].clone()
        return hist, cur, anchor, fut


def collate(batch):
    """Right-pad variable-length histories and futures; build masks."""
    hists, curs, anchors, futs = zip(*batch)
    B = len(batch)
    Lh = max(h.size(0) for h in hists)
    Lf = max(f.size(0) for f in futs)
    D = hists[0].size(1)
    hist_pad = torch.zeros(B, Lh, D)
    hist_mask = torch.ones(B, Lh, dtype=torch.bool)      # True = padded
    fut_pad = torch.zeros(B, Lf, D)
    fut_valid = torch.zeros(B, Lf)                        # 1 = real future step
    for b, (h, f) in enumerate(zip(hists, futs)):
        hist_pad[b, :h.size(0)] = h
        hist_mask[b, :h.size(0)] = False
        fut_pad[b, :f.size(0)] = f
        fut_valid[b, :f.size(0)] = 1.0
    return (hist_pad, torch.stack(curs), torch.stack(anchors),
            fut_pad, hist_mask, fut_valid)


def _fit_norms(ds, n=6000):
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    feats, deltas = [], []
    for k in idx:
        hist, cur, anchor, fut = ds[k]
        feats.append(hist)
        seq = torch.cat([cur.unsqueeze(0), fut], dim=0)
        deltas.append(seq[1:] - seq[:-1])
    feats, deltas = torch.cat(feats, 0), torch.cat(deltas, 0)
    return (feats.mean(0), feats.std(0).clamp_min(1e-6),
            deltas.mean(0), deltas.std(0).clamp_min(1e-6))


def train(args):
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    eps = load_episodes(args.data)
    perm = np.random.default_rng(args.seed).permutation(len(eps))
    n_val = max(1, int(0.1 * len(eps)))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]

    tr = StreamWindows(tr_eps, args.hist_cap, args.fut_cap, stride=args.stride)
    va = StreamWindows(val_eps, args.hist_cap, args.fut_cap, stride=20)  # denser, trustworthy val
    fm, fs, dm, dsd = (x.to(dev) for x in _fit_norms(tr))
    print(f"train windows {len(tr)}  val windows {len(va)}  "
          f"hist_cap={args.hist_cap} fut_cap={args.fut_cap} device={dev}", flush=True)

    net = SeqPredictor(context_dim=args.ctx).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    vdl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    def run(batch, train_mode):
        hist, cur, anchor, fut, hmask, fvalid = (t.to(dev) for t in batch)
        hist_n = (hist - fm) / fs
        if train_mode and args.noise > 0:                    # input-noise augmentation
            hist_n = hist_n + args.noise * torch.randn_like(hist_n)
        z = net.encode(hist_n, key_padding_mask=hmask)
        n_steps = fut.size(1)
        pred_dn, states = net.decode(z, cur, anchor, n_steps, fm, fs, dm, dsd)
        denom = fvalid.sum().clamp_min(1.0)
        # per-step delta loss (shape of the motion)
        seq = torch.cat([cur.unsqueeze(1), fut], dim=1)
        tgt_dn = ((seq[:, 1:] - seq[:, :-1]) - dm) / dsd
        delta_loss = (((pred_dn - tgt_dn) ** 2).sum(-1) * fvalid).sum() / denom
        # cumulative POSITION loss (what the window actually measures, in metres)
        pos_se = ((states[:, :, 0:3] - fut[:, :, 0:3]) ** 2).sum(-1)   # (B, n_steps) m^2
        pos_loss = (pos_se * fvalid).sum() / denom
        loss = delta_loss + args.pos_weight * pos_loss
        if train_mode:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        # select checkpoints on the position error (metres) — the metric we care about
        return loss.item(), pos_loss.item()

    save = lambda: torch.save(
        {"state_dict": net.state_dict(), "hist_cap": args.hist_cap, "ctx": args.ctx,
         "n_time_freq": net.n_time_freq,
         "feat_mean": fm.cpu().numpy(), "feat_std": fs.cpu().numpy(),
         "delta_mean": dm.cpu().numpy(), "delta_std": dsd.cpu().numpy()}, args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for ep in range(args.epochs):
        net.train(); tr = [run(b, True) for b in dl]
        net.eval()
        with torch.no_grad():
            va = [run(b, False) for b in vdl]
        sched.step()
        tr_loss = np.mean([x[0] for x in tr])
        va_pos = np.mean([x[1] for x in va])                 # val position error (m^2)
        tag = ""
        if va_pos < best:
            best = va_pos; save(); tag = "  <-- saved"
        print(f"epoch {ep+1:3d}  train {tr_loss:.4f}  "
              f"val_pos {va_pos*1e4:.1f}cm2  (rmse {np.sqrt(va_pos)*100:.1f}cm){tag}", flush=True)
    print(f"best val_pos rmse {np.sqrt(best)*100:.1f}cm  -> {args.out}", flush=True)

    print("\n" + "=" * 60 + "\nKEY RESULTS (share these)\n" + "=" * 60, flush=True)
    _diagnostics(args.out, args.data, dev="cpu")


# ----------------------------------------------------------------------------
def load_mem(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = SeqPredictor(context_dim=ck["ctx"], n_time_freq=ck.get("n_time_freq", 8)).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    t = lambda k: torch.tensor(ck[k], device=device)
    return net, ck["hist_cap"], t("feat_mean"), t("feat_std"), t("delta_mean"), t("delta_std")


@torch.no_grad()
def mem_predict(net, hist_cap, fm, fs, dm, dsd, history, cur, n_steps, device="cpu"):
    """history: (Lh,13) absolute (all observed so far). cur: (13,) current state."""
    w = torch.from_numpy(np.asarray(history[-hist_cap:])).float().to(device)
    anchor = torch.from_numpy(np.asarray(cur[0:3])).float().to(device)
    hn = w.clone(); hn[:, 0:3] = hn[:, 0:3] - anchor
    z = net.encode(((hn - fm) / fs).unsqueeze(0))
    cur_t = torch.from_numpy(np.asarray(cur)).float().unsqueeze(0).to(device)
    _, states = net.decode(z, cur_t, anchor.unsqueeze(0), n_steps, fm, fs, dm, dsd)
    return states[0].cpu().numpy()


def _diagnostics(ckpt, data, dev="cpu"):
    net, hist_cap, fm, fs, dm, dsd = load_mem(ckpt, dev)
    eps = load_episodes(data)
    ev = eps[1400:1470]
    dt = 0.008

    # (1) window: predict to landing from an early point (12 frames watched)
    errs = []
    for traj in ev:
        if len(traj) < 40:
            continue
        t0 = 12
        pred = mem_predict(net, hist_cap, fm, fs, dm, dsd, traj[:t0], traj[t0 - 1],
                           len(traj) - t0, dev)
        errs.append(np.linalg.norm(pred[:, 0:3] - traj[t0:, 0:3], axis=1))
    maxlen = max(len(e) for e in errs)
    avg = np.array([np.mean([e[i] for e in errs if len(e) > i]) for i in range(maxlen)])
    print("prediction error vs horizon (watched only first 12 frames):", flush=True)
    for sec in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        i = int(sec / dt)
        if i < len(avg):
            print(f"  {sec:4.2f}s ({sec*1.8:.2f} m fall): err {avg[i]*100:6.1f} cm", flush=True)
    cross = np.where(avg > 0.10)[0]
    print(f"  WINDOW (err<10cm): {cross[0]*dt:.2f}s ~= {cross[0]*dt*1.8:.2f} m of fall"
          if len(cross) else f"  WINDOW: FULL FALL <10cm! max {avg.max()*100:.1f}cm", flush=True)

    # (2) sharpens as it falls: fixed 0.48s-ahead error from growing observation
    H = 60
    print("\nsharpens as it falls? (fixed 0.48s-ahead error, growing memory):", flush=True)
    for fr in [0.20, 0.35, 0.50, 0.65]:
        es = []
        for traj in ev:
            t0 = int(fr * len(traj))
            if t0 < 12 or t0 + H >= len(traj):
                continue
            pred = mem_predict(net, hist_cap, fm, fs, dm, dsd, traj[:t0], traj[t0 - 1], H, dev)
            es.append(np.linalg.norm(pred[:, 0:3] - traj[t0:t0 + H, 0:3], axis=1)[-1])
        if es:
            print(f"  watched {int(fr*100):2d}% of fall: +0.48s err = {np.mean(es)*100:5.1f} cm",
                  flush=True)


def measure(args):
    print("=" * 60 + "\nKEY RESULTS\n" + "=" * 60, flush=True)
    _diagnostics(args.measure, args.data, dev="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--out", type=str, default="runs/leaf_mem.pt")
    ap.add_argument("--measure", type=str, default=None)
    ap.add_argument("--hist-cap", type=int, default=160)
    ap.add_argument("--fut-cap", type=int, default=220)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=10.0,
                    help="weight on the cumulative position (metre) loss")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    measure(args) if args.measure else train(args)


if __name__ == "__main__":
    main()
