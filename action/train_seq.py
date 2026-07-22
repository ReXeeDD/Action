"""
train_seq.py — train the MEMORY model (context encoder + rollout decoder).

The network watches a long history of the fall, distills it into a context vector
z (its "memory" of how this object interacts with this environment), then rolls the
whole future forward reading z at every step. We train on the ENTIRE rolled-out
horizon, not one step — so z is forced to be informative and the rollout is trained
to stay stable (the fix for the single-step compounding blow-up).

    python -m action.train_seq --data data/leaf --out runs/leaf_seq.pt \
        --hist 48 --pred 80 --epochs 25 --device cuda

Then measure how far the window opened:
    python -m action.train_seq --measure runs/leaf_seq.pt --data data/leaf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from action.dataset import load_episodes
from action.models import SeqPredictor
from action.leaf_world import STATE_DIM


class FallWindows(Dataset):
    """Yields (history_feat, current_state, anchor_pos, target_future_states)."""

    def __init__(self, episodes, hist: int, pred: int, stride: int = 2):
        self.eps = [torch.from_numpy(e).float() for e in episodes]
        self.hist, self.pred = hist, pred
        self.index = []
        for i, e in enumerate(self.eps):
            for t in range(hist, len(e) - pred, stride):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.eps[i]
        anchor = e[t - 1, 0:3]
        hist = e[t - self.hist:t].clone()                     # (H,13)
        hist[:, 0:3] = hist[:, 0:3] - anchor                  # translation-invariant
        cur = e[t - 1].clone()                                # absolute current state
        fut = e[t:t + self.pred].clone()                      # (P,13) absolute future
        return hist, cur, anchor, fut


def _fit_norms(ds, n=6000):
    """Estimate feature and delta normalization from a random sample."""
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    feats, deltas = [], []
    for k in idx:
        hist, cur, anchor, fut = ds[k]
        feats.append(hist)
        seq = torch.cat([cur.unsqueeze(0), fut], dim=0)       # (P+1,13)
        deltas.append(seq[1:] - seq[:-1])                     # (P,13)
    feats = torch.cat(feats, 0)
    deltas = torch.cat(deltas, 0)
    fm, fs = feats.mean(0), feats.std(0).clamp_min(1e-6)
    dm, dsd = deltas.mean(0), deltas.std(0).clamp_min(1e-6)
    return fm, fs, dm, dsd


def train(args):
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    eps = load_episodes(args.data)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, int(0.1 * len(eps)))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]

    tr = FallWindows(tr_eps, args.hist, args.pred, stride=args.stride)
    va = FallWindows(val_eps, args.hist, args.pred, stride=args.pred)  # sparse val
    fm, fs, dm, dsd = _fit_norms(tr)
    fm, fs, dm, dsd = (x.to(dev) for x in (fm, fs, dm, dsd))
    print(f"train windows {len(tr)}  val windows {len(va)}  "
          f"hist={args.hist} pred={args.pred} device={dev}")

    net = SeqPredictor(context_dim=args.ctx).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    vdl = DataLoader(va, batch_size=args.batch, shuffle=False)

    def run_batch(hist, cur, anchor, fut, train_mode):
        hist, cur, anchor, fut = (t.to(dev) for t in (hist, cur, anchor, fut))
        hist_n = (hist - fm) / fs
        seq = torch.cat([cur.unsqueeze(1), fut], dim=1)       # (B,P+1,13)
        tgt_delta = seq[:, 1:] - seq[:, :-1]                  # (B,P,13)
        tgt_delta_n = (tgt_delta - dm) / dsd
        z = net.encode(hist_n)
        pred_delta_n, _ = net.decode(z, cur, anchor, args.pred, fm, fs, dm, dsd)
        loss = ((pred_delta_n - tgt_delta_n) ** 2).sum(-1).mean()
        if train_mode:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        return loss.item()

    best = float("inf")
    for ep in range(args.epochs):
        net.train()
        tl = [run_batch(*b, True) for b in dl]
        net.eval()
        with torch.no_grad():
            vl = [run_batch(*b, False) for b in vdl]
        tr_l, va_l = np.mean(tl), np.mean(vl)
        tag = ""
        if va_l < best:
            best = va_l
            ckpt = {"state_dict": net.state_dict(), "hist": args.hist, "pred": args.pred,
                    "ctx": args.ctx, "feat_mean": fm.cpu().numpy(), "feat_std": fs.cpu().numpy(),
                    "delta_mean": dm.cpu().numpy(), "delta_std": dsd.cpu().numpy()}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, args.out)
            tag = "  <-- saved"
        print(f"epoch {ep+1:3d}  train {tr_l:.4f}  val {va_l:.4f}{tag}")
    print(f"best val {best:.4f}  -> {args.out}")


# ----------------------------------------------------------------------------
def load_seq(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = SeqPredictor(context_dim=ck["ctx"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    t = lambda k: torch.tensor(ck[k], device=device)
    return net, ck["hist"], t("feat_mean"), t("feat_std"), t("delta_mean"), t("delta_std")


@torch.no_grad()
def seq_predict(net, hist_len, fm, fs, dm, dsd, seed_window, n_steps, device="cpu"):
    """seed_window: (>=hist_len, 13) absolute. Returns (n_steps,13) predicted states."""
    w = torch.from_numpy(np.asarray(seed_window[-hist_len:])).float().to(device)
    anchor = w[-1, 0:3].clone()
    hist = w.clone(); hist[:, 0:3] = hist[:, 0:3] - anchor
    z = net.encode(((hist - fm) / fs).unsqueeze(0))
    cur = w[-1:].clone()
    _, states = net.decode(z, cur, anchor.unsqueeze(0), n_steps, fm, fs, dm, dsd)
    return states[0].cpu().numpy()


def measure(args):
    dev = "cpu"
    net, hist_len, fm, fs, dm, dsd = load_seq(args.measure, dev)
    eps = load_episodes(args.data)
    dt = 0.008
    errs = []
    for traj in eps[1400:1470]:
        if len(traj) < hist_len + 20:
            continue
        seed = traj[:hist_len]; truth = traj[hist_len:]
        pred = seq_predict(net, hist_len, fm, fs, dm, dsd, seed, len(truth), dev)
        errs.append(np.linalg.norm(pred[:, 0:3] - truth[:, 0:3], axis=1))
    maxlen = max(len(e) for e in errs)
    avg = np.array([np.mean([e[i] for e in errs if len(e) > i]) for i in range(maxlen)])
    print("MEMORY-model single-line error vs horizon:")
    for sec in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        i = int(sec / dt)
        if i < len(avg):
            print(f"  {sec:4.2f}s ({sec*1.8:.2f} m fall): err {avg[i]*100:6.1f} cm")
    cross = np.where(avg > 0.10)[0]
    print(f"window (err<10cm): {cross[0]*dt:.2f}s ~= {cross[0]*dt*1.8:.2f} m of fall"
          if len(cross) else f"FULL FALL <10cm! max {avg.max()*100:.1f}cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--out", type=str, default="runs/leaf_seq.pt")
    ap.add_argument("--measure", type=str, default=None, help="ckpt to measure instead of train")
    ap.add_argument("--hist", type=int, default=48)
    ap.add_argument("--pred", type=int, default=80)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.measure:
        measure(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
