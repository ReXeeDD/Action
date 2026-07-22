"""
train.py — fit the MLP predictor on leaf trajectories.

    python -m action.train --data data/leaf --epochs 40 --history 6

Saves runs/leaf_mlp.pt containing weights + normalization stats + config, so
rollout.py can reconstruct everything with no guessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from action.dataset import load_episodes, build_supervised, Normalizer
from action.models import MLPPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--out", type=str, default="runs/leaf_mlp.pt")
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    eps = load_episodes(args.data)
    # split by EPISODE so the val set is genuinely unseen falls, not shuffled frames
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]

    Xtr, Ytr = build_supervised(train_eps, args.history)
    Xva, Yva = build_supervised(val_eps, args.history)
    print(f"train samples={len(Xtr)}  val samples={len(Xva)}")

    xn = Normalizer.fit(Xtr)
    yn = Normalizer.fit(Ytr)

    def to_t(a):
        return torch.from_numpy(a).to(device)

    Xtr_t, Ytr_t = to_t(xn(Xtr)), to_t(yn(Ytr))
    Xva_t, Yva_t = to_t(xn(Xva)), to_t(yn(Yva))

    dl = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=args.batch, shuffle=True)

    model = MLPPredictor(history=args.history).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = torch.nn.MSELoss()

    # naive baseline: predict zero delta (leaf frozen), in normalized space
    base_val = torch.mean(Yva_t ** 2).item()

    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), Yva_t).item()
        tag = ""
        if vloss < best:
            best = vloss
            torch.save({
                "state_dict": model.state_dict(),
                "history": args.history,
                "x_norm": xn.as_dict(),
                "y_norm": yn.as_dict(),
            }, args.out)
            tag = "  <-- saved"
        print(f"epoch {ep+1:3d}  train {tot/len(Xtr):.4f}  "
              f"val {vloss:.4f}  (frozen-leaf baseline {base_val:.4f}){tag}")

    print(f"\nbest val {best:.4f} vs frozen-leaf {base_val:.4f}  "
          f"=> {base_val/best:.1f}x better than predicting no motion")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
