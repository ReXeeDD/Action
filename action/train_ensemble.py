"""
train_ensemble.py — fit a deep ensemble of Gaussian predictors (the cone engine).

Two independent sources of uncertainty feed the cone:
  * aleatoric  — each model's Gaussian head says how noisy the next step is;
  * epistemic  — K models trained on bootstrapped data disagree, and in a chaotic
                 system that disagreement amplifies into a spreading spray of futures.

CPU by default (the net is tiny; on MuJoCo-class boxes CPU beats GPU transfer overhead).

    python -m action.train_ensemble --data data/leaf --members 5 --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from action.dataset import load_episodes, build_supervised, Normalizer
from action.models import GaussianMLP


def train_one(Xtr_t, Ytr_t, Xva_t, Yva_t, history, epochs, batch, lr, device, seed,
              noise=0.05):
    torch.manual_seed(seed)
    # bootstrap: each member sees a resampled view of the training set
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr_t)
    boot = torch.randint(0, n, (n,), generator=g)
    dl = DataLoader(TensorDataset(Xtr_t[boot], Ytr_t[boot]),
                    batch_size=batch, shuffle=True)

    model = GaussianMLP(history=history).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            # inject small input noise so the net learns to correct off-manifold
            # states it will meet during its own autoregressive rollout (stability)
            xb_in = xb + noise * torch.randn_like(xb) if noise > 0 else xb
            loss = model.nll(xb_in, yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vnll = model.nll(Xva_t, Yva_t).item()
        if vnll < best:
            best = vnll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_state, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/leaf")
    ap.add_argument("--out", type=str, default="runs/leaf_ensemble.pt")
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")   # CPU per project preference
    ap.add_argument("--noise", type=float, default=0.05)   # input-noise augmentation
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"device = {device}  threads = {torch.get_num_threads()}")

    eps = load_episodes(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]

    Xtr, Ytr = build_supervised(train_eps, args.history)
    Xva, Yva = build_supervised(val_eps, args.history)
    print(f"train samples={len(Xtr):,}  val samples={len(Xva):,}")

    xn = Normalizer.fit(Xtr)
    yn = Normalizer.fit(Ytr)
    Xtr_t = torch.from_numpy(xn(Xtr)).to(device)
    Ytr_t = torch.from_numpy(yn(Ytr)).to(device)
    Xva_t = torch.from_numpy(xn(Xva)).to(device)
    Yva_t = torch.from_numpy(yn(Yva)).to(device)

    states = []
    for m in range(args.members):
        state, vnll = train_one(Xtr_t, Ytr_t, Xva_t, Yva_t, args.history,
                                 args.epochs, args.batch, args.lr, device,
                                 seed=args.seed + 100 * (m + 1), noise=args.noise)
        states.append(state)
        print(f"  member {m+1}/{args.members}  best val NLL {vnll:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "members": states,
        "history": args.history,
        "x_norm": xn.as_dict(),
        "y_norm": yn.as_dict(),
    }, args.out)
    print(f"saved ensemble of {args.members} -> {args.out}")


if __name__ == "__main__":
    main()
