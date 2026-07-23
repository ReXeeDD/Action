"""
train_memory.py — the STREAMING memory model, trained all the way to landing.

The encoder reads ALL of the motion observed so far (up to a cap), accumulating
evidence about the hidden environment as the object moves — "look at everything that
happened, then predict the rest." The decoder then rolls the whole remaining future
out, and the loss is over that entire horizon, so long-range accuracy is what we
actually optimize.

Two decoders are available (`--arch`):

  gru      the original autoregressive rollout. Integrates one step at a time.
  direct   the FLOW-MAP decoder (default). Predicts every horizon in parallel.
           Far faster to train (sequential depth 240 -> 1) and cannot compound
           error, because horizon 240 is supervised directly rather than through
           239 earlier predictions. See models.DirectTrajPredictor.

Two things make the loss trustworthy across a mixed-world dataset:

  * **Scale-normalized position loss.** Raw metres let the big, fast worlds own the
    gradient: a ball travelling 9 m produces ~500x the squared error of a pendulum
    swinging 0.4 m, so pendulums were effectively ignored even at 29% of the data —
    and the chaotic worlds, which have the largest and *least reducible* error of
    all, dominated hardest. Dividing each window's error by its own motion scale
    makes every world contribute comparably. Disable with --no-scale-norm.
  * **Per-world validation.** One blended RMSE across nine worlds with wildly
    different irreducible error is uninterpretable — it cannot tell "the model
    stopped learning" from "the chaotic worlds hit their physical floor".

    python -m action.train_memory --data data/all --out runs/general3.pt \
        --epochs 15 --batch 768 --fut-cap 240 --device cuda

    python -m action.train_memory --measure runs/general3.pt --data data/all
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from action.dataset import load_episodes, load_episodes_tagged
from action.entities import yaw_rotate
from action.models import SeqPredictor, DirectTrajPredictor

DT = 0.008


class StreamWindows(Dataset):
    """Growing-history windows. For a start point t: history = all frames up to t
    (capped), future = the rest of the episode up to t+fut_cap."""

    def __init__(self, episodes, hist_cap=160, fut_cap=240, min_hist=12, stride=3,
                 world_ids=None):
        self.eps = [torch.from_numpy(e).float() for e in episodes]
        self.hist_cap, self.fut_cap, self.min_hist = hist_cap, fut_cap, min_hist
        self.wid = world_ids if world_ids is not None else [0] * len(episodes)
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
        return hist, cur, anchor, fut, self.wid[i]


def collate(batch):
    """Right-pad variable-length histories and futures; build masks."""
    hists, curs, anchors, futs, wids = zip(*batch)
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
            fut_pad, hist_mask, fut_valid, torch.tensor(wids, dtype=torch.long))


def _fit_norms(ds, fut_cap, n=6000, yaw_aug=False):
    """Feature/delta normalizers plus the per-horizon displacement scale table.

    `horizon_std[k]` is the typical size of the displacement k steps ahead. The
    flow-map decoder predicts displacement in units of it, so its output stays O(1)
    whether it is asked for 1 step or 240 — without that, the same head would have to
    emit millimetres and metres from the same weights.

    `yaw_aug` MUST match training. The statistics have to describe the distribution the
    model actually sees, and augmentation changes it in a way that is easy to miss: a
    pendulum hinges about a horizontal axis, so its quaternion-z and angular-velocity-z
    are *exactly* zero in every frame and their fitted std is 0 (clamped to 1e-6).
    Un-augmented that is harmless — the numerator is zero too, so 0/1e-6 = 0. Rotate
    about the vertical axis and quaternion-z becomes nonzero, so the same feature
    normalizes to ~5e5, overflows bf16, and every loss goes NaN from the first step.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    feats, deltas = [], []
    hsum = torch.zeros(fut_cap, ds[0][0].size(1))
    hcnt = torch.zeros(fut_cap, 1)
    for k in idx:
        hist, cur, anchor, fut, _ = ds[k]
        if yaw_aug:
            th = float(rng.uniform(0, 2 * np.pi))
            hist, cur, fut = (yaw_rotate(hist, th), yaw_rotate(cur, th),
                              yaw_rotate(fut, th))
        feats.append(hist)
        seq = torch.cat([cur.unsqueeze(0), fut], dim=0)
        deltas.append(seq[1:] - seq[:-1])
        disp = fut - cur.unsqueeze(0)                    # (T,13) displacement from cur
        T = disp.size(0)
        hsum[:T] += disp ** 2
        hcnt[:T] += 1
    feats, deltas = torch.cat(feats, 0), torch.cat(deltas, 0)
    horizon_std = torch.sqrt(hsum / hcnt.clamp_min(1)).clamp_min(1e-4)
    # horizons sampled by too few windows inherit the last well-estimated row
    good = int((hcnt.squeeze(-1) >= 20).sum().item())
    if 0 < good < fut_cap:
        horizon_std[good:] = horizon_std[good - 1]
    return (feats.mean(0), feats.std(0).clamp_min(1e-6),
            deltas.mean(0), deltas.std(0).clamp_min(1e-6), horizon_std)


def build_net(arch, args, dev):
    if arch == "direct":
        return DirectTrajPredictor(context_dim=args.ctx, hidden=args.hidden,
                                   blocks=args.blocks, d_model=args.dmodel,
                                   enc_layers=args.enc_layers,
                                   max_horizon=max(args.fut_cap, 320),
                                   probabilistic=args.prob).to(dev)
    return SeqPredictor(context_dim=args.ctx, dec_hidden=args.dec_hidden,
                        d_model=args.dmodel, enc_layers=args.enc_layers).to(dev)


def train(args):
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    eps, names = load_episodes_tagged(args.data)
    worlds = sorted(set(names))
    w2i = {w: i for i, w in enumerate(worlds)}
    wids = [w2i[n] for n in names]
    print(f"worlds: " + ", ".join(f"{w}({names.count(w)})" for w in worlds), flush=True)

    perm = np.random.default_rng(args.seed).permutation(len(eps))
    n_val = max(1, int(0.1 * len(eps)))
    vi, ti = perm[:n_val], perm[n_val:]
    tr = StreamWindows([eps[i] for i in ti], args.hist_cap, args.fut_cap,
                       stride=args.stride, world_ids=[wids[i] for i in ti])
    va = StreamWindows([eps[i] for i in vi], args.hist_cap, args.fut_cap,
                       stride=20, world_ids=[wids[i] for i in vi])
    fm, fs, dm, dsd, hstd = _fit_norms(tr, args.fut_cap, yaw_aug=args.yaw_aug)
    fm, fs, dm, dsd, hstd = (x.to(dev) for x in (fm, fs, dm, dsd, hstd))
    print(f"train windows {len(tr)}  val windows {len(va)}  arch={args.arch}  "
          f"hist_cap={args.hist_cap} fut_cap={args.fut_cap} device={dev}", flush=True)

    net = build_net(args.arch, args, dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # AMP: 'auto' turns it on for the flow-map decoder (a wide parallel MLP, which is
    # exactly what tensor cores are for) and leaves it off for the GRU, whose 240-step
    # autoregressive chain can drift in fp16.
    use_amp = (dev.startswith("cuda") and args.amp != "off" and
               (args.amp != "auto" or args.arch == "direct"))
    # Pick bf16 ONLY on Ampere or newer (compute capability >= 8). Do NOT use
    # torch.cuda.is_bf16_supported(): it returns True on a Turing T4 because it counts
    # *software emulation*, and sm_75 has no bf16 tensor cores at all — autocasting to
    # bf16 there falls off the fast path entirely and ran at ~2% of the card's peak
    # (1030 s/epoch). fp16 is the correct choice on Turing and uses the tensor cores.
    cap = torch.cuda.get_device_capability()[0] if dev.startswith("cuda") else 0
    bf16 = use_amp and (cap >= 8 if args.amp in ("auto", "on") else args.amp == "bf16")
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not bf16)
    if dev.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"gpu={torch.cuda.get_device_name(0)} sm_{cap}x  "
              f"AMP={'bf16' if bf16 else ('fp16' if use_amp else 'off')}", flush=True)

    nw = args.workers if sys.platform != "win32" else 0   # spawn would copy all episodes
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                    collate_fn=collate, num_workers=nw, pin_memory=dev.startswith("cuda"),
                    persistent_workers=nw > 0)
    vdl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate,
                     num_workers=nw, pin_memory=dev.startswith("cuda"),
                     persistent_workers=nw > 0)
    print(f"{n_par/1e6:.2f}M params  batches/epoch {len(dl)}  dataloader workers {nw}",
          flush=True)

    def losses(batch, train_mode, horizon):
        hist, cur, anchor, fut, hmask, fvalid, wid = (
            t.to(dev, non_blocking=True) for t in batch)
        if horizon < fut.size(1):                       # curriculum: shorter rollouts
            fut, fvalid = fut[:, :horizon], fvalid[:, :horizon]

        if train_mode and args.yaw_aug:
            # Gravity singles out -z, so every world here is exactly invariant to
            # rotation about the vertical axis. The model has no way to know that --
            # it sees absolute quaternions and world-frame velocities -- so we hand it
            # the symmetry as free, exactly-correct extra data. Verified against MuJoCo:
            # re-simulating a yaw-rotated initial condition reproduces the rotated
            # trajectory to 1e-7 (float32 precision) on `ball`.
            #
            # The leaf is a subtler case worth recording: its wind is a fixed
            # world-frame vector, so an individual leaf episode is NOT yaw-symmetric.
            # But the wind is drawn as rng.normal(0, 0.7, size=3) -- isotropic in x/y --
            # so a rotated leaf trajectory is exactly what a rotated wind would have
            # produced, at identical probability density. The augmentation therefore
            # still preserves the data distribution, which is what actually matters.
            th = torch.rand(hist.size(0), device=dev, dtype=hist.dtype) * (2 * math.pi)
            hist = yaw_rotate(hist, th[:, None])        # same angle across the window
            fut = yaw_rotate(fut, th[:, None])
            cur = yaw_rotate(cur, th)
            c, s = torch.cos(th), torch.sin(th)
            anchor = torch.stack([c * anchor[:, 0] - s * anchor[:, 1],
                                  s * anchor[:, 0] + c * anchor[:, 1],
                                  anchor[:, 2]], dim=-1)

        hist_n = (hist - fm) / fs
        if train_mode and args.noise > 0:               # input-noise augmentation
            hist_n = hist_n + args.noise * torch.randn_like(hist_n)
        n_steps = fut.size(1)
        z = net.encode(hist_n, key_padding_mask=hmask)

        log_std = None
        if args.arch == "direct":
            pred_n, states, log_std = net.decode(z, cur, anchor, n_steps, fm, fs, hstd)
            idx = torch.clamp(torch.arange(n_steps, device=dev), max=hstd.size(0) - 1)
            tgt_n = (fut - cur.unsqueeze(1)) / hstd.index_select(0, idx).unsqueeze(0)
        else:
            pred_n, states = net.decode(z, cur, anchor, n_steps, fm, fs, dm, dsd,
                                        ckpt_chunk=args.ckpt_chunk)
            seq = torch.cat([cur.unsqueeze(1), fut], dim=1)
            tgt_n = ((seq[:, 1:] - seq[:, :-1]) - dm) / dsd

        # Reductions in fp32. Under fp16 autocast the scale-normalized term divides by
        # as little as min_scale^2 (=0.0025), a 400x amplification that can overflow
        # fp16's 65504 ceiling; the network stays half-precision, only the loss is
        # promoted, which costs nothing measurable.
        pred_n, states = pred_n.float(), states.float()
        tgt_n, fut, cur = tgt_n.float(), fut.float(), cur.float()
        if log_std is not None:
            log_std = log_std.float()

        denom = fvalid.sum().clamp_min(1.0)
        if log_std is None:
            shape_loss = (((pred_n - tgt_n) ** 2).sum(-1) * fvalid).sum() / denom
        else:
            # Gaussian NLL in normalized-displacement space. This is what makes the
            # spread MEAN something: the model is rewarded for widening exactly where
            # it is genuinely uncertain and punished for widening where it is not, so
            # the cone becomes calibrated rather than merely wide. The squared-error
            # term below is kept so the MEAN stays metrically accurate — pure NLL can
            # buy a better likelihood by inflating sigma instead of sharpening mu.
            inv_var = torch.exp(-2.0 * log_std)
            nll = (0.5 * ((tgt_n - pred_n) ** 2) * inv_var + log_std).sum(-1)
            shape_loss = (nll * fvalid).sum() / denom
        pos_se = ((states[..., 0:3] - fut[..., 0:3]) ** 2).sum(-1)       # (B,T) m^2
        abs_pos = (pos_se * fvalid).sum() / denom                        # metres^2
        if args.scale_norm:
            # each window's own motion scale: RMS true displacement from `cur`
            disp2 = ((fut[..., 0:3] - cur[:, None, 0:3]) ** 2).sum(-1)
            sc2 = ((disp2 * fvalid).sum(1) / fvalid.sum(1).clamp_min(1))
            sc2 = sc2.clamp_min(args.min_scale ** 2)                     # (B,)
            pos_term = ((pos_se / sc2[:, None]) * fvalid).sum() / denom
        else:
            pos_term = abs_pos
        loss = shape_loss + args.pos_weight * pos_term
        return loss, abs_pos, pos_se.detach(), fvalid, wid

    def save():
        torch.save({"state_dict": net.state_dict(), "arch": args.arch,
                    "probabilistic": args.prob,
                    "hist_cap": args.hist_cap, "ctx": args.ctx,
                    "dec_hidden": args.dec_hidden, "hidden": args.hidden,
                    "blocks": args.blocks, "dmodel": args.dmodel,
                    "enc_layers": args.enc_layers,
                    "n_time_freq": getattr(net, "n_time_freq", 12),
                    "max_horizon": max(args.fut_cap, 320),
                    "worlds": worlds,
                    "feat_mean": fm.cpu().numpy(), "feat_std": fs.cpu().numpy(),
                    "delta_mean": dm.cpu().numpy(), "delta_std": dsd.cpu().numpy(),
                    "horizon_std": hstd.cpu().numpy()}, args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def horizon_for(ep):
        """Curriculum: short rollouts first. Cheaper AND better conditioned early —
        the encoder learns to identify the environment before it is asked to hold a
        prediction together for two seconds."""
        if args.curriculum >= 1.0:
            return args.fut_cap
        ramp = min(1.0, ep / max(1.0, 0.6 * args.epochs))
        return max(16, int(round((args.curriculum + (1 - args.curriculum) * ramp)
                                 * args.fut_cap)))

    # ---- memory probe -------------------------------------------------------
    # The curriculum grows the rollout horizon every epoch, and the flow-map
    # decoder's activations are B x H x hidden — linear in H. So a batch that fits
    # comfortably at H=84 can OOM at H=240 *four epochs later*, after 8 minutes of
    # apparently healthy training. Run one forward+backward at the FULL horizon now,
    # so an oversized batch fails in seconds with an actionable message.
    if dev.startswith("cuda"):
        net.train()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                probe_loss, *_ = losses(next(iter(dl)), True, args.fut_cap)
            scaler.scale(probe_loss).backward()
        except torch.OutOfMemoryError:
            free, total = torch.cuda.mem_get_info()
            raise SystemExit(
                f"\nOOM during the memory probe at the FULL horizon H={args.fut_cap}.\n"
                f"  Peak memory scales with batch x horizon. --batch {args.batch} does "
                f"not fit at H={args.fut_cap}\n"
                f"  even though it would fit at the curriculum's first horizon "
                f"H={horizon_for(0)}.\n"
                f"  Retry with --batch {max(64, int(args.batch * 0.45))} "
                f"(or lower --fut-cap / --hidden).\n"
                f"  GPU: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free.\n")
        finally:
            opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        hit = next((e + 1 for e in range(args.epochs)
                    if horizon_for(e) >= args.fut_cap), None)
        when = f"the curriculum reaches it at epoch {hit}" if hit else \
               "the curriculum never reaches it in this run"
        print(f"memory probe at H={args.fut_cap}: OK, peak {peak:.1f} GiB "
              f"(the WORST case; {when})", flush=True)

    best = float("inf")
    for ep in range(args.epochs):
        H = horizon_for(ep)
        net.train()
        t0, tot = time.time(), []
        for batch in dl:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, _, _, _, _ = losses(batch, True, H)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot.append(loss.item())
        train_s = time.time() - t0

        net.eval()
        wsse = torch.zeros(len(worlds), device=dev)
        wcnt = torch.zeros(len(worlds), device=dev)
        with torch.no_grad():
            for batch in vdl:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _, _, pos_se, fvalid, wid = losses(batch, False, args.fut_cap)
                per = (pos_se.float() * fvalid).sum(1)                # (B,) m^2 summed
                cnt = fvalid.sum(1)
                wsse.index_add_(0, wid, per); wcnt.index_add_(0, wid, cnt)
        rmse = torch.sqrt(wsse / wcnt.clamp_min(1)).cpu().numpy()      # metres, per world
        # the checkpoint metric is the MEAN OF PER-WORLD RMSE, so a world is not
        # allowed to be ignored just because it moves in small numbers
        score = float(np.mean([r for r in rmse if np.isfinite(r)]))
        sched.step()
        tag = ""
        if score < best:
            best = score; save(); tag = "  <-- saved"
        per_world = "  ".join(f"{worlds[i][:9]} {rmse[i]*100:5.1f}" for i in range(len(worlds)))
        print(f"epoch {ep+1:3d} [H={H:3d} {train_s:5.1f}s]  train {np.mean(tot):8.4f}  "
              f"mean-world rmse {score*100:5.1f}cm{tag}\n            {per_world}", flush=True)
    print(f"\nbest mean-world rmse {best*100:.1f}cm  -> {args.out}", flush=True)

    print("\n" + "=" * 60 + "\nKEY RESULTS (share these)\n" + "=" * 60, flush=True)
    _diagnostics(args.out, args.data, dev="cpu")


# ----------------------------------------------------------------------------
class LoadedMemory:
    """A trained memory model plus everything needed to run it.

    Wraps both architectures behind one `predict` / `predict_batch` API so callers
    (live.py, diagnose_worlds.py) never branch on `arch`.
    """

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.arch = ck.get("arch", "gru")            # older checkpoints predate --arch
        self.device = device
        if self.arch == "direct":
            self.net = DirectTrajPredictor(
                context_dim=ck["ctx"], hidden=ck.get("hidden", 512),
                blocks=ck.get("blocks", 4), d_model=ck.get("dmodel", 96),
                enc_layers=ck.get("enc_layers", 3),
                max_horizon=ck.get("max_horizon", 320),
                probabilistic=ck.get("probabilistic", False)).to(device)
        else:
            self.net = SeqPredictor(
                context_dim=ck["ctx"], n_time_freq=ck.get("n_time_freq", 8),
                dec_hidden=ck.get("dec_hidden", 128), d_model=ck.get("dmodel", 64),
                enc_layers=ck.get("enc_layers", 2)).to(device)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        t = lambda k: torch.tensor(ck[k], device=device)
        self.hist_cap = ck["hist_cap"]
        self.fm, self.fs, self.dm, self.dsd = t("feat_mean"), t("feat_std"), \
            t("delta_mean"), t("delta_std")
        self.hstd = t("horizon_std") if "horizon_std" in ck else None
        self.worlds = ck.get("worlds", [])
        self.probabilistic = bool(ck.get("probabilistic", False))

    def _decode(self, z, cur, anchor, n_steps):
        if self.arch == "direct":
            return self.net.decode(z, cur, anchor, n_steps, self.fm, self.fs, self.hstd)[1]
        return self.net.decode(z, cur, anchor, n_steps, self.fm, self.fs,
                               self.dm, self.dsd)[1]

    @torch.no_grad()
    def predict_dist(self, history, cur, n_steps):
        """Mean trajectory plus the per-horizon standard deviation, both in absolute
        units. Returns (mean (T,13), sigma (T,13)); sigma is None for a deterministic
        checkpoint."""
        dev = self.device
        h = np.asarray(history[-self.hist_cap:], dtype=np.float32)
        curs = torch.from_numpy(np.asarray(cur, dtype=np.float32)).unsqueeze(0).to(dev)
        anchor = curs[:, 0:3]
        t = torch.from_numpy(h).to(dev).clone()
        t[:, 0:3] = t[:, 0:3] - anchor[0]
        z = self.net.encode(((t - self.fm) / self.fs).unsqueeze(0))
        if self.arch != "direct":
            _, st = self.net.decode(z, curs, anchor, n_steps, self.fm, self.fs,
                                    self.dm, self.dsd)
            return st[0].cpu().numpy(), None
        _, st, logs = self.net.decode(z, curs, anchor, n_steps, self.fm, self.fs,
                                      self.hstd)
        if logs is None:
            return st[0].cpu().numpy(), None
        idx = torch.clamp(torch.arange(n_steps, device=dev), max=self.hstd.size(0) - 1)
        scale = self.hstd.index_select(0, idx)                     # (T,13) absolute units
        return st[0].cpu().numpy(), (logs[0].exp() * scale).cpu().numpy()

    @torch.no_grad()
    def sample_futures(self, history, cur, n_steps, n_samples=64, temperature=1.0,
                       mode="persistent", ground_z=None, rng=None):
        """A CONE of futures: (n_samples, n_steps, 13).

        The shape of the noise matters more than its size. Two options:

        `persistent` (default) — draw ONE standard normal per sample and hold it fixed
            across the whole horizon, scaling by the predicted per-horizon sigma. Each
            sample is then a smooth trajectory that diverges steadily from the mean.
            This is the physically right structure here: what the model is uncertain
            about is the *environment* it inferred by watching — a set of constants
            fixed for the episode — and uncertainty in a constant propagates forward
            coherently. It is also what makes a cone look like a cone.

        `independent` — fresh noise at every horizon. Marginally identical, but each
            sample is a jittery path that no physical object could follow. Useful only
            as a contrast; it is what naive per-step sampling gives you.

        `ground_z` applies the physical guard from the old ensemble cone: once a sample
        reaches the ground it stays there, instead of sinking through the floor.
        """
        mean, sigma = self.predict_dist(history, cur, n_steps)
        if sigma is None:
            return mean[None].repeat(n_samples, axis=0)
        rng = rng or np.random.default_rng()
        shape = ((n_samples, 1, mean.shape[-1]) if mode == "persistent"
                 else (n_samples, n_steps, mean.shape[-1]))
        eps = rng.standard_normal(shape).astype(np.float32)
        out = mean[None] + eps * (sigma[None] * temperature)
        q = out[..., 3:7]
        out[..., 3:7] = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        if ground_z is not None:
            for s in range(out.shape[0]):
                below = np.where(out[s, :, 2] <= ground_z)[0]
                if len(below):
                    k = int(below[0])
                    out[s, k:, 0:3] = out[s, k, 0:3]       # freeze where it landed
                    out[s, k:, 7:13] = 0.0
        return out

    @torch.no_grad()
    def predict(self, history, cur, n_steps):
        """history: (Lh,13) absolute states observed so far. cur: (13,). -> (n,13)."""
        return self.predict_batch([history], np.asarray(cur)[None], n_steps)[0]

    @torch.no_grad()
    def predict_batch(self, histories, curs, n_steps):
        """Predict from MANY observation points at once -> (B, n_steps, 13).

        Every prediction is independent of the others, so they batch perfectly. This
        is what makes rendering fast: one padded encode + one decode replaces B
        separate forward passes, collapsing the wall-clock cost of a whole video from
        B sequential rollouts to one.
        """
        dev = self.device
        hs = [np.asarray(h[-self.hist_cap:], dtype=np.float32) for h in histories]
        B, L = len(hs), max(len(h) for h in hs)
        curs = torch.from_numpy(np.asarray(curs, dtype=np.float32)).to(dev)
        anchor = curs[:, 0:3]
        pad = torch.zeros(B, L, hs[0].shape[1], device=dev)
        mask = torch.ones(B, L, dtype=torch.bool, device=dev)
        for b, h in enumerate(hs):
            # .clone() is required: torch.from_numpy SHARES memory with the numpy
            # array, and on CPU .to("cpu") is a no-op, so subtracting the anchor
            # in place would silently rewrite the caller's trajectory.
            t = torch.from_numpy(h).to(dev).clone()
            t[:, 0:3] = t[:, 0:3] - anchor[b]
            pad[b, :len(h)] = t
            mask[b, :len(h)] = False
        z = self.net.encode((pad - self.fm) / self.fs, key_padding_mask=mask)
        return self._decode(z, curs, anchor, n_steps).cpu().numpy()


def load_mem(ckpt_path, device="cpu") -> LoadedMemory:
    return LoadedMemory(ckpt_path, device)


def mem_predict(mem, history, cur, n_steps):
    return mem.predict(history, cur, n_steps)


def _diagnostics(ckpt, data, dev="cpu"):
    mem = load_mem(ckpt, dev)
    eps = load_episodes(data)
    # evaluate on the tail of the dataset (a hardcoded eps[1400:1470] silently
    # produced an EMPTY set — and a crash — on any dataset smaller than that)
    ev = eps[-70:] if len(eps) > 70 else eps

    print("prediction error vs horizon (watched only first 12 frames):", flush=True)
    errs = []
    for traj in ev:
        if len(traj) < 40:
            continue
        t0 = 12
        pred = mem.predict(traj[:t0], traj[t0 - 1], len(traj) - t0)
        errs.append(np.linalg.norm(pred[:, 0:3] - traj[t0:, 0:3], axis=1))
    maxlen = max(len(e) for e in errs)
    avg = np.array([np.mean([e[i] for e in errs if len(e) > i]) for i in range(maxlen)])
    for sec in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        i = int(sec / DT)
        if i < len(avg):
            print(f"  {sec:4.2f}s: err {avg[i]*100:6.1f} cm", flush=True)

    print("\nsharpens as it watches? (fixed 0.48s-ahead error, growing memory):", flush=True)
    H = 60
    for fr in [0.20, 0.35, 0.50, 0.65]:
        es = []
        for traj in ev:
            t0 = int(fr * len(traj))
            if t0 < 12 or t0 + H >= len(traj):
                continue
            pred = mem.predict(traj[:t0], traj[t0 - 1], H)
            es.append(np.linalg.norm(pred[:, 0:3] - traj[t0:t0 + H, 0:3], axis=1)[-1])
        if es:
            print(f"  watched {int(fr*100):2d}%: +0.48s err = {np.mean(es)*100:5.1f} cm",
                  flush=True)


def measure(args):
    print("=" * 60 + "\nKEY RESULTS\n" + "=" * 60, flush=True)
    _diagnostics(args.measure, args.data, dev="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/all")
    ap.add_argument("--out", type=str, default="runs/general.pt")
    ap.add_argument("--measure", type=str, default=None)
    ap.add_argument("--arch", type=str, default="direct", choices=["direct", "gru"])
    ap.add_argument("--hist-cap", type=int, default=160)
    ap.add_argument("--fut-cap", type=int, default=240)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--dmodel", type=int, default=96)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=512, help="direct: decoder width")
    ap.add_argument("--blocks", type=int, default=4, help="direct: residual blocks")
    ap.add_argument("--dec-hidden", type=int, default=256, help="gru: decoder width")
    ap.add_argument("--ckpt-chunk", type=int, default=24, help="gru: rollout checkpointing")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--scale-norm", dest="scale_norm", action="store_true", default=True)
    ap.add_argument("--no-scale-norm", dest="scale_norm", action="store_false")
    ap.add_argument("--min-scale", type=float, default=0.05,
                    help="floor on a window's motion scale (m), so near-still windows "
                         "cannot blow up the normalized loss")
    ap.add_argument("--prob", action="store_true",
                    help="probabilistic head: predict mean AND spread, train with "
                         "Gaussian NLL. Enables the cone (see action/cone_eval.py)")
    # Yaw augmentation is OFF by default, on evidence. The symmetry is real and the
    # transform is verified exact against MuJoCo (1e-7 on `ball`), but ablating it
    # same-seed showed no benefit and slower convergence: 17.9 cm vs 17.0 cm after 6
    # epochs. The reason is visible in audit_randomness.py -- every world already
    # samples heading isotropically (|R| = 0.006-0.273; pendulums randomize their swing
    # plane per episode, ball/object launch in random directions), so there is no
    # directional gap left to fill and the augmentation only adds redundant variety.
    # Kept because it should matter for a world whose data does NOT cover all headings.
    ap.add_argument("--yaw-aug", dest="yaw_aug", action="store_true", default=False,
                    help="random rotation about the vertical axis (exact symmetry; "
                         "measured to give no gain on the current worlds)")
    ap.add_argument("--no-yaw-aug", dest="yaw_aug", action="store_false")
    ap.add_argument("--curriculum", type=float, default=0.35,
                    help="start training at this fraction of fut-cap, ramp to 1.0 "
                         "(1.0 = off)")
    ap.add_argument("--amp", type=str, default="auto",
                    choices=["auto", "on", "off", "fp16", "bf16"],
                    help="auto = on for --arch direct, dtype chosen by GPU capability "
                         "(bf16 needs sm_80+; a T4 is sm_75 and must use fp16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    measure(args) if args.measure else train(args)


if __name__ == "__main__":
    main()
