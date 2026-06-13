#!/usr/bin/env python3
"""Spectral routing on a STRONGER backbone (STID), equal-parameter control.

Three variants on benchmark flow data (PEMS04/08), 12-in 12-out:
  stid     : single STID backbone                 (params P)
  dual     : two STID backbones, outputs summed    (params 2P, NO decomposition)
  spectral : two STID backbones, FFT learnable-mask split (low->main, high->pert),
             outputs summed                          (params 2P + 7 mask weights)

dual vs spectral is the clean ablation: identical parameter count, the ONLY
difference is whether the input is spectrally routed. If spectral > dual, the
gain is the routing, not the extra capacity.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

T_H, T_P = 12, 12


class STIDBackbone(nn.Module):
    def __init__(self, num_nodes, c_in=1, d=32, n_layers=3, dropout=0.15):
        super().__init__()
        self.ts_proj = nn.Linear(T_H * c_in, d)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d))
        self.tod_emb = nn.Parameter(torch.empty(288, d))
        self.dow_emb = nn.Parameter(torch.empty(7, d))
        for p in (self.node_emb, self.tod_emb, self.dow_emb):
            nn.init.xavier_uniform_(p)
        h = 4 * d
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h))
            for _ in range(n_layers)])
        self.head = nn.Linear(h, T_P)

    def forward(self, x, tod_idx, dow_idx):
        B, N, T, C = x.shape
        ts = self.ts_proj(x.reshape(B, N, T * C))
        node = self.node_emb.unsqueeze(0).expand(B, N, -1)
        tod = self.tod_emb[tod_idx].unsqueeze(1).expand(B, N, -1)
        dow = self.dow_emb[dow_idx].unsqueeze(1).expand(B, N, -1)
        z = torch.cat([ts, node, tod, dow], dim=-1)
        for blk in self.blocks:
            z = z + blk(z)
        return self.head(z)


class FFTSplit(nn.Module):
    """Learnable per-bin sigmoid mask, init as step at K (same as FourierDualNet)."""
    def __init__(self, T_h=T_H, K=3):
        super().__init__()
        n_bins = T_h // 2 + 1
        init = torch.zeros(n_bins)
        init[:K] = 2.0
        init[K:] = -2.0
        self.bin_logit = nn.Parameter(init)

    def forward(self, x):  # x (B,N,T,C) -> main, pert
        B, N, T, C = x.shape
        xp = x.permute(0, 1, 3, 2).reshape(B * N * C, T)
        Fx = torch.fft.rfft(xp, dim=-1)
        mask = torch.sigmoid(self.bin_logit).to(Fx.dtype)
        main = torch.fft.irfft(Fx * mask, n=T, dim=-1)
        pert = torch.fft.irfft(Fx * (1 - mask), n=T, dim=-1)
        main = main.reshape(B, N, C, T).permute(0, 1, 3, 2).contiguous()
        pert = pert.reshape(B, N, C, T).permute(0, 1, 3, 2).contiguous()
        return main, pert


class Model(nn.Module):
    def __init__(self, variant, num_nodes, c_in=1, d=32):
        super().__init__()
        self.variant = variant
        self.b1 = STIDBackbone(num_nodes, c_in, d)
        if variant in ("dual", "spectral"):
            self.b2 = STIDBackbone(num_nodes, c_in, d)
        if variant == "spectral":
            self.fft = FFTSplit()

    def forward(self, x, tod, dow):
        if self.variant == "stid":
            return self.b1(x, tod, dow)
        if self.variant == "dual":
            return self.b1(x, tod, dow) + self.b2(x, tod, dow)
        main, pert = self.fft(x)
        return self.b1(main, tod, dow) + self.b2(pert, tod, dow)


def load_series(dataset, data_dir):
    npz = np.load(data_dir / f"{dataset.upper()}.npz")
    series = npz["data"][..., 0].astype(np.float32)
    return series, (0.6, 0.2, 0.2)


def make_windows(T_total, splits):
    n_tr, n_va = int(T_total * splits[0]), int(T_total * splits[1])
    seg = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, T_total)}
    return {k: np.arange(lo, hi - (T_H + T_P) + 1, dtype=np.int64) for k, (lo, hi) in seg.items()}


def batches(series_n, series_raw, starts, bs, shuffle, rng, device):
    idx = rng.permutation(len(starts)) if shuffle else np.arange(len(starts))
    for s in range(0, len(idx), bs):
        st = starts[idx[s:s + bs]]
        x = np.stack([series_n[t:t + T_H] for t in st])
        y = np.stack([series_raw[t + T_H:t + T_H + T_P] for t in st])
        last = st + T_H - 1
        tod = torch.from_numpy((last % 288).astype(np.int64)).to(device)
        dow = torch.from_numpy(((last // 288) % 7).astype(np.int64)).to(device)
        xb = torch.from_numpy(x).to(device).permute(0, 2, 1).unsqueeze(-1)
        yb = torch.from_numpy(y).to(device).permute(0, 2, 1)
        yield xb, yb, tod, dow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["pems04", "pems08"])
    p.add_argument("--variant", required=True, choices=["stid", "dual", "spectral"])
    p.add_argument("--data_dir", type=Path, default=Path("data/benchmarks"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs/benchmarks_stid"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir / args.dataset / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series, splits = load_series(args.dataset, args.data_dir)
    T_total, N = series.shape
    wins = make_windows(T_total, splits)
    tr = wins["train"]
    tvals = series[tr[0]:tr[-1] + T_H]
    mu, sd = float(tvals.mean()), float(tvals.std() + 1e-6)
    series_n = ((series - mu) / sd).astype(np.float32)

    net = Model(args.variant, N, c_in=1, d=args.d).to(device)
    nparam = sum(q.numel() for q in net.parameters() if q.requires_grad)
    print(f"{args.dataset} {args.variant} N={N} params={nparam:,} "
          f"train/val/test={len(wins['train'])}/{len(wins['val'])}/{len(wins['test'])}", flush=True)

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nb = (len(tr) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def mae(pred, y):
        return (pred - y).abs().mean()

    def ev(split):
        net.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb, tod, dow in batches(series_n, series, wins[split], args.batch_size, False, rng, device):
                pred = net(xb, tod, dow) * sd + mu
                tot += float(mae(pred, yb).item()) * xb.size(0)
                n += xb.size(0)
        net.train()
        return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        for bi, (xb, yb, tod, dow) in enumerate(batches(series_n, series, tr, args.batch_size, True, rng, device)):
            loss = mae(net(xb, tod, dow) * sd + mu, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        v = ev("val")
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f}  val L={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            torch.save({"model_state": net.state_dict(), "config": vars(args), "mu": mu, "sd": sd},
                       out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    net.load_state_dict(st["model_state"])
    net.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb, tod, dow in batches(series_n, series, wins["test"], args.batch_size, False, rng, device):
            preds.append((net(xb, tod, dow) * sd + mu).cpu().numpy())
            ys.append(yb.cpu().numpy())
    pred, y = np.concatenate(preds), np.concatenate(ys)
    diff = np.abs(pred - y)
    res = {"all": float(diff.mean()),
           "h3": float(diff[:, :, 2].mean()), "h6": float(diff[:, :, 5].mean()),
           "h12": float(diff[:, :, 11].mean()), "params": nparam,
           "dataset": args.dataset, "variant": args.variant, "seed": args.seed}
    print(f"\ntest MAE: all={res['all']:.3f}  15min={res['h3']:.3f}  "
          f"30min={res['h6']:.3f}  60min={res['h12']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
