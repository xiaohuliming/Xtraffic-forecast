#!/usr/bin/env python3
"""Equal-param routing-vs-capacity ablation on the GWN backbone (the decisive test).

Three variants on benchmark flow data (PEMS04/08), single-channel input, 12-in 12-out:
  gwn      : single GWN backbone                              (params P)
  dual     : two GWN backbones, outputs summed (NO decomp)     (params 2P)
  spectral : two GWN backbones, FFT learnable-mask split       (params 2P + 7)

dual-vs-spectral is the clean ablation that the headline FDN-vs-GWN benchmark lacked:
identical params, the ONLY difference is spectral routing. If spectral > dual, routing
is real; if spectral ~= dual, FDN's benchmark win was pure capacity (2P vs P).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import gwnet

T_H, T_P = 12, 12


class FFTSplit(nn.Module):
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


def make_gwn(N, supports, device, nhid):
    return gwnet(device=device, num_nodes=N, dropout=0.3, supports=supports,
                 gcn_bool=True, addaptadj=True, in_dim=1, out_dim=T_P,
                 residual_channels=nhid, dilation_channels=nhid,
                 skip_channels=nhid * 8, end_channels=nhid * 16,
                 blocks=4, layers=2)


class Model(nn.Module):
    def __init__(self, variant, N, supports, device, nhid=32):
        super().__init__()
        self.variant = variant
        self.b1 = make_gwn(N, supports, device, nhid)
        if variant in ("dual", "spectral"):
            self.b2 = make_gwn(N, supports, device, nhid)
        if variant == "spectral":
            self.fft = FFTSplit()

    @staticmethod
    def _io(net, x):  # x (B,N,T,1) -> (B,N,T_p)
        return net(x.permute(0, 3, 1, 2).contiguous()).squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x):
        if self.variant == "gwn":
            return self._io(self.b1, x)
        if self.variant == "dual":
            return self._io(self.b1, x) + self._io(self.b2, x)
        main, pert = self.fft(x)
        return self._io(self.b1, main) + self._io(self.b2, pert)


def supports_from_adj(A, device):
    deg = A.sum(1); dinv = np.where(deg > 0, 1.0 / deg, 0.0)
    degT = A.T.sum(1); dinvT = np.where(degT > 0, 1.0 / degT, 0.0)
    return [torch.from_numpy(dinv[:, None] * A).to(device),
            torch.from_numpy(dinvT[:, None] * A.T).to(device)]


def load(dataset, data_dir):
    npz = np.load(data_dir / f"{dataset.upper()}.npz")
    series = npz["data"][..., 0].astype(np.float32)
    N = series.shape[1]
    A = np.zeros((N, N), dtype=np.float32)
    import csv
    with open(data_dir / f"{dataset.upper()}_distance.csv") as f:
        for r in csv.DictReader(f):
            i, j = int(r["from"]), int(r["to"]); A[i, j] = 1.0; A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return series, A


def windows(T_total):
    n_tr, n_va = int(T_total * 0.6), int(T_total * 0.2)
    seg = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, T_total)}
    return {k: np.arange(lo, hi - (T_H + T_P) + 1, dtype=np.int64) for k, (lo, hi) in seg.items()}


def batches(sn, sr, starts, bs, shuffle, rng, device):
    idx = rng.permutation(len(starts)) if shuffle else np.arange(len(starts))
    for s in range(0, len(idx), bs):
        st = starts[idx[s:s + bs]]
        x = np.stack([sn[t:t + T_H] for t in st])
        y = np.stack([sr[t + T_H:t + T_H + T_P] for t in st])
        xb = torch.from_numpy(x).to(device).permute(0, 2, 1).unsqueeze(-1)
        yb = torch.from_numpy(y).to(device).permute(0, 2, 1)
        yield xb, yb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["pems04", "pems08"])
    p.add_argument("--variant", required=True, choices=["gwn", "dual", "spectral"])
    p.add_argument("--data_dir", type=Path, default=Path("data/benchmarks"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs/benchmarks_gwnsp"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir / args.dataset / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series, A = load(args.dataset, args.data_dir)
    T_total, N = series.shape
    wins = windows(T_total)
    tr = wins["train"]
    tv = series[tr[0]:tr[-1] + T_H]
    mu, sd = float(tv.mean()), float(tv.std() + 1e-6)
    sn = ((series - mu) / sd).astype(np.float32)
    supports = supports_from_adj(A, device)
    net = Model(args.variant, N, supports, device, args.nhid).to(device)
    nparam = sum(q.numel() for q in net.parameters() if q.requires_grad)
    print(f"{args.dataset} {args.variant} N={N} params={nparam:,} "
          f"train/val/test={len(wins['train'])}/{len(wins['val'])}/{len(wins['test'])}", flush=True)

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nb = (len(tr) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def mae(pred, y):
        return (pred - y).abs().mean()

    def ev(split):
        net.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(sn, series, wins[split], args.batch_size, False, rng, device):
                tot += float(mae(net(xb) * sd + mu, yb).item()) * xb.size(0); n += xb.size(0)
        net.train(); return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for xb, yb in batches(sn, series, tr, args.batch_size, True, rng, device):
            loss = mae(net(xb) * sd + mu, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            s += float(loss.item()) * xb.size(0); n += xb.size(0)
        v = ev("val")
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f}  val L={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            torch.save({"model_state": net.state_dict(), "config": vars(args), "mu": mu, "sd": sd},
                       out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    net.load_state_dict(st["model_state"]); net.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in batches(sn, series, wins["test"], args.batch_size, False, rng, device):
            preds.append((net(xb) * sd + mu).cpu().numpy()); ys.append(yb.cpu().numpy())
    pred, y = np.concatenate(preds), np.concatenate(ys)
    diff = np.abs(pred - y)
    res = {"all": float(diff.mean()), "h3": float(diff[:, :, 2].mean()),
           "h6": float(diff[:, :, 5].mean()), "h12": float(diff[:, :, 11].mean()),
           "params": nparam, "dataset": args.dataset, "variant": args.variant, "seed": args.seed}
    print(f"\ntest MAE: all={res['all']:.3f}  15min={res['h3']:.3f}  "
          f"30min={res['h6']:.3f}  60min={res['h12']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
