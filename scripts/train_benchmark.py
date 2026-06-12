#!/usr/bin/env python3
"""FDN vs GWN on standard benchmarks (METR-LA / PEMS04 / PEMS08).

Equal-budget internal comparison: both models see the single data channel only
(use_time_emb=False, matching our XTraffic baseline FDN). 12-in 12-out, z-score
on train stats, masked MAE in raw units (METR-LA: 0 = missing; PEMS: all valid).
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import gwnet
from fourier_dual_net.model import FourierDualNet

T_H, T_P = 12, 12


def load_series(dataset: str, data_dir: Path):
    if dataset == "metrla":
        series = np.load(data_dir / "metr-la.npz")["data"].astype(np.float32)  # (T,N) speed, 0=missing
        import pickle
        with open(data_dir / "adj_mx_metrla.pkl", "rb") as f:
            A = pickle.load(f, encoding="latin1")[2].astype(np.float32)
        splits = (0.7, 0.1, 0.2)
    else:
        npz = np.load(data_dir / f"{dataset.upper()}.npz")
        series = npz["data"][..., 0].astype(np.float32)    # (T, N) flow
        N = series.shape[1]
        A = np.zeros((N, N), dtype=np.float32)
        import csv
        with open(data_dir / f"{dataset.upper()}_distance.csv") as f:
            for row in csv.DictReader(f):
                i, j = int(row["from"]), int(row["to"])
                A[i, j] = 1.0
                A[j, i] = 1.0
        np.fill_diagonal(A, 1.0)
        splits = (0.6, 0.2, 0.2)
    return series, A, splits


def supports_from_adj(A: np.ndarray, device):
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / deg, 0.0)
    fwd = dinv[:, None] * A
    degT = A.T.sum(axis=1)
    dinvT = np.where(degT > 0, 1.0 / degT, 0.0)
    bwd = dinvT[:, None] * A.T
    return [torch.from_numpy(fwd).to(device), torch.from_numpy(bwd).to(device)]


def make_windows(T_total: int, splits):
    n_train = int(T_total * splits[0])
    n_val = int(T_total * splits[1])
    seg = {"train": (0, n_train), "val": (n_train, n_train + n_val),
           "test": (n_train + n_val, T_total)}
    out = {}
    for k, (lo, hi) in seg.items():
        starts = np.arange(lo, hi - (T_H + T_P) + 1, dtype=np.int64)
        out[k] = starts
    return out


def batches(series_n, series_raw, starts, batch_size, shuffle, rng, device):
    idx = rng.permutation(len(starts)) if shuffle else np.arange(len(starts))
    for s in range(0, len(idx), batch_size):
        st = starts[idx[s:s + batch_size]]
        x = np.stack([series_n[t:t + T_H] for t in st])              # (B,T_h,N)
        y = np.stack([series_raw[t + T_H:t + T_H + T_P] for t in st])  # (B,T_p,N) raw
        xb = torch.from_numpy(x).to(device).permute(0, 2, 1).unsqueeze(-1)  # (B,N,T_h,1)
        yb = torch.from_numpy(y).to(device).permute(0, 2, 1)                # (B,N,T_p)
        yield xb, yb


def masked_mae_raw(pred_raw, y_raw, null_zero):
    mask = (y_raw > 1e-3) if null_zero else torch.ones_like(y_raw, dtype=torch.bool)
    m = mask.float()
    return ((pred_raw - y_raw).abs() * m).sum() / m.sum().clamp(min=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["metrla", "pems04", "pems08"])
    p.add_argument("--model", required=True, choices=["fdn", "gwn"])
    p.add_argument("--data_dir", type=Path, default=Path("data/benchmarks"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs/benchmarks"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir / args.dataset / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series, A, splits = load_series(args.dataset, args.data_dir)
    T_total, N = series.shape
    null_zero = args.dataset == "metrla"
    wins = make_windows(T_total, splits)
    tr = wins["train"]
    train_vals = series[tr[0]:tr[-1] + T_H]
    valid = train_vals[train_vals > 1e-3] if null_zero else train_vals
    mu, sd = float(valid.mean()), float(valid.std() + 1e-6)
    series_n = ((series - mu) / sd).astype(np.float32)
    print(f"{args.dataset} N={N} T={T_total} windows train/val/test="
          f"{len(wins['train'])}/{len(wins['val'])}/{len(wins['test'])}  mu={mu:.2f} sd={sd:.2f}",
          flush=True)

    supports = supports_from_adj(A, device)
    if args.model == "gwn":
        net = gwnet(device=device, num_nodes=N, dropout=args.dropout, supports=supports,
                    gcn_bool=True, addaptadj=True, in_dim=1, out_dim=T_P,
                    residual_channels=args.nhid, dilation_channels=args.nhid,
                    skip_channels=args.nhid * 8, end_channels=args.nhid * 16).to(device)

        def forward(xb):
            return net(xb.permute(0, 3, 1, 2).contiguous()).squeeze(-1).permute(0, 2, 1).contiguous()
    else:
        net = FourierDualNet(num_nodes=N, supports=supports, T_h=T_H, T_p=T_P,
                             K=args.K, decomp_mode="learnable", in_dim_flow=1,
                             nhid=args.nhid, dropout=args.dropout, device=device,
                             main_blocks=4, main_layers=2, pert_blocks=4, pert_layers=2,
                             use_time_emb=False).to(device)

        def forward(xb):
            return net(xb)

    print(f"model={args.model} params={sum(q.numel() for q in net.parameters() if q.requires_grad):,}",
          flush=True)
    opt = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches = (len(wins["train"]) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(n_batches, 1),
                              eta_min=args.lr * 1e-2)

    def run_eval(split):
        net.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(series_n, series, wins[split], args.batch_size, False, rng, device):
                pred = forward(xb) * sd + mu
                tot += float(masked_mae_raw(pred, yb, null_zero).item()) * xb.size(0)
                n += xb.size(0)
        net.train()
        return tot / max(n, 1)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        for bi, (xb, yb) in enumerate(batches(series_n, series, wins["train"],
                                              args.batch_size, True, rng, device)):
            pred = forward(xb) * sd + mu
            loss = masked_mae_raw(pred, yb, null_zero)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            if bi % 200 == 0:
                print(f"  ep{ep:02d} {bi}/{n_batches} L={loss.item():.3f}", flush=True)
        val = run_eval("val")
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f}  val L={val:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        if val < best_val:
            best_val = val
            torch.save({"model_state": net.state_dict(), "val": val, "config": vars(args),
                        "mu": mu, "sd": sd}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    best = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    net.load_state_dict(best["model_state"])
    net.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in batches(series_n, series, wins["test"], args.batch_size, False, rng, device):
            preds.append((forward(xb) * sd + mu).cpu().numpy())
            ys.append(yb.cpu().numpy())
    pred = np.concatenate(preds)                    # (S,N,T_p)
    y = np.concatenate(ys)
    mask = (y > 1e-3) if null_zero else np.ones_like(y, dtype=bool)
    diff = np.abs(pred - y)
    mae_all = float(diff[mask].mean())
    per_h = {f"h{h+1}": float(diff[:, :, h][mask[:, :, h]].mean()) for h in (2, 5, 11)}
    print(f"\ntest MAE: all={mae_all:.3f}  15min={per_h['h3']:.3f}  "
          f"30min={per_h['h6']:.3f}  60min={per_h['h12']:.3f}", flush=True)
    np.savez_compressed(out_dir / "test_predictions.npz", pred=pred, actual=y, mask=mask)
    (out_dir / "summary.json").write_text(json.dumps(
        {"all": mae_all, **per_h, "best_val": best_val, "dataset": args.dataset,
         "model": args.model, "seed": args.seed, "mu": mu, "sd": sd}, indent=2))


if __name__ == "__main__":
    main()
