#!/usr/bin/env python3
"""Train RGDN variants on one XTraffic region. Same pipeline / masked-MAE / npz schema
as train_staeformer_xtraffic.py so numbers compare directly to FDN/GWN/STAEformer.

Variants: v0a single GWN raw | v0b single GWN de-seasonalized | v1 RGDN |
v2 RGDN no-inject | v3 dual main-gcn-on | v4 RGDN no-deseason.
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

from dist_net.data import MultiRegionDataset, make_loader
from fourier_dual_net.rgdn import RGDN
from fourier_dual_net.deseason import train_residual_std

VARIANTS = {
    "v0a": dict(deseason=False, dual=False, main_gcn=False, inject=False),
    "v0b": dict(deseason=True,  dual=False, main_gcn=False, inject=False),
    "v1":  dict(deseason=True,  dual=True,  main_gcn=False, inject=True),
    "v2":  dict(deseason=True,  dual=True,  main_gcn=False, inject=False),
    "v3":  dict(deseason=True,  dual=True,  main_gcn=True,  inject=False),
    "v4":  dict(deseason=False, dual=True,  main_gcn=False, inject=True),
}


def build_adj_supports(edge_index, N, device):
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return [torch.from_numpy(deg_inv[:, None] * A).to(device),
            torch.from_numpy(deg_inv[:, None] * A.T).to(device)]


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def make_model(variant, N, supports, T_h, T_p, device, args):
    return RGDN(N, supports, T_h, T_p, device=device,
                nhid_single=args.nhid_single, nhid_main=args.nhid_main,
                nhid_res=args.nhid_res, c_inject=args.c_inject, dropout=args.dropout,
                **VARIANTS[variant]).to(device)


def train_stats(rdata, T_h, T_p):
    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
    seg, segm = flows[:hi], fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    sd_res = train_residual_std(rdata.flow_series, rdata.flow_mask,
                                rdata.baseline_median, rdata.day_kind, rdata.tod, hi, ch=0)
    return mu, sd, sd_res


def forward_batch(model, batch, device):
    x_hist = batch["x_hist"].to(device)
    x_base = batch["x_baseline"].to(device)
    y_base = batch["y_baseline"].to(device)
    tf = batch["time_feat"].to(device)
    return model(x_hist, x_base, y_base, tf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda", choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/rgdn"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--nhid_single", type=int, default=32)
    p.add_argument("--nhid_main", type=int, default=26)
    p.add_argument("--nhid_res", type=int, default=22)
    p.add_argument("--c_inject", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="build all variants, print params, 1 fwd/bwd, exit")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} region {args.region} variant {args.variant} seed {args.seed}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    supports = build_adj_supports(rdata.edge_index, N, device)
    mu, sd, sd_res = train_stats(rdata, T_h, T_p)
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f} sd_res={sd_res:.3f}", flush=True)

    if args.smoke:
        sample = make_loader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(sample))
        assert batch["x_baseline"].shape == batch["x_hist"].shape, "x_baseline shape mismatch"
        for v in VARIANTS:
            m = make_model(v, N, supports, T_h, T_p, device, args)
            m.sd_res.fill_(sd_res); m.flow_mu.fill_(mu); m.flow_sd.fill_(sd)
            nparam = sum(q.numel() for q in m.parameters() if q.requires_grad)
            y = forward_batch(m, batch, device)
            loss = masked_mae(y, batch["y_true"][..., 0].to(device), batch["y_mask"][..., 0].to(device))
            loss.backward()
            print(f"  {v:4s} params={nparam:,} out={tuple(y.shape)} loss={loss.item():.3f} "
                  f"finite={bool(torch.isfinite(y).all())}", flush=True)
        print("SMOKE_OK", flush=True)
        return

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    out_dir = args.out_dir / args.region / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.variant, N, supports, T_h, T_p, device, args)
    model.sd_res.fill_(sd_res); model.flow_mu.fill_(mu); model.flow_sd.fill_(sd)
    nparam = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"variant {args.variant} params={nparam:,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def ev(loader):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                y = forward_batch(model, batch, device)
                t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(y, t, msk).item()) * y.size(0); n += y.size(0)
        model.train(); return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for batch in train_loader:
            y = forward_batch(model, batch, device)
            t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
            loss = masked_mae(y, t, msk)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            s += float(loss.item()) * y.size(0); n += y.size(0)
        v = ev(val_loader)
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f} val L={v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "mu": mu, "sd": sd, "sd_res": sd_res}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    print(f"\nbest val {best:.4f}\n=== test ===", flush=True)
    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"]); model.eval()
    S = len(test_ds)
    pred_flow = np.empty((S, T_p, N), dtype=np.float32)
    actual_flow = np.empty((S, T_p, N), dtype=np.float32)
    y_mask_flow = np.empty((S, T_p, N), dtype=bool)
    affected = np.empty((S, N), dtype=bool)
    sample_start = np.empty((S,), dtype=np.int64)
    region_code = np.empty((S,), dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            y = forward_batch(model, batch, device).permute(0, 2, 1).cpu().numpy()   # (B,T_p,N)
            bs = y.shape[0]
            pred_flow[cursor:cursor + bs] = y
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            sample_start[cursor:cursor + bs] = batch["sample_start"].numpy()
            region_code[cursor:cursor + bs] = batch["region_code"].numpy()
            cursor += bs

    np.savez_compressed(out_dir / "test_predictions.npz",
                        region_code=region_code, sample_start=sample_start,
                        region_node_idx=rdata.region_idx.astype(np.int64),
                        pred_raw_flow=pred_flow, actual_future_flow=actual_flow,
                        y_mask_flow=y_mask_flow, affected_mask=affected)
    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best, "seed": args.seed, "variant": args.variant, "params": nparam}
    print(f"\ntest MAE all={res['all']:.3f} affected={res['affected']:.3f} "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
