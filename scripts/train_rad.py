#!/usr/bin/env python3
"""Train RAD-GMixer on one XTraffic region. Flow-only, label-free, GWN-free. Same masked-MAE /
npz schema as train_rgdn.py so numbers compare directly to v0b/v0c/GWN/STAEformer.

Variants:
  rad_v0  regime anchor + multi-scale temporal mixer + residual decoder (no graph)
  rad_v1  + local directed road graph mixer (upstream/downstream from Fwy/Abs PM/direction)
  rad_v2  + low-rank landmark global mixer
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, make_loader
from fourier_dual_net.radgmixer import RADGMixer

VARIANTS = {
    "rad_v0": dict(local_graph=False, landmark=False),
    "rad_v1": dict(local_graph=True,  landmark=False),
    "rad_v2": dict(local_graph=True,  landmark=True),
}


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def per_node_residual_std(rdata, hi, ch=0, floor=1e-6):
    flow = rdata.flow_series[:hi, :, ch]
    mask = rdata.flow_mask[:hi, :, ch].astype(bool)
    base = rdata.baseline_median[rdata.day_kind[:hi], rdata.tod[:hi]][:, :, ch]
    res = np.where(mask, flow - base, np.nan)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(res, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > floor), sd, 1.0)
    return sd.astype(np.float32)                      # (N,)


def build_road_graph(rdata, feat_names, K=8):
    """Directed local road adjacency from static_meta (Fwy id, Abs PM order, direction one-hot).
    A_up = K nearest lower-PM same-fwy-same-dir neighbors; A_down = higher-PM. Row-normalized."""
    sm = rdata.static_meta
    fi = {n: i for i, n in enumerate(feat_names)}
    N = sm.shape[0]
    fwy = sm[:, fi["Fwy"]] if "Fwy" in fi else np.zeros(N, np.float32)
    pm = sm[:, fi["Abs PM"]] if "Abs PM" in fi else np.arange(N, dtype=np.float32)
    dir_cols = [fi[c] for c in ("dir_N", "dir_E", "dir_S", "dir_W") if c in fi]
    direction = sm[:, dir_cols].argmax(1) if dir_cols else np.zeros(N, np.int64)
    A_up = np.zeros((N, N), np.float32)
    A_down = np.zeros((N, N), np.float32)
    # group by (fwy rounded, direction); order by pm
    key = np.stack([np.round(fwy, 4), direction], 1)
    uniq = {}
    for i in range(N):
        uniq.setdefault((key[i, 0], key[i, 1]), []).append(i)
    for members in uniq.values():
        idx = np.array(sorted(members, key=lambda j: pm[j]))
        for rank, n in enumerate(idx):
            for k in range(1, K + 1):
                if rank - k >= 0:
                    A_up[n, idx[rank - k]] = 1.0
                if rank + k < len(idx):
                    A_down[n, idx[rank + k]] = 1.0
    for A in (A_up, A_down):
        s = A.sum(1, keepdims=True)
        np.divide(A, s, out=A, where=s > 0)
    return A_up.astype(np.float32), A_down.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda", choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/rad"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--n_land", type=int, default=16)
    p.add_argument("--local_K", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} region {args.region} variant {args.variant} seed {args.seed}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
    sd_node = torch.from_numpy(per_node_residual_std(rdata, hi))
    feat_names = [s.decode() if isinstance(s, bytes) else s
                  for s in np.array(rdata.__dict__.get("meta_feature_names",
                  [b"Lat", b"Lng", b"Abs PM", b"Fwy", b"dir_N", b"dir_E", b"dir_S", b"dir_W"]))]
    # meta_feature_names is on the h5 attrs; read from file to be safe
    import h5py
    with h5py.File(Path(args.data_dir) / f"{args.region}_traffic.h5", "r") as f:
        feat_names = [s.decode() for s in f.attrs["meta_feature_names"]]
    A_up = A_down = None
    if VARIANTS[args.variant]["local_graph"]:
        au, ad = build_road_graph(rdata, feat_names, K=args.local_K)
        A_up = torch.from_numpy(au).to(device); A_down = torch.from_numpy(ad).to(device)
        print(f"road graph: A_up nnz={int((au > 0).sum())} A_down nnz={int((ad > 0).sum())}", flush=True)
    print(f"N={N} T_h={T_h} T_p={T_p} sd_node[min/med/max]="
          f"{float(sd_node.min()):.2f}/{float(sd_node.median()):.2f}/{float(sd_node.max()):.2f}", flush=True)

    def make_model(variant):
        m = RADGMixer(N, T_h, T_p, hidden=args.hidden, n_land=args.n_land, dropout=args.dropout,
                      **VARIANTS[variant]).to(device)
        m.sd_node.copy_(sd_node.to(device))
        return m

    def forward_batch(model, batch):
        return model(batch["x_hist"].to(device), batch["x_baseline"].to(device),
                     batch["y_baseline"].to(device), A_up, A_down)

    if args.smoke:
        loader = make_loader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        for v in VARIANTS:
            au = ad = None
            if VARIANTS[v]["local_graph"]:
                _au, _ad = build_road_graph(rdata, feat_names, K=args.local_K)
                au = torch.from_numpy(_au).to(device); ad = torch.from_numpy(_ad).to(device)
            m = RADGMixer(N, T_h, T_p, hidden=args.hidden, n_land=args.n_land, dropout=args.dropout,
                          **VARIANTS[v]).to(device)
            m.sd_node.copy_(sd_node.to(device))
            y = m(batch["x_hist"].to(device), batch["x_baseline"].to(device),
                  batch["y_baseline"].to(device), au, ad)
            loss = masked_mae(y, batch["y_true"][..., 0].to(device), batch["y_mask"][..., 0].to(device))
            loss.backward()
            npm = sum(q.numel() for q in m.parameters() if q.requires_grad)
            print(f"  {v} params={npm:,} out={tuple(y.shape)} loss={loss.item():.3f} "
                  f"finite={bool(torch.isfinite(y).all())}", flush=True)
        print("SMOKE_OK", flush=True)
        return

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    out_dir = args.out_dir / args.region / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.variant)
    nparam = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"variant {args.variant} params={nparam:,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed,
                               num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def ev(loader):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                y = forward_batch(model, batch)
                t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(y, t, msk).item()) * y.size(0); n += y.size(0)
        model.train(); return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for batch in train_loader:
            y = forward_batch(model, batch)
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
            torch.save({"model_state": model.state_dict(), "config": vars(args)}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    print(f"\nbest val {best:.4f}\n=== test ===", flush=True)
    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"]); model.eval()
    S = len(test_ds)
    pred_flow = np.empty((S, T_p, N), dtype=np.float32); actual_flow = np.empty((S, T_p, N), dtype=np.float32)
    y_mask_flow = np.empty((S, T_p, N), dtype=bool); affected = np.empty((S, N), dtype=bool)
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            y = forward_batch(model, batch).permute(0, 2, 1).cpu().numpy()
            bs = y.shape[0]
            pred_flow[cursor:cursor + bs] = y
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            cursor += bs

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
