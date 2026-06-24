#!/usr/bin/env python3
"""STAEformer (ICCV'23 SOTA, label-free) on XTraffic — strong label-free baseline.

Faithful: imports the official model verbatim from baselines/STAEformer/STAEformer.py.
Same pipeline / masked-MAE / artifact schema as train_fourier_dual_net.py so the number
is directly comparable to FDN and (matched-window) IGSTGNN.

STAEformer wants x=(B, T_h, N, 3)=[flow_norm, tod/288, dow]; our batch provides x_hist
(B,N,T_h,3 raw flow/occ/speed) + time_feat (B,T_h,2)=[tod/288, dow/7]. We feed
[z(flow), tod/288, dow] (dow de-normalized to 0..6 since the model does dow.long()).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "STAEformer"))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, FullWindowRegionData, RegionData, make_loader
from STAEformer import STAEformer


def build_x(batch, mu, sd, device):
    """-> (B, T_h, N, 3) = [z(flow), tod/288, dow0..6]."""
    x_hist = batch["x_hist"].to(device)              # (B,N,T_h,3)
    tf = batch["time_feat"].to(device)               # (B,T_h,2) = tod/288, dow/7
    B, N, T_h, _ = x_hist.shape
    flow = (x_hist[..., 0] - mu) / sd                # (B,N,T_h)
    flow = flow.permute(0, 2, 1)                     # (B,T_h,N)
    tod = tf[:, :, 0].unsqueeze(-1).expand(B, T_h, N)        # already /288
    dow = (tf[:, :, 1] * 7.0).round().clamp(0, 6).unsqueeze(-1).expand(B, T_h, N)
    return torch.stack([flow, tod, dow], dim=-1)     # (B,T_h,N,3)


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/baselines/staeformer"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", choices=["event", "full_window"], default="event",
                   help="event = event-anchored windows (samples.h5); "
                        "full_window = standard sliding window over whole series")
    p.add_argument("--stride", type=int, default=1, help="full_window only: anchor stride")
    p.add_argument("--train_frac", type=float, default=0.7, help="full_window only")
    p.add_argument("--val_frac", type=float, default=0.1, help="full_window only")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop if val doesn't improve for N epochs (0 = off)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.protocol == "full_window":
        rd_cls = FullWindowRegionData
        rd_kw = {"stride": args.stride, "train_frac": args.train_frac, "val_frac": args.val_frac}
        out_base = args.out_dir.with_name(args.out_dir.name + "_fullwindow")
    else:
        rd_cls, rd_kw, out_base = RegionData, {}, args.out_dir
    out_dir = out_base / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  region: {args.region}  seed: {args.seed}  protocol: {args.protocol}",
          flush=True)

    def mk(split):
        return MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split=split,
                                  region_data_cls=rd_cls, region_data_kwargs=rd_kw)
    train_ds = mk("train")
    val_ds = mk("val")
    test_ds = mk("test")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)

    # flow z-score from train split (channel 0), masked by validity
    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
    seg = flows[:hi]; segm = fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f}", flush=True)

    model = STAEformer(num_nodes=N, in_steps=T_h, out_steps=T_p, steps_per_day=288,
                       input_dim=1, output_dim=1).to(device)
    print(f"STAEformer params: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}",
          flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def ev(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                x = build_x(batch, mu, sd, device)
                pred = model(x).squeeze(-1).permute(0, 2, 1) * sd + mu   # (B,N,T_p)
                y = batch["y_true"][..., 0].to(device)
                m = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(pred, y, m).item()) * x.size(0)
                n += x.size(0)
        model.train()
        return tot / max(n, 1)

    best = float("inf")
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        for bi, batch in enumerate(train_loader):
            x = build_x(batch, mu, sd, device)
            pred = model(x).squeeze(-1).permute(0, 2, 1) * sd + mu
            y = batch["y_true"][..., 0].to(device)
            m = batch["y_mask"][..., 0].to(device)
            loss = masked_mae(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * x.size(0)
            n += x.size(0)
            if bi % 300 == 0:
                print(f"  ep{ep:02d} batch {bi}/{nb} L={loss.item():.3f}", flush=True)
        v = ev(val_loader)
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f}  val L={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "mu": mu, "sd": sd, "N": N, "T_h": T_h, "T_p": T_p}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"    early stop: no val improvement for {args.patience} epochs", flush=True)
                break

    print(f"\nbest val: {best:.4f}\n=== test inference ===", flush=True)
    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"])
    model.eval()
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
            x = build_x(batch, mu, sd, device)
            pred = (model(x).squeeze(-1).permute(0, 2, 1) * sd + mu).permute(0, 2, 1).cpu().numpy()
            bs = pred.shape[0]
            pred_flow[cursor:cursor + bs] = pred
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
           "best_val": best, "seed": args.seed, "params": sum(q.numel() for q in model.parameters())}
    print(f"\ntest MAE: all={res['all']:.3f}  affected={res['affected']:.3f}  "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
