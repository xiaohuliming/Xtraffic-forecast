#!/usr/bin/env python3
"""Train STID (CIKM'22) per region — label-free MLP baseline, no graph.

Same data pipeline / loss / artifact schema as train_fourier_dual_net.py:
outputs <out_dir>/<tag>/<region>/{ckpt_best.pt, test_predictions.npz, summary.json}.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dist_net.data import MultiRegionDataset, make_loader  # noqa: E402


class STID(nn.Module):
    """Spatial-Temporal Identity: per-node MLP + node/ToD/DoW identity embeddings."""

    def __init__(self, num_nodes: int, T_h: int, T_p: int, c_in: int = 3,
                 d: int = 32, n_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.ts_proj = nn.Linear(T_h * c_in, d)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d))
        self.tod_emb = nn.Parameter(torch.empty(288, d))
        self.dow_emb = nn.Parameter(torch.empty(7, d))
        for p in (self.node_emb, self.tod_emb, self.dow_emb):
            nn.init.xavier_uniform_(p)
        h = 4 * d
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(h, T_p)

    def forward(self, x_hist: torch.Tensor, tod_idx: torch.Tensor, dow_idx: torch.Tensor):
        # x_hist (B, N, T_h, C); tod_idx/dow_idx (B,)
        B, N, T, C = x_hist.shape
        ts = self.ts_proj(x_hist.reshape(B, N, T * C))            # (B, N, d)
        node = self.node_emb.unsqueeze(0).expand(B, N, -1)        # (B, N, d)
        tod = self.tod_emb[tod_idx].unsqueeze(1).expand(B, N, -1)  # (B, N, d)
        dow = self.dow_emb[dow_idx].unsqueeze(1).expand(B, N, -1)  # (B, N, d)
        z = torch.cat([ts, node, tod, dow], dim=-1)               # (B, N, 4d)
        for blk in self.blocks:
            z = z + blk(z)
        return self.head(z)                                        # (B, N, T_p)


def masked_mae(pred, target, mask):
    mask = mask.float()
    num = (pred - target).abs() * mask
    return num.sum() / mask.sum().clamp(min=1.0)


def time_indices(time_feat: torch.Tensor):
    # time_feat (B, T_h, 2) = [tod/288, dow/7] per step; use last history step
    tod = (time_feat[:, -1, 0] * 288.0).round().long().clamp(0, 287)
    dow = (time_feat[:, -1, 1] * 7.0).round().long().clamp(0, 6)
    return tod, dow


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["x_hist"].to(device)
        y = batch["y_true"][..., 0].to(device)
        m = batch["y_mask"][..., 0].to(device)
        tod, dow = time_indices(batch["time_feat"].to(device))
        pred = model(x, tod, dow)
        total += float(masked_mae(pred, y, m).item()) * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/baselines/stid"))
    p.add_argument("--tag", default=None)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = (args.out_dir if args.tag is None else args.out_dir.parent / args.tag) / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  region: {args.region}  seed: {args.seed}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    print(f"N={N}  T_h={T_h}  T_p={T_p}  |train|={len(train_ds)}  |val|={len(val_ds)}  |test|={len(test_ds)}", flush=True)

    model = STID(N, T_h, T_p, c_in=3, d=args.d, n_layers=args.n_layers,
                 dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"STID params: {n_params:,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(n_batches, 1),
                              eta_min=args.lr * 1e-2)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        for bi, batch in enumerate(train_loader):
            x = batch["x_hist"].to(device)
            y = batch["y_true"][..., 0].to(device)
            m = batch["y_mask"][..., 0].to(device)
            tod, dow = time_indices(batch["time_feat"].to(device))
            loss = masked_mae(model(x, tod, dow), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * x.size(0)
            n += x.size(0)
            if bi % 200 == 0:
                print(f"  ep{epoch:02d} batch {bi}/{n_batches} L={loss.item():.3f}", flush=True)
        val = evaluate(model, val_loader, device)
        print(f"==> ep{epoch:02d} train L={s/max(n,1):.4f}  val L={val:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val < best_val:
            best_val = val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val": val, "config": vars(args), "N": N, "T_h": T_h, "T_p": T_p},
                       out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    print(f"\nbest val: {best_val:.4f}\n=== test inference ===", flush=True)
    best = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
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
            x = batch["x_hist"].to(device)
            tod, dow = time_indices(batch["time_feat"].to(device))
            pred = model(x, tod, dow).permute(0, 2, 1).cpu().numpy()
            bs = x.size(0)
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
    mae_all = float(diff[y_mask_flow].mean())
    mae_aff = float(diff[y_mask_flow & aff3].mean())
    mae_un = float(diff[y_mask_flow & ~aff3].mean())
    print(f"\ntest MAE: all={mae_all:.3f}  affected={mae_aff:.3f}  unaffected={mae_un:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"all": mae_all, "affected": mae_aff, "unaffected": mae_un,
         "best_val": best_val, "params": n_params, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
