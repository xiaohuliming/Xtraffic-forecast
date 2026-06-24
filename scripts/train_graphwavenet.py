#!/usr/bin/env python3
"""Train GraphWaveNet per region on our HDF5 cache.

GraphWaveNet does not support varying N across regions, so we train one
model per region and write per-region checkpoint + per-region test predictions
in the same .npz schema as scripts/run_distnet_test_inference.py.

Reads:
  outputs/dist_net/region_data/<region>_traffic.h5      historical traffic
  outputs/dist_net/region_data/<region>_samples.h5      event-anchored samples
  outputs/region_graphs/<region_key>_sparse_adj.npz     edge_index → adj matrix

Writes:
  outputs/baselines/graphwavenet/<region>/
    ckpt_best.pt
    test_predictions.npz   (same schema as DIST-Net)
    summary.json
    train.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines" / "GraphWaveNet"))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, FullWindowRegionData, RegionData, make_loader
from model import gwnet

REGION_TO_GRAPH_KEY = {
    "Alameda":     "alameda",
    "ContraCosta": "contra_costa",
    "Orange":      "orange",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=list(REGION_TO_GRAPH_KEY.keys()))
    p.add_argument("--data-dir", type=Path,
                   default=Path("outputs/dist_net/region_data"))
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/baselines/graphwavenet"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", choices=["event", "full_window"], default="event",
                   help="event = event-anchored (samples.h5); full_window = standard sliding window")
    p.add_argument("--stride", type=int, default=1, help="full_window only: anchor stride")
    p.add_argument("--train_frac", type=float, default=0.7, help="full_window only")
    p.add_argument("--val_frac", type=float, default=0.1, help="full_window only")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop if val doesn't improve for N epochs (0 = off)")
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_adj_supports(edge_index: np.ndarray, N: int, device: torch.device
                       ) -> list[torch.Tensor]:
    """Build (forward, backward) random-walk normalized adjacency from edge_index.
    GraphWaveNet's 'doubletransition' adjacency."""
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    A_fwd = deg_inv[:, None] * A
    A_bwd = (deg_inv[:, None] * A.T)
    return [
        torch.from_numpy(A_fwd).to(device),
        torch.from_numpy(A_bwd).to(device),
    ]


def to_gwnet_input(x_hist: torch.Tensor) -> torch.Tensor:
    """(B, N, T_h, C_x) → (B, C_x, N, T_h)."""
    return x_hist.permute(0, 3, 1, 2).contiguous()


def from_gwnet_output(out: torch.Tensor) -> torch.Tensor:
    """(B, T_p, N, 1) → (B, N, T_p)."""
    return out.squeeze(-1).permute(0, 2, 1).contiguous()


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid positions only.
    pred/target/mask all (B, N, T_p)."""
    mask = mask.float()
    num = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return num.sum() / denom


@torch.no_grad()
def evaluate(model, loader, device, supports) -> dict[str, float]:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        y_true = batch["y_true"][..., 0].to(device)
        y_mask = batch["y_mask"][..., 0].to(device)
        x_in = to_gwnet_input(x_hist)
        out = model(x_in)
        pred = from_gwnet_output(out)
        l = masked_mae(pred, y_true, y_mask)
        total += float(l.item()) * x_hist.size(0)
        n += x_hist.size(0)
    model.train()
    return {"L_main": total / max(n, 1)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"region: {args.region}", flush=True)

    if args.protocol == "full_window":
        rd_cls = FullWindowRegionData
        rd_kw = {"stride": args.stride, "train_frac": args.train_frac, "val_frac": args.val_frac}
        out_base = args.out_dir.with_name(args.out_dir.name + "_fullwindow")
    else:
        rd_cls, rd_kw, out_base = RegionData, {}, args.out_dir
    out_dir = out_base / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    def mk(split):
        return MultiRegionDataset(
            region_names=[args.region], data_dir=args.data_dir,
            graph_dir=args.graph_dir, split=split, lazy=False,
            region_data_cls=rd_cls, region_data_kwargs=rd_kw,
        )
    train_ds = mk("train")
    val_ds = mk("val")
    test_ds = mk("test")
    rdata = train_ds.regions[args.region]
    N = int(rdata.N)
    C_x = 3
    T_h = int(rdata.T_h)
    T_p = int(rdata.T_p)
    print(f"N={N}  C_x={C_x}  T_h={T_h}  T_p={T_p}", flush=True)
    print(f"|train|={len(train_ds)}  |val|={len(val_ds)}  |test|={len(test_ds)}",
          flush=True)

    supports = build_adj_supports(rdata.edge_index, N, device)
    model = gwnet(
        device=device, num_nodes=N, dropout=args.dropout, supports=supports,
        gcn_bool=True, addaptadj=True, in_dim=C_x, out_dim=T_p,
        residual_channels=args.nhid, dilation_channels=args.nhid,
        skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"gwnet params: {n_params:,}", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * n_batches,
                                   eta_min=args.lr * 1e-2)

    best_val = float("inf")
    no_improve = 0
    log_path = out_dir / "train.log"
    history = []
    with open(log_path, "w", encoding="utf-8") as lf:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            sum_loss, n = 0.0, 0
            for bi, batch in enumerate(train_loader):
                x_hist = batch["x_hist"].to(device)
                y_true = batch["y_true"][..., 0].to(device)
                y_mask = batch["y_mask"][..., 0].to(device)
                x_in = to_gwnet_input(x_hist)
                pred = from_gwnet_output(model(x_in))
                loss = masked_mae(pred, y_true, y_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                sum_loss += float(loss.item()) * x_hist.size(0)
                n += x_hist.size(0)
                if bi % 100 == 0:
                    print(f"  ep{epoch:02d} batch {bi}/{n_batches} L_main={loss.item():.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            train_avg = sum_loss / max(n, 1)
            val = evaluate(model, val_loader, device, supports)
            t = time.time() - t0
            print(f"==> ep{epoch:02d} train L_main={train_avg:.4f}  val L_main={val['L_main']:.4f}  ({t:.0f}s)",
                  flush=True)
            history.append({"epoch": epoch, "train_L_main": train_avg, "val_L_main": val["L_main"],
                            "time_s": t})
            lf.write(json.dumps(history[-1]) + "\n")
            lf.flush()
            if val["L_main"] < best_val:
                best_val = val["L_main"]
                no_improve = 0
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "val_L_main": val["L_main"], "config": vars(args),
                    "N": N, "C_x": C_x, "T_h": T_h, "T_p": T_p,
                }, out_dir / "ckpt_best.pt")
                print(f"    saved best", flush=True)
            else:
                no_improve += 1
                if args.patience and no_improve >= args.patience:
                    print(f"    early stop: no val improvement for {args.patience} epochs", flush=True)
                    break

    print(f"\nbest val L_main: {best_val:.4f}", flush=True)

    print(f"\n=== test inference ===", flush=True)
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
            x_hist = batch["x_hist"].to(device)
            # permute (B, N, T_p) → (B, T_p, N) for DIST-Net-compatible output schema
            y_true = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            aff = batch["affected_mask"].numpy()
            ss = batch["sample_start"].numpy()
            rc = batch["region_code"].numpy()
            x_in = to_gwnet_input(x_hist)
            pred = from_gwnet_output(model(x_in)).permute(0, 2, 1).cpu().numpy()
            bs = x_hist.size(0)
            pred_flow[cursor:cursor + bs] = pred
            actual_flow[cursor:cursor + bs] = y_true
            y_mask_flow[cursor:cursor + bs] = y_mask
            affected[cursor:cursor + bs] = aff
            sample_start[cursor:cursor + bs] = ss
            region_code[cursor:cursor + bs] = rc
            cursor += bs

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        region_code=region_code, sample_start=sample_start,
        region_node_idx=rdata.region_idx.astype(np.int64),
        pred_raw_flow=pred_flow, actual_future_flow=actual_flow,
        y_mask_flow=y_mask_flow, affected_mask=affected,
    )

    diff = np.abs(pred_flow - actual_flow)
    valid = y_mask_flow
    aff_TpN = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    mae_all = float(diff[valid].mean()) if valid.any() else float("nan")
    valid_aff = valid & aff_TpN
    mae_aff = float(diff[valid_aff].mean()) if valid_aff.any() else float("nan")
    valid_un = valid & (~aff_TpN)
    mae_un = float(diff[valid_un].mean()) if valid_un.any() else float("nan")
    summary = {"all": mae_all, "affected": mae_aff, "unaffected": mae_un,
               "best_val_L_main": best_val}
    print(f"\ntest MAE: all={mae_all:.3f}  affected={mae_aff:.3f}  unaffected={mae_un:.3f}",
          flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
