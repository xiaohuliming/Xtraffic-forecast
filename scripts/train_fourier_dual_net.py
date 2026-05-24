#!/usr/bin/env python3
"""Train FourierDualNet per region.

Two FFT-decomposed branches (main + pert) feeding two independent GraphWaveNet
backbones; final prediction = sum of branch outputs.

Two decomposition modes are supported:
  --decomp_mode fixed_k     : hard cutoff at bin K (default K=3)
  --decomp_mode learnable   : per-bin sigmoid mask (initialised at step K)

Writes:
  outputs/fourier_dual_net/<tag>/<region>/
    ckpt_best.pt
    test_predictions.npz   (same schema as GraphWaveNet baseline)
    summary.json
    train.log
    decomp_mask.json       (final per-bin mask, learnable mode only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, make_loader
from fourier_dual_net.model import FourierDualNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/fourier_dual_net"))
    p.add_argument("--tag", default=None, help="run tag; default=<decomp_mode>_K<K>")
    p.add_argument("--decomp_mode", choices=["fixed_k", "learnable", "conditioned"], default="fixed_k")
    p.add_argument("--K", type=int, default=3)
    # FDN++ architectural specialisation
    p.add_argument("--use_time_emb", action="store_true",
                   help="Main branch sees ToD/DoW as extra input channels (direction A)")
    p.add_argument("--main_blocks", type=int, default=4, help="Main branch GWN blocks (default 4 = original)")
    p.add_argument("--main_layers", type=int, default=2, help="Main branch GWN layers per block (default 2)")
    p.add_argument("--pert_blocks", type=int, default=4, help="Pert branch GWN blocks")
    p.add_argument("--pert_layers", type=int, default=2, help="Pert branch GWN layers per block")
    p.add_argument("--use_cross_attn", action="store_true",
                   help="Pert branch attends to Main signal at input level (direction B)")
    p.add_argument("--cross_attn_dim", type=int, default=16)
    p.add_argument("--cross_attn_heads", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="single epoch, 20 train batches, for quick sanity check")
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_adj_supports(edge_index: np.ndarray, N: int, device: torch.device
                       ) -> list[torch.Tensor]:
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    A_fwd = deg_inv[:, None] * A
    A_bwd = (deg_inv[:, None] * A.T)
    return [torch.from_numpy(A_fwd).to(device),
            torch.from_numpy(A_bwd).to(device)]


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    num = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return num.sum() / denom


@torch.no_grad()
def evaluate(model, loader, device, sensor_meta_tensor=None) -> dict[str, float]:
    model.eval()
    total, n = 0.0, 0
    need_time = model.use_time_emb or model.requires_sensor_meta
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        y_true = batch["y_true"][..., 0].to(device)
        y_mask = batch["y_mask"][..., 0].to(device)
        time_feat = batch["time_feat"].to(device) if need_time else None
        sm = sensor_meta_tensor if model.requires_sensor_meta else None
        pred = model(x_hist, time_feat=time_feat, sensor_meta=sm)
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
    print(f"region: {args.region}  decomp_mode={args.decomp_mode}  K={args.K}", flush=True)

    tag = args.tag or f"{args.decomp_mode}_K{args.K}"
    out_dir = args.out_dir / tag / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiRegionDataset(
        region_names=[args.region], data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="train", lazy=False,
    )
    val_ds = MultiRegionDataset(
        region_names=[args.region], data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="val", lazy=False,
    )
    test_ds = MultiRegionDataset(
        region_names=[args.region], data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="test", lazy=False,
    )
    rdata = train_ds.regions[args.region]
    N = int(rdata.N)
    C_x = 3
    T_h = int(rdata.T_h)
    T_p = int(rdata.T_p)
    print(f"N={N}  C_x={C_x}  T_h={T_h}  T_p={T_p}", flush=True)
    print(f"|train|={len(train_ds)}  |val|={len(val_ds)}  |test|={len(test_ds)}",
          flush=True)

    supports = build_adj_supports(rdata.edge_index, N, device)
    sensor_meta_dim = int(rdata.C_meta)
    model = FourierDualNet(
        num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
        K=args.K, decomp_mode=args.decomp_mode,
        in_dim_flow=C_x, nhid=args.nhid, dropout=args.dropout, device=device,
        main_blocks=args.main_blocks, main_layers=args.main_layers,
        pert_blocks=args.pert_blocks, pert_layers=args.pert_layers,
        use_time_emb=args.use_time_emb,
        use_cross_attn=args.use_cross_attn,
        cross_attn_dim=args.cross_attn_dim,
        cross_attn_heads=args.cross_attn_heads,
        sensor_meta_dim=sensor_meta_dim,
    ).to(device)
    # Pre-load static sensor_meta tensor for conditioned mask
    sensor_meta_tensor = torch.from_numpy(rdata.static_meta.astype("float32")).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_decomp = sum(p.numel() for p in model.decomp.parameters() if p.requires_grad)
    print(f"FourierDualNet params: total={n_params:,}  decomp={n_params_decomp}", flush=True)
    if args.decomp_mode != "conditioned":
        print(f"initial decomp mask: {model.decomp.get_mask().detach().cpu().numpy().round(3).tolist()}",
              flush=True)
    else:
        print(f"conditioned decomp mask: per-(sample,sensor) MLP — no static snapshot", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1) * max(n_batches, 1),
        eta_min=args.lr * 1e-2,
    )

    epochs = 1 if args.smoke else args.epochs
    max_batches = 20 if args.smoke else None

    best_val = float("inf")
    log_path = out_dir / "train.log"
    history = []
    with open(log_path, "w", encoding="utf-8") as lf:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            sum_loss, n = 0.0, 0
            for bi, batch in enumerate(train_loader):
                if max_batches is not None and bi >= max_batches:
                    break
                x_hist = batch["x_hist"].to(device)
                y_true = batch["y_true"][..., 0].to(device)
                y_mask = batch["y_mask"][..., 0].to(device)
                need_time = model.use_time_emb or model.requires_sensor_meta
                time_feat = batch["time_feat"].to(device) if need_time else None
                sm = sensor_meta_tensor if model.requires_sensor_meta else None
                pred = model(x_hist, time_feat=time_feat, sensor_meta=sm)
                loss = masked_mae(pred, y_true, y_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                sum_loss += float(loss.item()) * x_hist.size(0)
                n += x_hist.size(0)
                if bi % 100 == 0:
                    print(f"  ep{epoch:02d} batch {bi}/{n_batches} "
                          f"L_main={loss.item():.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            train_avg = sum_loss / max(n, 1)
            val = evaluate(model, val_loader, device, sensor_meta_tensor) if not args.smoke else {"L_main": float("nan")}
            t = time.time() - t0
            if args.decomp_mode == "conditioned":
                mask_snapshot = "conditioned"   # per-sample mask not summarisable
            else:
                mask_snapshot = model.decomp.get_mask().detach().cpu().numpy().round(3).tolist()
            print(f"==> ep{epoch:02d} train L_main={train_avg:.4f}  "
                  f"val L_main={val['L_main']:.4f}  ({t:.0f}s)  mask={mask_snapshot}",
                  flush=True)
            history.append({"epoch": epoch, "train_L_main": train_avg,
                            "val_L_main": val["L_main"], "time_s": t,
                            "mask": mask_snapshot})
            lf.write(json.dumps(history[-1]) + "\n")
            lf.flush()
            if val["L_main"] < best_val:
                best_val = val["L_main"]
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "val_L_main": val["L_main"], "config": vars(args),
                    "N": N, "C_x": C_x, "T_h": T_h, "T_p": T_p,
                }, out_dir / "ckpt_best.pt")
                print(f"    saved best", flush=True)

    if args.smoke:
        print("\nsmoke run complete (no test inference)", flush=True)
        return

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
            y_true = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            aff = batch["affected_mask"].numpy()
            ss = batch["sample_start"].numpy()
            rc = batch["region_code"].numpy()
            need_time = model.use_time_emb or model.requires_sensor_meta
            time_feat = batch["time_feat"].to(device) if need_time else None
            sm = sensor_meta_tensor if model.requires_sensor_meta else None
            pred = model(x_hist, time_feat=time_feat, sensor_meta=sm).permute(0, 2, 1).cpu().numpy()
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
    if args.decomp_mode == "conditioned":
        final_mask = "conditioned (per-sample)"
    else:
        final_mask = model.decomp.get_mask().detach().cpu().numpy().tolist()
    summary = {"all": mae_all, "affected": mae_aff, "unaffected": mae_un,
               "best_val_L_main": best_val,
               "decomp_mode": args.decomp_mode, "K": args.K,
               "final_decomp_mask": final_mask}
    print(f"\ntest MAE: all={mae_all:.3f}  affected={mae_aff:.3f}  unaffected={mae_un:.3f}",
          flush=True)
    print(f"final decomp mask: {[round(v, 3) for v in final_mask]}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    (out_dir / "decomp_mask.json").write_text(
        json.dumps({"mode": args.decomp_mode, "K": args.K, "mask": final_mask}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
