#!/usr/bin/env python3
"""Run a trained DIST-Net checkpoint on the test split, save raw-flow predictions.

Output per region (under <ckpt dir>/eval/ or --out-dir):
  {region}_test_predictions.npz with arrays:
    region_code            (S,)        int64
    sample_start           (S,)        int64
    region_node_idx        (N,)        int64     index in global 16972 space
    pred_raw_flow          (S, T_p, N) float32   flow channel only
    actual_future_flow     (S, T_p, N) float32
    y_mask_flow            (S, T_p, N) bool
    affected_mask          (S, N)      bool

Also prints per-region MAE on (all valid positions) and (affected nodes only).
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

from dist_net.data import MultiRegionDataset, make_loader, REGION_NAME_TO_CODE
from dist_net.model import DISTNet, DISTNetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("outputs/dist_net/region_data"))
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--regions", nargs="+",
                   default=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <ckpt parent>/eval")
    p.add_argument("--device", default=None)
    p.add_argument("--blind", action="store_true",
                   help="Zero incident_feat and incident_mask at inference. "
                        "Tests whether the model actually uses incident labels "
                        "or just relies on traffic history.")
    p.add_argument("--single-branch", action="store_true",
                   help="Build a single-branch model (no IncidentBranch / cross-attn / gate). "
                        "Must match the ckpt's training config.")
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_from_ckpt(ckpt: dict, device: torch.device,
                          single_branch_override: bool = False) -> DISTNet:
    cfg_dict = ckpt["model_config"]
    cfg = DISTNetConfig(
        c_x=int(cfg_dict.get("c_x", 3)),
        c_meta=int(cfg_dict.get("c_meta", 8)),
        c_e=int(cfg_dict.get("c_e", 13)),
        d_t=int(cfg_dict.get("d_t", 5)),
        n_regions=int(cfg_dict.get("n_regions", 3)),
        hidden_dim=int(cfg_dict.get("hidden_dim", 64)),
        t_h=int(cfg_dict.get("t_h", 12)),
        t_p=int(cfg_dict.get("t_p", 12)),
        single_branch=bool(cfg_dict.get("single_branch", False)) or single_branch_override,
    )
    model = DISTNet(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    return model


@torch.no_grad()
def run_region(region_name: str, model: DISTNet, args: argparse.Namespace,
               device: torch.device, out_dir: Path) -> dict[str, float] | None:
    print(f"\n=== {region_name} ===", flush=True)
    t0 = time.time()
    ds = MultiRegionDataset(
        region_names=[region_name], data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="test", lazy=False,
    )
    print(f"  test split size: {len(ds)}")
    if len(ds) == 0:
        print("  empty, skipping")
        return None
    rdata = ds.regions[region_name]
    edge_index = torch.from_numpy(rdata.edge_index).to(device)
    N = rdata.N
    T_p = rdata.T_p
    S = len(ds)

    pred_flow = np.empty((S, T_p, N), dtype=np.float32)
    actual_flow = np.empty((S, T_p, N), dtype=np.float32)
    y_mask_flow = np.empty((S, T_p, N), dtype=bool)
    affected = np.empty((S, N), dtype=bool)
    sample_start = np.empty((S,), dtype=np.int64)
    region_code = np.empty((S,), dtype=np.int64)

    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    cursor = 0
    last_print = 0
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        if args.blind or args.single_branch:
            batch_dev["incident_feat"] = torch.zeros_like(batch_dev["incident_feat"])
            batch_dev["incident_mask"] = torch.zeros_like(batch_dev["incident_mask"])
            if "rel_feat" in batch_dev:
                batch_dev["rel_feat"] = torch.zeros_like(batch_dev["rel_feat"])
        out = model(batch_dev, edge_index=edge_index)
        pred = out["pred"]                                 # (B, N, T_p, C_x)
        # convert to (B, T_p, N) flow channel
        pred_f = pred[..., 0].permute(0, 2, 1).cpu().numpy()
        y_f    = batch_dev["y_true"][..., 0].permute(0, 2, 1).cpu().numpy()
        mask_f = batch_dev["y_mask"][..., 0].permute(0, 2, 1).cpu().numpy()
        aff    = batch_dev["affected_mask"].cpu().numpy()
        ss     = batch_dev["sample_start"].cpu().numpy()
        rc     = batch_dev["region_code"].cpu().numpy()

        bs = pred_f.shape[0]
        pred_flow[cursor:cursor + bs]   = pred_f
        actual_flow[cursor:cursor + bs] = y_f
        y_mask_flow[cursor:cursor + bs] = mask_f
        affected[cursor:cursor + bs]    = aff
        sample_start[cursor:cursor + bs] = ss
        region_code[cursor:cursor + bs]  = rc
        cursor += bs
        if cursor - last_print > 1000 or cursor == S:
            print(f"  inference: {cursor}/{S}", flush=True)
            last_print = cursor

    if args.single_branch:
        suffix = "_single"
    elif args.blind:
        suffix = "_blind"
    else:
        suffix = ""
    out_path = out_dir / f"{region_name}_test_predictions{suffix}.npz"
    np.savez_compressed(
        out_path,
        region_code=region_code,
        sample_start=sample_start,
        region_node_idx=rdata.region_idx.astype(np.int64),
        pred_raw_flow=pred_flow,
        actual_future_flow=actual_flow,
        y_mask_flow=y_mask_flow,
        affected_mask=affected,
    )
    sz_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"  saved -> {out_path} ({sz_mb:.1f} MB) in {time.time() - t0:.1f}s")

    # Quick MAE summary on raw flow
    diff = np.abs(pred_flow - actual_flow)
    valid = y_mask_flow
    mae_all = float(diff[valid].mean()) if valid.any() else float("nan")
    # affected mask expanded to (S, T_p, N)
    aff_TpN = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    valid_aff = valid & aff_TpN
    mae_aff = float(diff[valid_aff].mean()) if valid_aff.any() else float("nan")
    valid_un = valid & (~aff_TpN)
    mae_un = float(diff[valid_un].mean()) if valid_un.any() else float("nan")
    print(f"  raw-flow MAE: all={mae_all:.3f}  affected={mae_aff:.3f}  unaffected={mae_un:.3f}")
    return {"all": mae_all, "affected": mae_aff, "unaffected": mae_un}


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    if args.blind:
        print("*** BLIND MODE: incident_feat / incident_mask / rel_feat zeroed ***", flush=True)
    if args.single_branch:
        print("*** SINGLE-BRANCH MODE: building model without IncidentBranch / cross-attn / gate ***",
              flush=True)

    out_dir = args.out_dir or (args.ckpt.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {out_dir}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device, single_branch_override=args.single_branch)
    print(f"loaded ckpt: epoch={ckpt.get('epoch', '?')}, "
          f"saved at val L_main={ckpt.get('val_metrics', {}).get('L_main', '?')}")

    summary = {}
    for region_name in args.regions:
        try:
            res = run_region(region_name, model, args, device, out_dir)
            if res:
                summary[region_name] = res
        except FileNotFoundError as e:
            print(f"  skip {region_name}: missing data ({e})")

    if args.single_branch:
        suffix = "_single"
    elif args.blind:
        suffix = "_blind"
    else:
        suffix = ""
    summary_path = out_dir / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
