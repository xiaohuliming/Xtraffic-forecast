#!/usr/bin/env python3
"""Load impact-correction adapter, run inference on the FULL test split (no
sampling cap, sequential order matching the H5 cache test indices), and
denormalize the predicted residuals to raw flow units using the side cache
built by `build_test_raw_flow_cache.py`.

Outputs npz with `pred_residual_flow (N,H,K)`, `pred_raw_flow (N,H,K)`,
`source_raw_flow (N,H,K)`, plus pass-through metadata from the side cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from analyze_dual_branch_gate import make_model, torch_load
from build_impact_labels import CHANNEL_FLOW
from compare_dual_branch_group_metrics import resolve_cache_path
from train_dual_branch_gate_baseline import infer_cache_shapes
from train_full_candidate_stgnn_heatmap_model import (
    SPLIT_TO_CODE,
    compute_stats,
    make_loader,
    split_indices,
)
from train_impact_correction_adapter import ImpactCorrectionAdapter
from train_impact_residual_model import choose_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", type=Path, required=True,
                   help="Directory containing adapter model.pt + config.json")
    p.add_argument("--side-cache", type=Path,
                   default=Path("outputs/impact_guided_next_stage/headtohead_igstgnn/test_raw_flow_side_cache.npz"))
    p.add_argument("--out", type=Path,
                   default=None,
                   help="Output npz path (default: <adapter-dir>/test_raw_flow_predictions.npz)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = args.adapter_dir / "test_raw_flow_predictions.npz"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"device: {device}", flush=True)

    adapter_dir = args.adapter_dir.resolve()
    config = json.loads((adapter_dir / "config.json").read_text(encoding="utf-8"))
    source_dir = Path(config["model_dir"]).resolve()
    if not source_dir.is_absolute():
        source_dir = (adapter_dir.parent.parent / config["model_dir"]).resolve()

    print(f"adapter: {adapter_dir}")
    print(f"source : {source_dir}")
    src_ckpt = torch_load(source_dir / "model.pt")
    cache_path = resolve_cache_path(source_dir, src_ckpt)
    print(f"cache  : {cache_path}")

    base = make_model(src_ckpt, cache_path, device)
    shapes = infer_cache_shapes(cache_path)

    adapter_ckpt = torch_load(adapter_dir / "model.pt")
    base_beta = float(adapter_ckpt.get("base_beta", config.get("base_beta", 1.0)))
    print(f"base_beta: {base_beta}")

    model = ImpactCorrectionAdapter(
        base_model=base,
        base_beta=base_beta,
        channels=int(shapes["channels"]),
        horizon_steps=int(shapes["horizon_steps"]),
        global_context_dim=int(shapes["global_context_dim"]),
        hidden_dim=int(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.05)),
        max_correction=float(config.get("max_correction", 0.35)),
        correction_node_gate_mode=str(config.get("correction_node_gate_mode", "none")),
        correction_node_gate_floor=float(config.get("correction_node_gate_floor", 0.0)),
        correction_node_gate_temperature=float(config.get("correction_node_gate_temperature", 1.0)),
        correction_anomaly_gate_mode=str(config.get("correction_anomaly_gate_mode", "none")),
        correction_anomaly_gate_threshold=float(config.get("correction_anomaly_gate_threshold", 0.5)),
        correction_anomaly_gate_temperature=float(config.get("correction_anomaly_gate_temperature", 0.25)),
        correction_anomaly_gate_floor=float(config.get("correction_anomaly_gate_floor", 0.0)),
    ).to(device)

    state = model.state_dict()
    state.update({k: v for k, v in adapter_ckpt["adapter_state_dict"].items()})
    model.load_state_dict(state, strict=True)
    model.eval()

    stats = compute_stats(cache_path)
    splits = split_indices(cache_path)
    test_indices = np.sort(splits["test"])
    print(f"test samples (full split): {test_indices.size}")

    side = np.load(args.side_cache)
    if side["region_code"].size != test_indices.size:
        raise ValueError(
            f"side cache size {side['region_code'].size} != H5 test size {test_indices.size}"
        )
    print(f"side cache aligned: {test_indices.size} samples")

    # Sanity: cross-check first 5 sample node_idx between H5 and side
    with h5py.File(cache_path, "r") as h5:
        h5_node_idx = h5["node_idx"][test_indices[:5]]
    if not np.array_equal(h5_node_idx, side["node_idx"][:5]):
        raise RuntimeError("node_idx mismatch between H5 test order and side cache")
    print("sanity OK: node_idx aligned")

    loader = make_loader(cache_path, test_indices, stats, args.batch_size, shuffle=False)
    n_total = test_indices.size
    horizon = int(shapes["horizon_steps"])
    with h5py.File(cache_path, "r") as _h5:
        nodes = int(_h5["y_residual"].shape[2])
    pred_residual = np.empty((n_total, horizon, nodes), dtype=np.float32)
    source_residual = np.empty((n_total, horizon, nodes), dtype=np.float32)
    cursor = 0
    print("running adapter forward over test split", flush=True)
    with torch.no_grad():
        for batch in loader:
            (
                hist, hist_normal, node, global_ctx, normal_delta,
                _y, _y_mask, _impact, _impact_mask, _event_aux, _node_aff, _node_valid,
            ) = [item.to(device) for item in batch]
            if model.hist_input_channels > hist.shape[-1]:
                hist_in = torch.cat([hist, hist_normal], dim=-1)
            else:
                hist_in = hist
            pred_y, _pi, _pe, _pn, details = model(hist_in, node, global_ctx, normal_delta, return_details=True)
            bs = pred_y.shape[0]
            pred_residual[cursor:cursor + bs] = pred_y[..., CHANNEL_FLOW].detach().cpu().numpy()
            source_residual[cursor:cursor + bs] = details["source_pred"][..., CHANNEL_FLOW].detach().cpu().numpy()
            cursor += bs
            if cursor % 2048 == 0:
                print(f"  {cursor}/{n_total}", flush=True)
    if cursor != n_total:
        raise RuntimeError(f"cursor mismatch: {cursor} != {n_total}")
    print(f"forward done: {cursor} samples", flush=True)

    # Denormalize: raw_flow = normal_pred + pred_residual * fut_scale
    fut_scale = side["fut_scale_flow"].astype(np.float32)
    normal_pred = side["normal_pred_flow"].astype(np.float32)
    actual_flow = side["actual_future_flow"].astype(np.float32)
    pred_raw_flow = normal_pred + pred_residual * fut_scale
    source_raw_flow = normal_pred + source_residual * fut_scale

    # Quick sanity numbers (overall MAE on flow over y_mask)
    y_mask = side["y_mask_flow"].astype(bool)
    valid = y_mask & np.isfinite(pred_raw_flow) & np.isfinite(source_raw_flow) & np.isfinite(actual_flow)
    n_valid = int(valid.sum())
    if n_valid:
        mae_adapter = float(np.abs(pred_raw_flow[valid] - actual_flow[valid]).sum() / n_valid)
        mae_source = float(np.abs(source_raw_flow[valid] - actual_flow[valid]).sum() / n_valid)
        print(f"  sanity raw-flow MAE  adapter={mae_adapter:.4f}  source={mae_source:.4f}  N={n_valid}")

    np.savez_compressed(
        args.out,
        region_code=side["region_code"],
        sample_start=side["sample_start"],
        node_idx=side["node_idx"],
        node_valid=side["node_valid"],
        y_mask_flow=side["y_mask_flow"],
        actual_future_flow=actual_flow,
        normal_pred_flow=normal_pred,
        fut_scale_flow=fut_scale,
        pred_residual_flow=pred_residual,
        source_residual_flow=source_residual,
        pred_raw_flow=pred_raw_flow.astype(np.float32),
        source_raw_flow=source_raw_flow.astype(np.float32),
    )
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
