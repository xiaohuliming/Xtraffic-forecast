#!/usr/bin/env python3
"""Run a DualBranchSTTISGate base model directly on the test split (no adapter,
no fine-tunes), denormalize to raw flow units using the side cache, and write
predictions npz with the same schema as run_adapter_test_inference.py output.
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
from train_dual_branch_gate_baseline import infer_cache_shapes
from train_full_candidate_stgnn_heatmap_model import (
    SPLIT_TO_CODE,
    compute_stats,
    make_loader,
    split_indices,
)
from train_impact_residual_model import choose_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing base model.pt + config.json")
    p.add_argument("--side-cache", type=Path,
                   default=Path("outputs/impact_guided_next_stage/headtohead_igstgnn/test_raw_flow_side_cache.npz"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--residual-beta", type=float, default=None,
                   help="Override residual scaling beta (defaults to ckpt's residual_beta or 1.0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = args.model_dir / "test_raw_flow_predictions.npz"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)

    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cache_path_cfg = config.get("cache_path") or "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"
    cache_path = Path(cache_path_cfg)
    print(f"cache  : {cache_path}")
    print(f"model  : {model_dir}")

    ckpt = torch_load(model_dir / "model.pt")
    model = make_model(ckpt, cache_path, device)
    model.eval()

    if args.residual_beta is not None:
        residual_beta = float(args.residual_beta)
    else:
        residual_beta = float(ckpt.get("residual_beta", config.get("residual_beta", 1.0)))
    print(f"residual_beta: {residual_beta}")

    stats = compute_stats(cache_path)
    splits = split_indices(cache_path)
    test_indices = np.sort(splits["test"])
    print(f"test samples (full split): {test_indices.size}")

    side = np.load(args.side_cache)
    if side["region_code"].size != test_indices.size:
        raise ValueError(
            f"side cache size {side['region_code'].size} != H5 test size {test_indices.size}"
        )
    with h5py.File(cache_path, "r") as h5:
        h5_node_idx = h5["node_idx"][test_indices[:5]]
    if not np.array_equal(h5_node_idx, side["node_idx"][:5]):
        raise RuntimeError("node_idx mismatch between H5 test order and side cache")
    print("sanity OK: node_idx aligned")

    shapes = infer_cache_shapes(cache_path)
    horizon = int(shapes["horizon_steps"])
    with h5py.File(cache_path, "r") as _h5:
        nodes = int(_h5["y_residual"].shape[2])
    n_total = test_indices.size
    pred_residual = np.empty((n_total, horizon, nodes), dtype=np.float32)
    cursor = 0
    loader = make_loader(cache_path, test_indices, stats, args.batch_size, shuffle=False)
    print("running base model forward over test split", flush=True)
    hist_input_channels = int(getattr(model, "hist_input_channels", shapes["channels"]))
    with torch.no_grad():
        for batch in loader:
            (
                hist, hist_normal, node, global_ctx, normal_delta,
                _y, _y_mask, _impact, _impact_mask, _event_aux, _node_aff, _node_valid,
            ) = [item.to(device) for item in batch]
            if hist_input_channels > hist.shape[-1]:
                hist_in = torch.cat([hist, hist_normal], dim=-1)
            else:
                hist_in = hist
            out = model(hist_in, node, global_ctx, normal_delta)
            pred_y = out[0] if isinstance(out, (tuple, list)) else out
            scaled = residual_beta * pred_y
            bs = scaled.shape[0]
            pred_residual[cursor:cursor + bs] = scaled[..., CHANNEL_FLOW].detach().cpu().numpy()
            cursor += bs
            if cursor % 2048 == 0:
                print(f"  {cursor}/{n_total}", flush=True)
    if cursor != n_total:
        raise RuntimeError(f"cursor mismatch: {cursor} != {n_total}")
    print(f"forward done: {cursor} samples", flush=True)

    fut_scale = side["fut_scale_flow"].astype(np.float32)
    normal_pred = side["normal_pred_flow"].astype(np.float32)
    actual_flow = side["actual_future_flow"].astype(np.float32)
    pred_raw_flow = normal_pred + pred_residual * fut_scale

    y_mask = side["y_mask_flow"].astype(bool)
    valid = y_mask & np.isfinite(pred_raw_flow) & np.isfinite(actual_flow)
    n_valid = int(valid.sum())
    if n_valid:
        mae = float(np.abs(pred_raw_flow[valid] - actual_flow[valid]).sum() / n_valid)
        print(f"  sanity raw-flow MAE  base={mae:.4f}  N={n_valid}")

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
        # Reuse adapter schema: source==pred==base for compatibility with compare script
        source_residual_flow=pred_residual,
        pred_raw_flow=pred_raw_flow.astype(np.float32),
        source_raw_flow=pred_raw_flow.astype(np.float32),
    )
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
