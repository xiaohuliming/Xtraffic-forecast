#!/usr/bin/env python3
"""Build a side cache of raw-flow ground truth + learned normal predictions for
the existing test split, so adapter outputs (normalized residuals) can be
denormalized to raw flow units for head-to-head comparison with IGSTGNN.

Iterates the same regions/events/offsets as the original H5 cache builder to
preserve sample order with `outputs/.../full_candidate_samples.h5` test split.

Outputs npz with arrays aligned to that test order:
  region_code (N,), sample_start (N,), node_idx (N, K), node_valid (N, K),
  fut_scale_flow (N, H, K), normal_pred_flow (N, H, K), actual_future_flow (N, H, K),
  y_mask_flow (N, H, K, bool)

Only the FLOW channel is stored (IGSTGNN forecasts flow only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_impact_labels import (
    CHANNEL_FLOW,
    build_baseline_valid_mask,
    build_matches,
    build_robust_baseline,
    load_incidents_2023,
    load_region_traffic,
    load_sensor_meta,
    region_specs,
)
from train_candidate_residual_model import (
    build_candidate_lookup,
    build_node_context,
    select_candidate_nodes,
)
from train_full_candidate_stgnn_heatmap_model import (
    LearnedNormalRegion,
    SPLIT_TO_CODE,
)
from train_impact_residual_model import choose_device, split_name
from validate_forecast_error_against_impact import fit_blend_alphas, parse_incident_ids


REGIONS = ["Alameda", "ContraCosta", "Orange"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("archive"))
    p.add_argument("--event-root", type=Path,
                   default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"))
    p.add_argument("--raw-label-dir", type=Path, default=Path("outputs/impact_labels"))
    p.add_argument("--normal-model-dir", type=Path,
                   default=Path("outputs/impact_guided_next_stage/normal_stgnn_forecaster"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/impact_guided_next_stage/headtohead_igstgnn/test_raw_flow_side_cache.npz"))
    p.add_argument("--regions", nargs="+", default=REGIONS)
    p.add_argument("--input-steps", type=int, default=12)
    p.add_argument("--horizon-steps", type=int, default=12)
    p.add_argument("--max-candidate-nodes", type=int, default=36)
    p.add_argument("--sample-offsets", nargs="+", type=int, default=[0, 6, 12])
    p.add_argument("--candidate-pm-radius", type=float, default=5.0)
    p.add_argument("--anchor-pm-radius", type=float, default=2.0)
    p.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    p.add_argument("--min-baseline-count", type=int, default=8)
    p.add_argument("--normal-infer-batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else choose_device()
    print(f"device={device}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    event_root = args.event_root.resolve()
    raw_label_dir = args.raw_label_dir.resolve()
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)

    out_region = []
    out_start = []
    out_node_idx = []
    out_node_valid = []
    out_fut_scale = []
    out_normal_pred = []
    out_actual = []
    out_y_mask = []

    for region_code, region_name in enumerate(args.regions):
        region = region_specs()[region_name]
        region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
        region_meta = region_meta.reset_index(drop=True)
        region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)

        print(f"[{region_name}] loading traffic", flush=True)
        traffic, times = load_region_traffic(data_dir, region_node_idx)
        day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
        tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
        total_steps = traffic.shape[0]

        print(f"[{region_name}] fitting baseline + alphas", flush=True)
        matches = build_matches(
            inc=inc, region_meta=region_meta, times=times,
            candidate_pm_radius=args.candidate_pm_radius,
            anchor_pm_radius=args.anchor_pm_radius,
            baseline_mask_extra_steps=args.baseline_mask_extra_steps,
        )
        baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)
        train_valid = baseline_valid.copy()
        train_valid[int(total_steps * 0.70):, :] = False
        baseline, scale, _ = build_robust_baseline(
            traffic=traffic, times=times,
            baseline_valid=train_valid,
            min_count=args.min_baseline_count,
        )
        alphas = fit_blend_alphas(
            traffic=traffic, times=times, train_valid=train_valid,
            baseline=baseline, input_steps=args.input_steps, horizon_steps=args.horizon_steps,
        )
        print(f"[{region_name}] loading learned normal model", flush=True)
        learned_normal = LearnedNormalRegion.load(args.normal_model_dir, region_name, device)
        if learned_normal.model.horizon_steps != args.horizon_steps:
            raise ValueError("normal model horizon mismatch")

        print(f"[{region_name}] iterating events", flush=True)
        events = pd.read_csv(event_root / region_name / "event_labels.csv")
        raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
        candidate_lookup = build_candidate_lookup(raw_nodes)

        pending: list[dict] = []
        n_emitted = 0

        def flush() -> None:
            nonlocal n_emitted
            if not pending:
                return
            hist = np.stack([it["hist"] for it in pending]).astype(np.float32)
            blend = np.stack([it["blend"] for it in pending]).astype(np.float32)
            node_idx_arr = np.stack([it["node_idx"] for it in pending]).astype(np.int32)
            node_valid_arr = np.stack([it["node_valid"] for it in pending]).astype(np.float32)
            sample_start_arr = np.asarray([it["sample_start"] for it in pending], dtype=np.int32)
            normal_pred_batch = learned_normal.predict_many(
                hist=hist, blend=blend,
                node_idx=node_idx_arr, node_valid=node_valid_arr,
                times=times, sample_start=sample_start_arr,
            )
            for i, it in enumerate(pending):
                normal_pred = normal_pred_batch[i]  # (H, K, C)
                fut_scale = it["fut_scale"]
                actual = it["actual"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_res_z = (actual - normal_pred) / fut_scale
                valid_mask = it["node_valid"].astype(bool)
                y_mask = np.isfinite(y_res_z)
                y_mask[:, ~valid_mask, :] = False
                if not y_mask.any():
                    continue  # matches H5 cache filter
                # Only store FLOW channel
                out_region.append(region_code)
                out_start.append(it["sample_start"])
                out_node_idx.append(it["node_idx"].astype(np.int32))
                out_node_valid.append(it["node_valid"].astype(np.float32))
                out_fut_scale.append(fut_scale[:, :, CHANNEL_FLOW].astype(np.float32))
                out_normal_pred.append(normal_pred[:, :, CHANNEL_FLOW].astype(np.float32))
                out_actual.append(actual[:, :, CHANNEL_FLOW].astype(np.float32))
                out_y_mask.append(y_mask[:, :, CHANNEL_FLOW].astype(bool))
                n_emitted += 1
            pending.clear()

        for row in events.itertuples(index=False):
            incident_ids = parse_incident_ids(row.incident_ids)
            candidates = select_candidate_nodes(
                incident_ids=incident_ids,
                candidate_lookup=candidate_lookup,
                anchor_region_idx=int(row.anchor_region_idx),
                max_nodes=args.max_candidate_nodes,
            )
            node_idx, node_valid, _node_affected, _node_ctx = build_node_context(
                candidates=candidates,
                max_nodes=args.max_candidate_nodes,
                pm_radius=args.candidate_pm_radius,
                anchor_idx=int(row.anchor_region_idx),
            )

            for offset in args.sample_offsets:
                sample_start = int(row.start_idx) + int(offset)
                input_start = sample_start - args.input_steps
                future_end = sample_start + args.horizon_steps
                if input_start < 0 or future_end > total_steps:
                    continue
                sample_split = split_name(sample_start, total_steps)
                if sample_split != "test":
                    continue  # only collect test samples

                input_idx = np.arange(input_start, sample_start, dtype=np.int32)
                future_idx = np.arange(sample_start, future_end, dtype=np.int32)
                hist = traffic[input_idx][:, node_idx, :]
                fut_base = baseline[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
                fut_scale = scale[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
                actual_future = traffic[future_idx][:, node_idx, :]
                last_obs = traffic[sample_start - 1, node_idx, :]
                blend_pred = np.empty_like(actual_future, dtype=np.float32)
                for h in range(args.horizon_steps):
                    blend_pred[h] = fut_base[h] + alphas[h][None, :] * (last_obs - fut_base[h])

                pending.append({
                    "hist": hist.astype(np.float32),
                    "blend": blend_pred.astype(np.float32),
                    "fut_scale": fut_scale.astype(np.float32),
                    "actual": actual_future.astype(np.float32),
                    "node_idx": node_idx.astype(np.int32),
                    "node_valid": node_valid.astype(np.float32),
                    "sample_start": int(sample_start),
                })
                if len(pending) >= args.normal_infer_batch_size:
                    flush()

        flush()
        del learned_normal, traffic, baseline, scale, alphas
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"[{region_name}] emitted test samples: {n_emitted}", flush=True)

    print(f"saving {len(out_region)} test samples to {args.out}")
    np.savez_compressed(
        args.out,
        region_code=np.asarray(out_region, dtype=np.int8),
        sample_start=np.asarray(out_start, dtype=np.int32),
        node_idx=np.stack(out_node_idx).astype(np.int32),
        node_valid=np.stack(out_node_valid).astype(np.float32),
        fut_scale_flow=np.stack(out_fut_scale).astype(np.float32),
        normal_pred_flow=np.stack(out_normal_pred).astype(np.float32),
        actual_future_flow=np.stack(out_actual).astype(np.float32),
        y_mask_flow=np.stack(out_y_mask),
    )
    print("done")


if __name__ == "__main__":
    main()
