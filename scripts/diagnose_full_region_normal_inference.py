#!/usr/bin/env python3
"""Compare local candidate-subgraph and full-region learned-normal inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_impact_labels import (
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
    CHANNELS,
    LearnedNormalRegion,
    build_blend_prediction_batch,
)
from train_impact_residual_model import choose_device, split_name
from validate_forecast_error_against_impact import fit_blend_alphas, parse_incident_ids


SUBSETS = ("all", "affected", "unaffected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--event-root", type=Path, default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"))
    parser.add_argument("--raw-label-dir", type=Path, default=Path("outputs/impact_labels"))
    parser.add_argument("--normal-model-dir", type=Path, default=Path("outputs/impact_guided_next_stage/normal_stgnn_forecaster"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/impact_guided_next_stage/full_region_normal_diagnostics"))
    parser.add_argument("--regions", nargs="+", default=["Alameda", "ContraCosta", "Orange"])
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--max-candidate-nodes", type=int, default=36)
    parser.add_argument("--sample-offsets", nargs="+", type=int, default=[0, 6, 12])
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def empty_sums() -> dict[str, dict[str, float]]:
    return {
        subset: {
            "count": 0.0,
            "normal_diff_abs": 0.0,
            "normal_diff_robust": 0.0,
            "local_target_robust": 0.0,
            "full_target_robust": 0.0,
        }
        for subset in SUBSETS
    }


def add_metrics(
    sums: dict[str, dict[str, float]],
    local_pred: np.ndarray,
    full_pred: np.ndarray,
    actual: np.ndarray,
    scale: np.ndarray,
    node_valid: np.ndarray,
    node_affected: np.ndarray,
) -> None:
    safe_scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    base_mask = (
        node_valid[:, None, :, None].astype(bool)
        & np.isfinite(local_pred)
        & np.isfinite(full_pred)
        & np.isfinite(actual)
        & np.isfinite(safe_scale)
    )
    affected_mask = base_mask & node_affected[:, None, :, None].astype(bool)
    unaffected_mask = base_mask & (~node_affected[:, None, :, None].astype(bool))
    masks = {"all": base_mask, "affected": affected_mask, "unaffected": unaffected_mask}

    diff_abs = np.abs(full_pred - local_pred)
    diff_robust = diff_abs / safe_scale
    local_target = np.abs(actual - local_pred) / safe_scale
    full_target = np.abs(actual - full_pred) / safe_scale
    for subset, mask in masks.items():
        count = float(mask.sum())
        if count <= 0:
            continue
        sums[subset]["count"] += count
        sums[subset]["normal_diff_abs"] += float(diff_abs[mask].sum())
        sums[subset]["normal_diff_robust"] += float(diff_robust[mask].sum())
        sums[subset]["local_target_robust"] += float(local_target[mask].sum())
        sums[subset]["full_target_robust"] += float(full_target[mask].sum())


def finalize_sums(sums: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for subset, values in sums.items():
        count = max(values["count"], 1.0)
        local_target = values["local_target_robust"] / count
        full_target = values["full_target_robust"] / count
        out[f"{subset}_values"] = values["count"]
        out[f"{subset}_normal_diff_raw_mae"] = values["normal_diff_abs"] / count
        out[f"{subset}_normal_diff_robust_mae"] = values["normal_diff_robust"] / count
        out[f"{subset}_local_target_robust_mae"] = local_target
        out[f"{subset}_full_target_robust_mae"] = full_target
        out[f"{subset}_target_change_pct"] = 100.0 * (full_target - local_target) / local_target if local_target > 0 else float("nan")
    return out


def flush_batch(
    pending: list[dict[str, object]],
    learned_normal: LearnedNormalRegion,
    traffic: np.ndarray,
    baseline: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    alphas: np.ndarray,
    times: pd.DatetimeIndex,
    args: argparse.Namespace,
    sums: dict[str, dict[str, float]],
) -> int:
    if not pending:
        return 0
    hist = np.stack([item["hist"] for item in pending]).astype(np.float32)
    blend = np.stack([item["blend"] for item in pending]).astype(np.float32)
    actual = np.stack([item["actual"] for item in pending]).astype(np.float32)
    scale = np.stack([item["scale"] for item in pending]).astype(np.float32)
    node_idx = np.stack([item["node_idx"] for item in pending]).astype(np.int32)
    node_valid = np.stack([item["node_valid"] for item in pending]).astype(np.float32)
    node_affected = np.stack([item["node_affected"] for item in pending]).astype(np.float32)
    sample_start = np.asarray([item["sample_start"] for item in pending], dtype=np.int32)

    local_pred = learned_normal.predict_many(
        hist=hist,
        blend=blend,
        node_idx=node_idx,
        node_valid=node_valid,
        times=times,
        sample_start=sample_start,
    )
    full_hist = np.stack([traffic[int(start) - args.input_steps : int(start)] for start in sample_start]).astype(np.float32)
    full_blend = build_blend_prediction_batch(
        traffic=traffic,
        baseline=baseline,
        day_kind=day_kind,
        tod=tod,
        alphas=alphas,
        starts=sample_start,
        horizon_steps=args.horizon_steps,
    )
    full_pred = learned_normal.predict_many_full(
        full_hist=full_hist,
        full_blend=full_blend,
        node_idx=node_idx,
        node_valid=node_valid,
        times=times,
        sample_start=sample_start,
    )
    add_metrics(
        sums=sums,
        local_pred=local_pred,
        full_pred=full_pred,
        actual=actual,
        scale=scale,
        node_valid=node_valid,
        node_affected=node_affected,
    )
    n = len(pending)
    pending.clear()
    return n


def run_region(region_name: str, region_code: int, args: argparse.Namespace, meta: pd.DataFrame, inc: pd.DataFrame) -> dict[str, float]:
    device = choose_device(args.device)
    spec = region_specs()[region_name]
    region_meta = meta[(meta["County"] == spec.county) & (meta["Type"] == "Mainline")].copy().reset_index(drop=True)
    region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)

    print(f"[{region_name}] loading traffic", flush=True)
    traffic, times = load_region_traffic(args.data_dir.resolve(), region_node_idx)
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    total_steps = traffic.shape[0]

    print(f"[{region_name}] fitting statistical blend baseline", flush=True)
    matches = build_matches(
        inc=inc,
        region_meta=region_meta,
        times=times,
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
        baseline_mask_extra_steps=args.baseline_mask_extra_steps,
    )
    baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)
    train_valid = baseline_valid.copy()
    train_valid[int(total_steps * 0.70) :, :] = False
    baseline, scale, _ = build_robust_baseline(
        traffic=traffic,
        times=times,
        baseline_valid=train_valid,
        min_count=args.min_baseline_count,
    )
    alphas = fit_blend_alphas(
        traffic=traffic,
        times=times,
        train_valid=train_valid,
        baseline=baseline,
        input_steps=args.input_steps,
        horizon_steps=args.horizon_steps,
    )

    print(f"[{region_name}] loading learned normal branch", flush=True)
    learned_normal = LearnedNormalRegion.load(args.normal_model_dir, region_name, device)
    events = pd.read_csv(args.event_root / region_name / "event_labels.csv")
    raw_nodes = pd.read_csv(args.raw_label_dir / region_name / "node_labels.csv")
    candidate_lookup = build_candidate_lookup(raw_nodes)

    sums = empty_sums()
    split_counts = {name: 0 for name in ("train", "val", "test")}
    pending: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed + region_code)
    events = events.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    for row in events.itertuples(index=False):
        if all(count >= args.max_samples_per_split for count in split_counts.values()):
            break
        incident_ids = parse_incident_ids(row.incident_ids)
        candidates = select_candidate_nodes(
            incident_ids=incident_ids,
            candidate_lookup=candidate_lookup,
            anchor_region_idx=int(row.anchor_region_idx),
            max_nodes=args.max_candidate_nodes,
        )
        node_idx, node_valid, node_affected, _node_ctx = build_node_context(
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
            if split_counts[sample_split] >= args.max_samples_per_split:
                continue
            input_idx = np.arange(input_start, sample_start, dtype=np.int32)
            future_idx = np.arange(sample_start, future_end, dtype=np.int32)
            hist = traffic[input_idx][:, node_idx, :]
            future_base = baseline[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
            fut_scale = scale[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
            actual_future = traffic[future_idx][:, node_idx, :]
            last_obs = traffic[sample_start - 1, node_idx, :]
            blend = np.empty_like(actual_future, dtype=np.float32)
            for h in range(args.horizon_steps):
                blend[h] = future_base[h] + alphas[h][None, :] * (last_obs - future_base[h])
            blend[~np.isfinite(blend)] = 0.0

            pending.append(
                {
                    "hist": hist.astype(np.float32),
                    "blend": blend.astype(np.float32),
                    "actual": actual_future.astype(np.float32),
                    "scale": fut_scale.astype(np.float32),
                    "node_idx": node_idx.astype(np.int32),
                    "node_valid": node_valid.astype(np.float32),
                    "node_affected": node_affected.astype(np.float32),
                    "sample_start": sample_start,
                }
            )
            split_counts[sample_split] += 1
            if len(pending) >= args.batch_size:
                flush_batch(pending, learned_normal, traffic, baseline, day_kind, tod, alphas, times, args, sums)

    flush_batch(pending, learned_normal, traffic, baseline, day_kind, tod, alphas, times, args, sums)
    row = {"region": region_name, **split_counts, **finalize_sums(sums)}
    print(
        f"[{region_name}] samples={sum(split_counts.values())} "
        f"all_diff={row['all_normal_diff_robust_mae']:.4f} "
        f"affected_diff={row['affected_normal_diff_robust_mae']:.4f}",
        flush=True,
    )
    return row


def write_report(output_dir: Path, rows: list[dict[str, float]]) -> None:
    df = pd.DataFrame(rows)
    weighted: dict[str, float | str] = {"region": "weighted"}
    for subset in SUBSETS:
        count_col = f"{subset}_values"
        count = float(df[count_col].sum())
        weighted[count_col] = count
        for metric in [
            "normal_diff_raw_mae",
            "normal_diff_robust_mae",
            "local_target_robust_mae",
            "full_target_robust_mae",
        ]:
            col = f"{subset}_{metric}"
            weighted[col] = float((df[col] * df[count_col]).sum() / max(count, 1.0))
        local_target = float(weighted[f"{subset}_local_target_robust_mae"])
        full_target = float(weighted[f"{subset}_full_target_robust_mae"])
        weighted[f"{subset}_target_change_pct"] = (
            100.0 * (full_target - local_target) / local_target if local_target > 0 else float("nan")
        )
    out_df = pd.concat([df, pd.DataFrame([weighted])], ignore_index=True)
    out_df.to_csv(output_dir / "full_region_normal_diagnostics.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": out_df.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)

    keep_cols = [
        "region",
        "train",
        "val",
        "test",
        "all_normal_diff_robust_mae",
        "all_local_target_robust_mae",
        "all_full_target_robust_mae",
        "all_target_change_pct",
        "affected_normal_diff_robust_mae",
        "affected_local_target_robust_mae",
        "affected_full_target_robust_mae",
        "affected_target_change_pct",
    ]
    lines = ["# Full-Region Normal Inference Diagnostic", ""]
    lines.append("This compares candidate-subgraph learned-normal inference against full-region inference sliced back to the same candidate nodes.")
    lines.append("")
    lines.append(out_df[keep_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `normal_diff_robust_mae` measures how much the normal forecast changes when using the full region graph.")
    lines.append("- `target_change_pct` measures how the incident residual target magnitude changes under full-region normal inference.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_sensor_meta(args.data_dir.resolve())
    inc = load_incidents_2023(args.data_dir.resolve())
    rows = [run_region(region, code, args, meta, inc) for code, region in enumerate(args.regions)]
    write_report(output_dir, rows)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
