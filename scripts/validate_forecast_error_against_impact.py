#!/usr/bin/env python3
"""Validate whether impact labels explain forecast difficulty.

This script trains a lightweight "normal-only" forecaster from non-incident
windows and tests whether incident-window forecast errors correlate with the
derived impact labels more strongly than with incident type.

The forecaster is intentionally simple and transparent:
1. Build a robust normal baseline from train-split non-incident windows.
2. Learn a per-horizon/per-channel blend weight between:
   - the future-slot normal baseline, and
   - the most recent observation before the forecast window.
3. Evaluate event-window forecast error on the top-k impacted local nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


CHANNELS = ("flow", "occupancy", "speed")


def finite_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive"),
        help="Path to the XTraffic archive directory.",
    )
    parser.add_argument(
        "--raw-label-dir",
        type=Path,
        default=Path("outputs/impact_labels"),
        help="Directory produced by build_impact_labels.py.",
    )
    parser.add_argument(
        "--event-root",
        type=Path,
        default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"),
        help="Directory containing aggregated <Region>/event_labels.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/forecast_error_validation/region_area_sensor_window"),
        help="Directory for validation outputs.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["Alameda", "ContraCosta", "Orange"],
        help="Region names to process.",
    )
    parser.add_argument(
        "--candidate-pm-radius",
        type=float,
        default=5.0,
        help="Must match the raw impact-label build setting.",
    )
    parser.add_argument(
        "--anchor-pm-radius",
        type=float,
        default=2.0,
        help="Must match the raw impact-label build setting.",
    )
    parser.add_argument(
        "--baseline-mask-extra-steps",
        type=int,
        default=12,
        help="Extra masked steps after raw incident duration.",
    )
    parser.add_argument(
        "--min-baseline-count",
        type=int,
        default=8,
        help="Minimum valid train samples for a baseline slot.",
    )
    parser.add_argument(
        "--input-steps",
        type=int,
        default=12,
        help="History length. The lightweight forecaster uses the last observed step.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=12,
        help="Forecast horizon in 5-minute steps.",
    )
    parser.add_argument(
        "--local-topk-nodes",
        type=int,
        default=5,
        help="Evaluate on the top-k impacted local nodes per event.",
    )
    parser.add_argument(
        "--normal-control-samples",
        type=int,
        default=2000,
        help="Maximum sampled non-incident test windows per region for control statistics.",
    )
    return parser.parse_args()


def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    sub = df[[group_col, value_col]].dropna()
    if sub.empty:
        return float("nan")
    values = sub[value_col].to_numpy(float)
    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in sub.groupby(group_col):
        vals = group[value_col].to_numpy(float)
        between += len(vals) * (vals.mean() - values.mean()) ** 2
    return float(between / total)


def spearman_pair(df: pd.DataFrame, left: str, right: str) -> float:
    if left not in df.columns or right not in df.columns:
        return float("nan")
    sub = df[[left, right]].dropna()
    if len(sub) < 2:
        return float("nan")
    return float(sub.corr(method="spearman").iloc[0, 1])


def make_split_bounds(total_steps: int) -> dict[str, tuple[int, int]]:
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_steps),
    }


def fit_blend_alphas(
    traffic: np.ndarray,
    times: pd.DatetimeIndex,
    train_valid: np.ndarray,
    baseline: np.ndarray,
    input_steps: int,
    horizon_steps: int,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Learn blend weights for pred = base + alpha * (last_obs - base)."""
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)

    total_steps = traffic.shape[0]
    n_channels = traffic.shape[2]
    alphas = np.zeros((horizon_steps, n_channels), dtype=np.float32)

    start_candidates = np.arange(input_steps, total_steps - horizon_steps + 1, dtype=np.int32)
    keep = start_candidates < (int(total_steps * 0.70) - horizon_steps + 1)
    starts = start_candidates[keep]

    for h in range(horizon_steps):
        numer = np.zeros(n_channels, dtype=np.float64)
        denom = np.zeros(n_channels, dtype=np.float64)
        for chunk_start in range(0, starts.size, chunk_size):
            batch = starts[chunk_start : chunk_start + chunk_size]
            future_idx = batch + h
            base_h = baseline[day_kind[future_idx], tod[future_idx]]
            prev_obs = traffic[batch - 1]
            target = traffic[future_idx]
            valid = train_valid[batch - 1] & train_valid[future_idx]
            for c in range(n_channels):
                delta = prev_obs[:, :, c] - base_h[:, :, c]
                target_delta = target[:, :, c] - base_h[:, :, c]
                mask = valid & np.isfinite(delta) & np.isfinite(target_delta)
                if not np.any(mask):
                    continue
                x = delta[mask]
                y = target_delta[mask]
                numer[c] += float(np.dot(x, y))
                denom[c] += float(np.dot(x, x))
        for c in range(n_channels):
            if denom[c] <= 1e-8:
                alpha = 0.0
            else:
                alpha = float(numer[c] / denom[c])
            alphas[h, c] = np.clip(alpha, 0.0, 1.0)
    return alphas


def parse_incident_ids(text: object) -> list[str]:
    if pd.isna(text):
        return []
    out = []
    for item in str(text).split(";"):
        item = item.strip()
        if item:
            out.append(item)
    return out


def build_node_lookup(raw_nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    cols = ["region_node_idx", "any_z_auc", "affected"]
    for incident_id, group in raw_nodes.groupby("incident_id", sort=False):
        lookup[str(incident_id)] = group[cols].to_numpy()
    return lookup


def select_local_nodes(
    incident_ids: list[str],
    node_lookup: dict[str, np.ndarray],
    anchor_region_idx: int,
    topk: int,
) -> np.ndarray:
    pieces = []
    for incident_id in incident_ids:
        arr = node_lookup.get(incident_id)
        if arr is not None and len(arr):
            pieces.append(arr)
    if not pieces:
        return np.asarray([anchor_region_idx], dtype=np.int32)

    merged = np.vstack(pieces)
    df = pd.DataFrame(merged, columns=["region_node_idx", "any_z_auc", "affected"])
    df["region_node_idx"] = df["region_node_idx"].astype(int)
    df["any_z_auc"] = pd.to_numeric(df["any_z_auc"], errors="coerce")
    df["affected"] = pd.to_numeric(df["affected"], errors="coerce").fillna(0).astype(int)
    by_node = (
        df.groupby("region_node_idx", as_index=False)
        .agg(any_z_auc=("any_z_auc", "max"), affected=("affected", "max"))
        .sort_values(["affected", "any_z_auc"], ascending=[False, False])
    )
    node_idx = by_node["region_node_idx"].head(topk).to_numpy(dtype=np.int32)
    if node_idx.size == 0:
        return np.asarray([anchor_region_idx], dtype=np.int32)
    return node_idx


def predict_event_window(
    traffic: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
    alphas: np.ndarray,
    start_idx: int,
    node_idx: np.ndarray,
    horizon_steps: int,
    anchor_idx: int,
) -> dict[str, float] | None:
    total_steps = traffic.shape[0]
    if start_idx <= 0 or start_idx + horizon_steps > total_steps:
        return None

    future_idx = np.arange(start_idx, start_idx + horizon_steps, dtype=np.int32)
    base = baseline[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
    scl = scale[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
    last_obs = traffic[start_idx - 1, node_idx, :]
    actual = traffic[future_idx][:, node_idx, :]

    pred = np.empty_like(actual, dtype=np.float32)
    for h in range(horizon_steps):
        pred[h] = base[h] + alphas[h][None, :] * (last_obs - base[h])

    abs_err = np.abs(pred - actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        robust_err = abs_err / scl
    robust_err[~np.isfinite(robust_err)] = np.nan

    metrics: dict[str, float] = {
        "local_mae_all": finite_mean(abs_err),
        "local_robust_mae_all": finite_mean(robust_err),
    }
    for c, name in enumerate(CHANNELS):
        metrics[f"local_mae_{name}"] = finite_mean(abs_err[:, :, c])
        metrics[f"local_robust_mae_{name}"] = finite_mean(robust_err[:, :, c])

    anchor_arr = np.asarray([anchor_idx], dtype=np.int32)
    anchor_base = baseline[day_kind[future_idx], tod[future_idx]][:, anchor_arr, :]
    anchor_scl = scale[day_kind[future_idx], tod[future_idx]][:, anchor_arr, :]
    anchor_last_obs = traffic[start_idx - 1, anchor_arr, :]
    anchor_actual = traffic[future_idx][:, anchor_arr, :]
    anchor_pred = np.empty_like(anchor_actual, dtype=np.float32)
    for h in range(horizon_steps):
        anchor_pred[h] = anchor_base[h] + alphas[h][None, :] * (anchor_last_obs - anchor_base[h])
    anchor_abs_err = np.abs(anchor_pred - anchor_actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        anchor_robust_err = anchor_abs_err / anchor_scl
    anchor_robust_err[~np.isfinite(anchor_robust_err)] = np.nan

    metrics["anchor_mae_all"] = finite_mean(anchor_abs_err)
    metrics["anchor_robust_mae_all"] = finite_mean(anchor_robust_err)
    metrics["eval_nodes_used"] = int(len(node_idx))
    return metrics


def sample_normal_control(
    traffic: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
    alphas: np.ndarray,
    baseline_valid: np.ndarray,
    split_bounds: tuple[int, int],
    horizon_steps: int,
    input_steps: int,
    max_samples: int,
) -> pd.DataFrame:
    start_lo, start_hi = split_bounds
    candidate_starts = np.arange(max(start_lo, input_steps), start_hi - horizon_steps + 1, dtype=np.int32)
    if candidate_starts.size == 0:
        return pd.DataFrame()

    mid_node = int(traffic.shape[1] // 2)
    valid_start = baseline_valid[candidate_starts - 1, mid_node]
    future_mask = np.ones(candidate_starts.size, dtype=bool)
    for h in range(horizon_steps):
        future_mask &= baseline_valid[candidate_starts + h, mid_node]
    starts = candidate_starts[valid_start & future_mask]
    if starts.size == 0:
        return pd.DataFrame()

    if starts.size > max_samples:
        rng = np.random.default_rng(7)
        starts = np.sort(rng.choice(starts, size=max_samples, replace=False))

    rows = []
    for start_idx in starts:
        metrics = predict_event_window(
            traffic=traffic,
            day_kind=day_kind,
            tod=tod,
            baseline=baseline,
            scale=scale,
            alphas=alphas,
            start_idx=int(start_idx),
            node_idx=np.asarray([mid_node], dtype=np.int32),
            horizon_steps=horizon_steps,
            anchor_idx=mid_node,
        )
        if metrics is None:
            continue
        if not np.isfinite(metrics["local_robust_mae_all"]):
            continue
        rows.append(metrics)
    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        ("severity_any_z_auc_topk", "Severity"),
        ("recovery_time_min", "Recovery time (min)"),
        ("spread_nodes", "Spread nodes"),
    ]
    for ax, (col, label) in zip(axes, pairs):
        ax.scatter(df[col], df["local_robust_mae_all"], s=8, alpha=0.25)
        ax.set_xlabel(label)
        ax.set_ylabel("Local robust MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "impact_vs_error_scatter.png", dpi=180)
    plt.close(fig)


def save_type_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["primary_type"].value_counts()
    keep = counts[counts >= 100].index.tolist()
    plot_df = df[df["primary_type"].isin(keep)].copy()
    if plot_df.empty:
        return
    order = (
        plot_df.groupby("primary_type")["local_robust_mae_all"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [
        plot_df.loc[plot_df["primary_type"] == name, "local_robust_mae_all"].dropna().to_numpy()
        for name in order
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("local_robust_mae_all")
    ax.set_title("Forecast error overlap across incident types")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_type_boxplot.png", dpi=180)
    plt.close(fig)


def save_severity_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["weak", "mild", "moderate", "severe"]
    data = [
        df.loc[df["severity_class_z"] == name, "local_robust_mae_all"].dropna().to_numpy()
        for name in order
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("local_robust_mae_all")
    ax.set_title("Forecast error by impact severity class")
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_severity_class.png", dpi=180)
    plt.close(fig)


def write_report(
    event_df: pd.DataFrame,
    region_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]],
    alphas_by_region: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    lines = ["# Forecast Error vs Impact Validation", ""]
    lines.append(f"- events evaluated: {len(event_df)}")
    lines.append("")

    if control_rows:
        control_df = pd.DataFrame(control_rows)
        lines.append("## Incident vs Normal Control")
        lines.append("")
        lines.append(
            f"- incident local_robust_mae_all median: {event_df['local_robust_mae_all'].median():.3f}"
        )
        lines.append(
            f"- normal control local_robust_mae_all median: {control_df['local_robust_mae_all'].median():.3f}"
        )
        lines.append(
            f"- incident anchor_robust_mae_all median: {event_df['anchor_robust_mae_all'].median():.3f}"
        )
        lines.append(
            f"- normal control anchor_robust_mae_all median: {control_df['anchor_robust_mae_all'].median():.3f}"
        )
        lines.append("")

    lines.append("## Regional Summary")
    lines.append("")
    lines.append(pd.DataFrame(region_rows).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    metrics = []
    for col in ["severity_any_z_auc_topk", "recovery_time_min", "spread_nodes", "spread_pm"]:
        metrics.append(
            {
                "pair": f"local_robust_mae_all vs {col}",
                "spearman": spearman_pair(event_df, "local_robust_mae_all", col),
            }
        )
    lines.append("## Error Correlation")
    lines.append("")
    lines.append(pd.DataFrame(metrics).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    eta_rows = [
        {
            "target": "local_robust_mae_all",
            "eta_squared_by_primary_type": eta_squared(event_df, "primary_type", "local_robust_mae_all"),
            "eta_squared_by_severity_class": eta_squared(
                event_df, "severity_class_z", "local_robust_mae_all"
            ),
            "eta_squared_by_region": eta_squared(event_df, "region", "local_robust_mae_all"),
        }
    ]
    lines.append("## Effect Size")
    lines.append("")
    lines.append(pd.DataFrame(eta_rows).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Learned Blend Weights")
    lines.append("")
    for region, alphas in alphas_by_region.items():
        alpha_df = pd.DataFrame(alphas, columns=[f"{name}_alpha" for name in CHANNELS])
        alpha_df.insert(0, "horizon_step", np.arange(1, len(alpha_df) + 1))
        lines.append(f"### {region}")
        lines.append("")
        lines.append(alpha_df.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    raw_label_dir = args.raw_label_dir.resolve()
    event_root = args.event_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = region_specs()
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)

    all_event_rows: list[pd.DataFrame] = []
    region_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    alphas_by_region: dict[str, np.ndarray] = {}

    for region_name in args.regions:
        region = specs[region_name]
        print(f"[{region_name}] loading traffic and labels", flush=True)
        region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
        region_meta = region_meta.reset_index(drop=True)
        region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)
        traffic, times = load_region_traffic(data_dir, region_node_idx)
        day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
        tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
        split = make_split_bounds(traffic.shape[0])

        print(f"[{region_name}] rebuilding incident mask for normal-only training", flush=True)
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
        train_valid[split["train"][1] :, :] = False

        print(f"[{region_name}] fitting train baseline", flush=True)
        baseline, scale, _ = build_robust_baseline(
            traffic=traffic,
            times=times,
            baseline_valid=train_valid,
            min_count=args.min_baseline_count,
        )

        print(f"[{region_name}] learning blend weights", flush=True)
        alphas = fit_blend_alphas(
            traffic=traffic,
            times=times,
            train_valid=train_valid,
            baseline=baseline,
            input_steps=args.input_steps,
            horizon_steps=args.horizon_steps,
        )
        alphas_by_region[region_name] = alphas

        print(f"[{region_name}] preparing event/node tables", flush=True)
        agg_events = pd.read_csv(event_root / region_name / "event_labels.csv")
        raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
        raw_nodes["incident_id"] = raw_nodes["incident_id"].astype(str)
        node_lookup = build_node_lookup(raw_nodes)

        test_lo, test_hi = split["test"]
        agg_events = agg_events[
            (agg_events["start_idx"] >= test_lo)
            & (agg_events["start_idx"] + args.horizon_steps <= test_hi)
        ].copy()

        print(f"[{region_name}] evaluating {len(agg_events)} test events", flush=True)
        event_rows = []
        for row in agg_events.itertuples(index=False):
            incident_ids = parse_incident_ids(row.incident_ids)
            node_idx = select_local_nodes(
                incident_ids=incident_ids,
                node_lookup=node_lookup,
                anchor_region_idx=int(row.anchor_region_idx),
                topk=args.local_topk_nodes,
            )
            metrics = predict_event_window(
                traffic=traffic,
                day_kind=day_kind,
                tod=tod,
                baseline=baseline,
                scale=scale,
                alphas=alphas,
                start_idx=int(row.start_idx),
                node_idx=node_idx,
                horizon_steps=args.horizon_steps,
                anchor_idx=int(row.anchor_region_idx),
            )
            if metrics is None:
                continue
            if not np.isfinite(metrics["local_robust_mae_all"]):
                continue
            event_row = row._asdict()
            event_row.update(metrics)
            event_rows.append(event_row)

        region_event_df = pd.DataFrame(event_rows)
        region_out = output_dir / region_name
        region_out.mkdir(parents=True, exist_ok=True)
        region_event_df.to_csv(region_out / "event_error_labels.csv", index=False)
        all_event_rows.append(region_event_df)

        control_df = sample_normal_control(
            traffic=traffic,
            day_kind=day_kind,
            tod=tod,
            baseline=baseline,
            scale=scale,
            alphas=alphas,
            baseline_valid=baseline_valid,
            split_bounds=split["test"],
            horizon_steps=args.horizon_steps,
            input_steps=args.input_steps,
            max_samples=args.normal_control_samples,
        )
        if not control_df.empty:
            control_df["region"] = region_name
            control_rows.extend(control_df.to_dict(orient="records"))
            control_df.to_csv(region_out / "normal_control_errors.csv", index=False)

        region_rows.append(
            {
                "region": region_name,
                "events": len(region_event_df),
                "local_robust_mae_p50": region_event_df["local_robust_mae_all"].median(),
                "anchor_robust_mae_p50": region_event_df["anchor_robust_mae_all"].median(),
                "severity_error_spearman": spearman_pair(
                    region_event_df, "severity_any_z_auc_topk", "local_robust_mae_all"
                ),
                "recovery_error_spearman": spearman_pair(
                    region_event_df, "recovery_time_min", "local_robust_mae_all"
                ),
                "spread_error_spearman": spearman_pair(
                    region_event_df, "spread_nodes", "local_robust_mae_all"
                ),
                "type_eta_squared": eta_squared(
                    region_event_df, "primary_type", "local_robust_mae_all"
                ),
            }
        )

    event_df = pd.concat(all_event_rows, ignore_index=True)
    event_df.to_csv(output_dir / "all_event_error_labels.csv", index=False)
    save_scatter(event_df, output_dir)
    save_type_boxplot(event_df, output_dir)
    save_severity_boxplot(event_df, output_dir)
    write_report(
        event_df=event_df,
        region_rows=region_rows,
        control_rows=control_rows,
        alphas_by_region=alphas_by_region,
        output_dir=output_dir,
    )

    config = {
        "data_dir": str(data_dir),
        "raw_label_dir": str(raw_label_dir),
        "event_root": str(event_root),
        "regions": args.regions,
        "input_steps": args.input_steps,
        "horizon_steps": args.horizon_steps,
        "local_topk_nodes": args.local_topk_nodes,
        "normal_control_samples": args.normal_control_samples,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"wrote validation outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
