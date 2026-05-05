#!/usr/bin/env python3
"""Build incident impact labels for XTraffic regional subsets.

This script implements the first-stage labeling pipeline:
1. Select the 2023 Mainline sensors for Alameda, Contra Costa, and Orange.
2. Match incidents to same-freeway/same-direction candidate sensors by Abs PM.
3. Build a counterfactual normal baseline from non-incident windows.
4. Derive channel-wise impact labels plus recovery, spread, and directionality.

The script intentionally keeps flow/speed/occupancy impacts separate. Scalar
labels such as recovery and spread use robust z-scores so that no hand-picked
flow/speed/occupancy fusion weights are required.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHANNELS = ("flow", "occupancy", "speed")
CHANNEL_FLOW = 0
CHANNEL_OCC = 1
CHANNEL_SPEED = 2


@dataclass(frozen=True)
class RegionSpec:
    name: str
    county: str


@dataclass
class IncidentMatch:
    incident_id: str
    row_idx: int
    start_idx: int
    duration_min: float
    duration_steps_for_mask: int
    incident_type: str
    area: str
    description: str
    freeway: float
    direction: str
    abs_pm: float
    latitude: float
    longitude: float
    anchor_region_idx: int
    anchor_sensor_id: int
    anchor_pm_dist: float
    candidate_region_idx: np.ndarray
    candidate_pm_dist: np.ndarray
    candidate_side: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive"),
        help="Path to the XTraffic archive directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_labels"),
        help="Directory for generated CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["Alameda", "ContraCosta", "Orange"],
        help="Region names to process: Alameda ContraCosta Orange.",
    )
    parser.add_argument(
        "--candidate-pm-radius",
        type=float,
        default=5.0,
        help="Abs PM radius for candidate affected sensors.",
    )
    parser.add_argument(
        "--anchor-pm-radius",
        type=float,
        default=2.0,
        help="Maximum Abs PM distance for accepting an anchor sensor.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=36,
        help="Post-incident observation horizon in 5-minute steps.",
    )
    parser.add_argument(
        "--baseline-mask-extra-steps",
        type=int,
        default=12,
        help="Extra steps after raw duration to exclude from baseline.",
    )
    parser.add_argument(
        "--topk-nodes",
        type=int,
        default=5,
        help="Top-k candidate nodes used for event-level aggregation.",
    )
    parser.add_argument(
        "--affected-z",
        type=float,
        default=3.0,
        help="Robust z-score threshold for node affected mask.",
    )
    parser.add_argument(
        "--recovery-z",
        type=float,
        default=2.0,
        help="Robust z-score threshold for recovery.",
    )
    parser.add_argument(
        "--recovery-consecutive-steps",
        type=int,
        default=3,
        help="Consecutive recovered steps required for recovery time.",
    )
    parser.add_argument(
        "--min-baseline-count",
        type=int,
        default=8,
        help="Minimum non-incident samples for a baseline group.",
    )
    parser.add_argument(
        "--max-relative-impact",
        type=float,
        default=3.0,
        help="Upper cap for relative channel impacts.",
    )
    parser.add_argument(
        "--min-valid-impact-steps",
        type=int,
        default=12,
        help="Minimum finite post-incident steps required for node-level impact labels.",
    )
    parser.add_argument(
        "--skip-node-labels",
        action="store_true",
        help="Only write event-level labels.",
    )
    return parser.parse_args()


def region_specs() -> dict[str, RegionSpec]:
    return {
        "Alameda": RegionSpec(name="Alameda", county="Alameda"),
        "ContraCosta": RegionSpec(name="ContraCosta", county="Contra Costa"),
        "Orange": RegionSpec(name="Orange", county="Orange"),
    }


def topk_mean(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    k = min(k, values.size)
    if k <= 0:
        return float("nan")
    part = np.partition(values, values.size - k)[values.size - k :]
    return float(np.mean(part))


def safe_nanmean(values: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def safe_nanmax(values: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(values, axis=axis)


def normalize_direction(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()[:1]


def direction_side(sensor_pm: np.ndarray, incident_pm: float, direction: str) -> np.ndarray:
    """Return -1 upstream, +1 downstream, 0 near/equal.

    Caltrans postmiles generally increase in the N/E direction. Vehicles moving
    N/E come from smaller postmiles, while vehicles moving S/W come from larger
    postmiles.
    """
    delta = sensor_pm - incident_pm
    side = np.zeros_like(delta, dtype=np.int8)
    tol = 1e-6
    if direction in {"N", "E"}:
        side[delta < -tol] = -1
        side[delta > tol] = 1
    elif direction in {"S", "W"}:
        side[delta > tol] = -1
        side[delta < -tol] = 1
    else:
        side[delta < -tol] = -1
        side[delta > tol] = 1
    return side


def load_sensor_meta(data_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv", sep="\t")
    node_order = np.load(data_dir / "node_order.npy")
    ordered = meta.set_index("station_id").loc[node_order].reset_index()
    ordered = ordered.rename(columns={"index": "station_id"})
    ordered["node_idx"] = np.arange(len(ordered), dtype=np.int32)
    ordered["Direction"] = ordered["Direction"].map(normalize_direction)
    ordered["Fwy"] = pd.to_numeric(ordered["Fwy"], errors="coerce")
    ordered["Abs PM"] = pd.to_numeric(ordered["Abs PM"], errors="coerce")
    ordered["Lat"] = pd.to_numeric(ordered["Lat"], errors="coerce")
    ordered["Lng"] = pd.to_numeric(ordered["Lng"], errors="coerce")
    return ordered


def load_incidents_2023(data_dir: Path) -> pd.DataFrame:
    inc = pd.read_csv(data_dir / "incidents_y2023.csv", sep="\t", low_memory=False)
    inc["incident_id"] = inc["incident_id"].astype(str)
    inc["dt_parsed"] = pd.to_datetime(inc["dt"], errors="coerce")
    inc["duration"] = pd.to_numeric(inc["duration"], errors="coerce")
    inc["Abs PM"] = pd.to_numeric(inc["Abs PM"], errors="coerce")
    inc["Fwy"] = pd.to_numeric(inc["Fwy"], errors="coerce")
    inc["Latitude"] = pd.to_numeric(inc["Latitude"], errors="coerce")
    inc["Longitude"] = pd.to_numeric(inc["Longitude"], errors="coerce")
    inc["Freeway_direction"] = inc["Freeway_direction"].map(normalize_direction)
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-01-01")
    inc = inc[(inc["dt_parsed"] >= start) & (inc["dt_parsed"] < end)].copy()
    inc = inc.dropna(subset=["dt_parsed", "Abs PM", "Fwy"])
    return inc.reset_index(drop=True)


def load_region_traffic(data_dir: Path, region_node_idx: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    monthly = []
    for month in range(1, 13):
        arr = np.load(data_dir / f"p{month:02d}_done.npy", mmap_mode="r")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unexpected traffic shape for p{month:02d}: {arr.shape}")
        monthly.append(np.asarray(arr[:, region_node_idx, :], dtype=np.float32))
    traffic = np.concatenate(monthly, axis=0)
    times = pd.date_range("2023-01-01", "2024-01-01", freq="5min", inclusive="left")
    if len(times) != traffic.shape[0]:
        raise ValueError(f"Expected {len(times)} slots, got {traffic.shape[0]}")
    return traffic, times


def build_sensor_groups(region_meta: pd.DataFrame) -> dict[tuple[float, str], np.ndarray]:
    groups: dict[tuple[float, str], list[int]] = {}
    for region_idx, row in region_meta.iterrows():
        fwy = row["Fwy"]
        direction = row["Direction"]
        abs_pm = row["Abs PM"]
        if not np.isfinite(fwy) or not np.isfinite(abs_pm) or not direction:
            continue
        key = (float(fwy), direction)
        groups.setdefault(key, []).append(int(region_idx))
    return {key: np.asarray(idx, dtype=np.int32) for key, idx in groups.items()}


def build_matches(
    inc: pd.DataFrame,
    region_meta: pd.DataFrame,
    times: pd.DatetimeIndex,
    candidate_pm_radius: float,
    anchor_pm_radius: float,
    baseline_mask_extra_steps: int,
) -> list[IncidentMatch]:
    groups = build_sensor_groups(region_meta)
    start_time = times[0]
    matches: list[IncidentMatch] = []
    abs_pm_arr = region_meta["Abs PM"].to_numpy(dtype=np.float64)
    sensor_id_arr = region_meta["station_id"].to_numpy(dtype=np.int64)

    for row_idx, row in inc.iterrows():
        direction = normalize_direction(row["Freeway_direction"])
        if not direction or not np.isfinite(row["Fwy"]) or not np.isfinite(row["Abs PM"]):
            continue
        key = (float(row["Fwy"]), direction)
        sensor_idx = groups.get(key)
        if sensor_idx is None or sensor_idx.size == 0:
            continue

        pm = float(row["Abs PM"])
        pm_dist_all = np.abs(abs_pm_arr[sensor_idx] - pm)
        within = pm_dist_all <= candidate_pm_radius
        if not np.any(within):
            continue
        candidate_idx = sensor_idx[within]
        candidate_pm_dist = pm_dist_all[within]
        anchor_pos = int(np.argmin(candidate_pm_dist))
        anchor_pm_dist = float(candidate_pm_dist[anchor_pos])
        if anchor_pm_dist > anchor_pm_radius:
            continue

        dt = row["dt_parsed"]
        start_idx = int((dt - start_time).total_seconds() // 300)
        if start_idx < 0 or start_idx >= len(times):
            continue

        duration_min = float(row["duration"]) if np.isfinite(row["duration"]) else float("nan")
        if np.isfinite(duration_min) and duration_min > 0:
            duration_steps = max(1, int(math.ceil(duration_min / 5.0)))
        else:
            duration_steps = 12
        duration_steps_for_mask = max(1, duration_steps + baseline_mask_extra_steps)

        candidate_pm = abs_pm_arr[candidate_idx]
        side = direction_side(candidate_pm, pm, direction)
        matches.append(
            IncidentMatch(
                incident_id=str(row["incident_id"]),
                row_idx=int(row_idx),
                start_idx=start_idx,
                duration_min=duration_min,
                duration_steps_for_mask=duration_steps_for_mask,
                incident_type=str(row["Type"]) if not pd.isna(row["Type"]) else "",
                area=str(row["AREA"]) if not pd.isna(row["AREA"]) else "",
                description=str(row["DESCRIPTION"]) if not pd.isna(row["DESCRIPTION"]) else "",
                freeway=float(row["Fwy"]),
                direction=direction,
                abs_pm=pm,
                latitude=float(row["Latitude"]) if np.isfinite(row["Latitude"]) else float("nan"),
                longitude=float(row["Longitude"]) if np.isfinite(row["Longitude"]) else float("nan"),
                anchor_region_idx=int(candidate_idx[anchor_pos]),
                anchor_sensor_id=int(sensor_id_arr[candidate_idx[anchor_pos]]),
                anchor_pm_dist=anchor_pm_dist,
                candidate_region_idx=candidate_idx.astype(np.int32),
                candidate_pm_dist=candidate_pm_dist.astype(np.float32),
                candidate_side=side,
            )
        )
    return matches


def build_baseline_valid_mask(
    shape: tuple[int, int],
    matches: Iterable[IncidentMatch],
) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    total_steps = shape[0]
    for match in matches:
        start = match.start_idx
        end = min(total_steps, start + match.duration_steps_for_mask)
        if end <= start:
            continue
        valid[start:end, match.candidate_region_idx] = False
    return valid


def build_robust_baseline(
    traffic: np.ndarray,
    times: pd.DatetimeIndex,
    baseline_valid: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median, robust scale, and valid counts by day-kind/time-of-day."""
    n_nodes = traffic.shape[1]
    n_channels = traffic.shape[2]
    baseline = np.full((2, 288, n_nodes, n_channels), np.nan, dtype=np.float32)
    scale = np.full_like(baseline, np.nan)
    counts = np.zeros((2, 288, n_nodes, n_channels), dtype=np.int16)

    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)

    for dk in (0, 1):
        for slot in range(288):
            idx = np.flatnonzero((day_kind == dk) & (tod == slot))
            if idx.size == 0:
                continue
            vals = traffic[idx].copy()
            vals[~baseline_valid[idx], :] = np.nan
            finite = np.isfinite(vals)
            cnt = finite.sum(axis=0)
            counts[dk, slot] = np.minimum(cnt, np.iinfo(np.int16).max)
            insufficient = cnt < min_count
            vals[:, insufficient] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(vals, axis=0)
            baseline[dk, slot] = med.astype(np.float32)

            diff = np.abs(vals - med[None, :, :])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mad = np.nanmedian(diff, axis=0)
            robust_scale = (1.4826 * mad).astype(np.float32)

            # Channel-specific lower bounds avoid exploding z-scores in very
            # stable low-variance slots.
            robust_scale[:, CHANNEL_FLOW] = np.maximum(robust_scale[:, CHANNEL_FLOW], 3.0)
            robust_scale[:, CHANNEL_OCC] = np.maximum(robust_scale[:, CHANNEL_OCC], 0.005)
            robust_scale[:, CHANNEL_SPEED] = np.maximum(robust_scale[:, CHANNEL_SPEED], 2.0)
            scale[dk, slot] = robust_scale

    return baseline, scale, counts


def relative_impacts(
    actual: np.ndarray,
    baseline: np.ndarray,
    max_relative_impact: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps_flow = 5.0
    eps_occ = 0.005
    eps_speed = 2.0

    flow = np.clip(
        (baseline[:, :, CHANNEL_FLOW] - actual[:, :, CHANNEL_FLOW])
        / np.maximum(baseline[:, :, CHANNEL_FLOW], eps_flow),
        0,
        max_relative_impact,
    )
    speed = np.clip(
        (baseline[:, :, CHANNEL_SPEED] - actual[:, :, CHANNEL_SPEED])
        / np.maximum(baseline[:, :, CHANNEL_SPEED], eps_speed),
        0,
        max_relative_impact,
    )
    occ = np.clip(
        (actual[:, :, CHANNEL_OCC] - baseline[:, :, CHANNEL_OCC])
        / np.maximum(baseline[:, :, CHANNEL_OCC], eps_occ),
        0,
        max_relative_impact,
    )

    flow[~np.isfinite(flow)] = np.nan
    speed[~np.isfinite(speed)] = np.nan
    occ[~np.isfinite(occ)] = np.nan
    return flow.astype(np.float32), speed.astype(np.float32), occ.astype(np.float32)


def robust_z_impacts(
    actual: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flow_z = np.maximum(
        0,
        (baseline[:, :, CHANNEL_FLOW] - actual[:, :, CHANNEL_FLOW])
        / scale[:, :, CHANNEL_FLOW],
    )
    speed_z = np.maximum(
        0,
        (baseline[:, :, CHANNEL_SPEED] - actual[:, :, CHANNEL_SPEED])
        / scale[:, :, CHANNEL_SPEED],
    )
    occ_z = np.maximum(
        0,
        (actual[:, :, CHANNEL_OCC] - baseline[:, :, CHANNEL_OCC])
        / scale[:, :, CHANNEL_OCC],
    )
    for arr in (flow_z, speed_z, occ_z):
        arr[~np.isfinite(arr)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        any_z = np.nanmax(np.stack([flow_z, speed_z, occ_z], axis=0), axis=0)
    any_z[~np.isfinite(any_z)] = np.nan
    return (
        flow_z.astype(np.float32),
        speed_z.astype(np.float32),
        occ_z.astype(np.float32),
        any_z.astype(np.float32),
    )


def find_recovery_time(
    event_curve: np.ndarray,
    recovery_z: float,
    consecutive_steps: int,
) -> tuple[float, int]:
    if event_curve.size < consecutive_steps:
        return float(event_curve.size * 5), 1
    for start in range(0, event_curve.size - consecutive_steps + 1):
        window = event_curve[start : start + consecutive_steps]
        if np.all(np.isfinite(window)) and np.all(window < recovery_z):
            return float(start * 5), 0
    return float(event_curve.size * 5), 1


def severity_class_from_any_z(severity_any_z: float) -> str:
    if not np.isfinite(severity_any_z):
        return "unknown"
    if severity_any_z < 1.0:
        return "weak"
    if severity_any_z < 2.0:
        return "mild"
    if severity_any_z < 3.0:
        return "moderate"
    return "severe"


def directionality_class(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > 0.3:
        return "upstream_dominant"
    if value < -0.3:
        return "downstream_dominant"
    return "balanced"


def build_labels_for_region(
    region: RegionSpec,
    data_dir: Path,
    output_dir: Path,
    inc: pd.DataFrame,
    meta: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    region_dir = output_dir / region.name
    region_dir.mkdir(parents=True, exist_ok=True)

    region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
    region_meta = region_meta.reset_index(drop=True)
    region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)
    print(f"[{region.name}] loading traffic for {len(region_meta)} mainline sensors", flush=True)
    traffic, times = load_region_traffic(data_dir, region_node_idx)

    print(f"[{region.name}] matching incidents", flush=True)
    matches = build_matches(
        inc=inc,
        region_meta=region_meta,
        times=times,
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
        baseline_mask_extra_steps=args.baseline_mask_extra_steps,
    )
    print(f"[{region.name}] matched incidents: {len(matches)}", flush=True)

    print(f"[{region.name}] building baseline mask", flush=True)
    baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)

    print(f"[{region.name}] building robust counterfactual baseline", flush=True)
    baseline, scale, counts = build_robust_baseline(
        traffic=traffic,
        times=times,
        baseline_valid=baseline_valid,
        min_count=args.min_baseline_count,
    )

    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    sensor_id_arr = region_meta["station_id"].to_numpy(dtype=np.int64)
    abs_pm_arr = region_meta["Abs PM"].to_numpy(dtype=np.float64)

    event_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    total_slots = traffic.shape[0]

    print(f"[{region.name}] deriving labels", flush=True)
    for seq, match in enumerate(matches, start=1):
        if seq % 5000 == 0:
            print(f"[{region.name}] processed {seq}/{len(matches)} matches", flush=True)
        start = match.start_idx
        end = min(total_slots, start + args.horizon_steps)
        if end <= start:
            continue
        ts = np.arange(start, end, dtype=np.int64)
        cand = match.candidate_region_idx
        actual = traffic[ts][:, cand, :]
        base = baseline[day_kind[ts], tod[ts]][:, cand, :]
        scl = scale[day_kind[ts], tod[ts]][:, cand, :]

        flow_rel, speed_rel, occ_rel = relative_impacts(
            actual=actual,
            baseline=base,
            max_relative_impact=args.max_relative_impact,
        )
        flow_z, speed_z, occ_z, any_z = robust_z_impacts(actual=actual, baseline=base, scale=scl)

        min_valid_steps = min(args.min_valid_impact_steps, max(3, int(math.ceil(actual.shape[0] * 0.5))))
        valid_flow = np.isfinite(flow_rel)
        valid_speed = np.isfinite(speed_rel)
        valid_occ = np.isfinite(occ_rel)
        valid_any = np.isfinite(any_z)

        reliable_flow = valid_flow.sum(axis=0) >= min_valid_steps
        reliable_speed = valid_speed.sum(axis=0) >= min_valid_steps
        reliable_occ = valid_occ.sum(axis=0) >= min_valid_steps
        reliable_any = valid_any.sum(axis=0) >= min_valid_steps

        flow_rel[:, ~reliable_flow] = np.nan
        speed_rel[:, ~reliable_speed] = np.nan
        occ_rel[:, ~reliable_occ] = np.nan
        flow_z[:, ~reliable_flow] = np.nan
        speed_z[:, ~reliable_speed] = np.nan
        occ_z[:, ~reliable_occ] = np.nan
        any_z[:, ~reliable_any] = np.nan

        node_flow_auc = safe_nanmean(flow_rel, axis=0)
        node_speed_auc = safe_nanmean(speed_rel, axis=0)
        node_occ_auc = safe_nanmean(occ_rel, axis=0)
        node_flow_peak = safe_nanmax(flow_rel, axis=0)
        node_speed_peak = safe_nanmax(speed_rel, axis=0)
        node_occ_peak = safe_nanmax(occ_rel, axis=0)

        node_flow_z_auc = safe_nanmean(flow_z, axis=0)
        node_speed_z_auc = safe_nanmean(speed_z, axis=0)
        node_occ_z_auc = safe_nanmean(occ_z, axis=0)
        node_any_z_auc = safe_nanmean(any_z, axis=0)
        node_any_z_peak = safe_nanmax(any_z, axis=0)
        affected = np.isfinite(node_any_z_peak) & (node_any_z_peak >= args.affected_z)

        event_curve = np.array(
            [topk_mean(any_z[t], args.topk_nodes) for t in range(any_z.shape[0])],
            dtype=np.float32,
        )
        recovery_time, recovery_censored = find_recovery_time(
            event_curve=event_curve,
            recovery_z=args.recovery_z,
            consecutive_steps=args.recovery_consecutive_steps,
        )
        if np.any(np.isfinite(event_curve)):
            time_to_peak = float(np.nanargmax(event_curve) * 5)
            peak_any_z = float(np.nanmax(event_curve))
        else:
            time_to_peak = float("nan")
            peak_any_z = float("nan")

        upstream_mask = match.candidate_side == -1
        downstream_mask = match.candidate_side == 1
        upstream_impact = topk_mean(node_any_z_auc[upstream_mask], args.topk_nodes)
        downstream_impact = topk_mean(node_any_z_auc[downstream_mask], args.topk_nodes)
        directionality = (
            (upstream_impact - downstream_impact)
            / (upstream_impact + downstream_impact + 1e-6)
            if np.isfinite(upstream_impact) and np.isfinite(downstream_impact)
            else float("nan")
        )

        affected_pm = abs_pm_arr[cand][affected]
        if affected_pm.size:
            spread_nodes = int(affected_pm.size)
            spread_pm = float(np.nanmax(affected_pm) - np.nanmin(affected_pm))
            spread_radius_pm = float(np.nanmax(np.abs(affected_pm - match.abs_pm)))
        else:
            spread_nodes = 0
            spread_pm = 0.0
            spread_radius_pm = 0.0

        severity_any_z_auc_topk = topk_mean(node_any_z_auc, args.topk_nodes)
        event_row = {
            "region": region.name,
            "incident_id": match.incident_id,
            "incident_row_idx": match.row_idx,
            "start_idx": match.start_idx,
            "dt": str(times[match.start_idx]),
            "duration_min": match.duration_min,
            "type": match.incident_type,
            "area": match.area,
            "description": match.description,
            "fwy": match.freeway,
            "direction": match.direction,
            "incident_abs_pm": match.abs_pm,
            "latitude": match.latitude,
            "longitude": match.longitude,
            "anchor_sensor_id": match.anchor_sensor_id,
            "anchor_region_idx": match.anchor_region_idx,
            "anchor_pm_dist": match.anchor_pm_dist,
            "candidate_nodes": int(cand.size),
            "severity_flow_auc_topk": topk_mean(node_flow_auc, args.topk_nodes),
            "severity_speed_auc_topk": topk_mean(node_speed_auc, args.topk_nodes),
            "severity_occ_auc_topk": topk_mean(node_occ_auc, args.topk_nodes),
            "severity_flow_peak_topk": topk_mean(node_flow_peak, args.topk_nodes),
            "severity_speed_peak_topk": topk_mean(node_speed_peak, args.topk_nodes),
            "severity_occ_peak_topk": topk_mean(node_occ_peak, args.topk_nodes),
            "severity_flow_z_auc_topk": topk_mean(node_flow_z_auc, args.topk_nodes),
            "severity_speed_z_auc_topk": topk_mean(node_speed_z_auc, args.topk_nodes),
            "severity_occ_z_auc_topk": topk_mean(node_occ_z_auc, args.topk_nodes),
            "severity_any_z_auc_topk": severity_any_z_auc_topk,
            "severity_any_z_peak_topk": peak_any_z,
            "severity_class_z": severity_class_from_any_z(severity_any_z_auc_topk),
            "recovery_time_min": recovery_time,
            "recovery_censored": recovery_censored,
            "time_to_peak_min": time_to_peak,
            "spread_nodes": spread_nodes,
            "spread_pm": spread_pm,
            "spread_radius_pm": spread_radius_pm,
            "upstream_impact_z_auc_topk": upstream_impact,
            "downstream_impact_z_auc_topk": downstream_impact,
            "directionality": directionality,
            "directionality_class": directionality_class(directionality),
        }
        event_rows.append(event_row)

        if not args.skip_node_labels:
            for local_pos, region_idx in enumerate(cand):
                side = int(match.candidate_side[local_pos])
                if side == -1:
                    side_name = "upstream"
                elif side == 1:
                    side_name = "downstream"
                else:
                    side_name = "at_incident"
                node_rows.append(
                    {
                        "region": region.name,
                        "incident_id": match.incident_id,
                        "sensor_id": int(sensor_id_arr[region_idx]),
                        "region_node_idx": int(region_idx),
                        "pm_dist": float(match.candidate_pm_dist[local_pos]),
                        "side": side_name,
                        "affected": int(bool(affected[local_pos])),
                        "flow_auc": float(node_flow_auc[local_pos]),
                        "speed_auc": float(node_speed_auc[local_pos]),
                        "occ_auc": float(node_occ_auc[local_pos]),
                        "flow_peak": float(node_flow_peak[local_pos]),
                        "speed_peak": float(node_speed_peak[local_pos]),
                        "occ_peak": float(node_occ_peak[local_pos]),
                        "flow_z_auc": float(node_flow_z_auc[local_pos]),
                        "speed_z_auc": float(node_speed_z_auc[local_pos]),
                        "occ_z_auc": float(node_occ_z_auc[local_pos]),
                        "any_z_auc": float(node_any_z_auc[local_pos]),
                        "any_z_peak": float(node_any_z_peak[local_pos]),
                    }
                )

    event_df = pd.DataFrame(event_rows)
    event_path = region_dir / "event_labels.csv"
    event_df.to_csv(event_path, index=False)

    node_path = None
    if not args.skip_node_labels:
        node_df = pd.DataFrame(node_rows)
        node_path = region_dir / "node_labels.csv"
        node_df.to_csv(node_path, index=False)

    meta_path = region_dir / "region_sensors.csv"
    region_meta.to_csv(meta_path, index=False)

    config = {
        "region": region.name,
        "county": region.county,
        "mainline_sensors": int(len(region_meta)),
        "matched_incidents": int(len(matches)),
        "event_labels": str(event_path),
        "node_labels": str(node_path) if node_path else None,
        "candidate_pm_radius": args.candidate_pm_radius,
        "anchor_pm_radius": args.anchor_pm_radius,
        "horizon_steps": args.horizon_steps,
        "topk_nodes": args.topk_nodes,
        "affected_z": args.affected_z,
        "recovery_z": args.recovery_z,
    }
    with (region_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary = summarize_event_df(region.name, event_df)
    with (region_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[{region.name}] wrote {event_path}", flush=True)
    if node_path:
        print(f"[{region.name}] wrote {node_path}", flush=True)
    return config


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def summarize_event_df(region_name: str, df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# {region_name} Impact Label Summary")
    lines.append("")
    lines.append(f"- events: {len(df)}")
    if len(df) == 0:
        return "\n".join(lines) + "\n"

    quant_cols = [
        "severity_flow_auc_topk",
        "severity_speed_auc_topk",
        "severity_occ_auc_topk",
        "severity_any_z_auc_topk",
        "recovery_time_min",
        "spread_nodes",
        "spread_pm",
        "directionality",
    ]
    lines.append(f"- recovery_censored_rate: {fmt(df['recovery_censored'].mean())}")
    lines.append("")
    lines.append("## Quantiles")
    lines.append("")
    lines.append("| metric | p25 | p50 | p75 | p90 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for col in quant_cols:
        q = df[col].quantile([0.25, 0.5, 0.75, 0.9])
        lines.append(
            f"| {col} | {fmt(q.loc[0.25])} | {fmt(q.loc[0.5])} | "
            f"{fmt(q.loc[0.75])} | {fmt(q.loc[0.9])} |"
        )

    lines.append("")
    lines.append("## Severity Class")
    lines.append("")
    severity_counts = df["severity_class_z"].value_counts(dropna=False)
    for name, count in severity_counts.items():
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("## Directionality Class")
    lines.append("")
    direction_counts = df["directionality_class"].value_counts(dropna=False)
    for name, count in direction_counts.items():
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("## Median By Incident Type")
    lines.append("")
    by_type = (
        df.groupby("type", dropna=False)[
            [
                "duration_min",
                "severity_any_z_auc_topk",
                "recovery_time_min",
                "spread_nodes",
                "directionality",
            ]
        ]
        .median()
        .sort_values("severity_any_z_auc_topk", ascending=False)
        .head(12)
    )
    lines.append(by_type.to_markdown())
    lines.append("")
    return "\n".join(lines) + "\n"


def write_global_summary(output_dir: Path, configs: list[dict[str, object]]) -> None:
    lines = ["# Impact Label Build Summary", ""]
    rows = []
    for cfg in configs:
        event_path = Path(str(cfg["event_labels"]))
        df = pd.read_csv(event_path)
        rows.append(
            {
                "region": cfg["region"],
                "sensors": cfg["mainline_sensors"],
                "events": len(df),
                "severity_p50": df["severity_any_z_auc_topk"].median(),
                "recovery_p50": df["recovery_time_min"].median(),
                "spread_nodes_p50": df["spread_nodes"].median(),
                "censored_rate": df["recovery_censored"].mean(),
                "upstream_dominant_rate": (df["directionality_class"] == "upstream_dominant").mean(),
            }
        )
    summary_df = pd.DataFrame(rows)
    lines.append(summary_df.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for cfg in configs:
        lines.append(f"- {cfg['region']}: `{cfg['event_labels']}`")
        if cfg.get("node_labels"):
            lines.append(f"- {cfg['region']} nodes: `{cfg['node_labels']}`")
    lines.append("")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = region_specs()
    selected_regions = []
    for name in args.regions:
        if name not in specs:
            raise ValueError(f"Unknown region {name}; choose from {sorted(specs)}")
        selected_regions.append(specs[name])

    print("loading sensor metadata and 2023 incidents", flush=True)
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)
    print(f"incidents in 2023 table after basic filtering: {len(inc)}", flush=True)

    configs: list[dict[str, object]] = []
    for region in selected_regions:
        configs.append(
            build_labels_for_region(
                region=region,
                data_dir=data_dir,
                output_dir=output_dir,
                inc=inc,
                meta=meta,
                args=args,
            )
        )
    write_global_summary(output_dir, configs)
    print(f"wrote global summary to {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
