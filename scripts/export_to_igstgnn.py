#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REGION_TO_COUNTY = {
    "Alameda": "Alameda",
    "Contra_Costa": "Contra Costa",
    "Orange": "Orange",
}

US_FEDERAL_HOLIDAYS_2023 = {
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09",
    "2023-11-10", "2023-11-23", "2023-12-25",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("/Users/xhlm/Desktop/Study/科研实习/archive"))
    p.add_argument("--out-root", type=Path, default=Path("/Users/xhlm/Desktop/Study/科研实习/baselines/IGSTGNN/data/xtraffic"))
    p.add_argument("--region", type=str, required=True, choices=list(REGION_TO_COUNTY.keys()))
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--input-steps", type=int, default=12)
    p.add_argument("--horizon-steps", type=int, default=12)
    p.add_argument("--match-radius-km", type=float, default=5.0,
                   help="Drop incidents whose nearest county-mainline node is farther than this")
    p.add_argument("--sigma-e-km", type=float, default=5.0)
    p.add_argument("--sigma-r-km", type=float, default=5.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-incidents", type=int, default=0,
                   help="Cap incidents (0 = no cap, for smoke testing)")
    return p.parse_args()


def load_region_meta(meta_path: Path, node_order_path: Path, county: str) -> tuple[np.ndarray, pd.DataFrame]:
    node_order = np.load(node_order_path)
    meta = pd.read_csv(meta_path, sep="\t")
    meta_by_id = meta.set_index("station_id")
    aligned = meta_by_id.loc[node_order].reset_index(drop=False).rename(columns={"index": "station_id"})
    is_region = (aligned["County"] == county) & (aligned["Type"] == "Mainline")
    region_idx = np.where(is_region.values)[0]
    region_meta = aligned.iloc[region_idx].reset_index(drop=True)
    return region_idx, region_meta


def _strip_units(value, default):
    if pd.isna(value):
        return default
    s = str(value).strip().split()
    try:
        return float(s[0])
    except (ValueError, IndexError):
        return default


def build_sensor_info(region_meta: pd.DataFrame) -> dict:
    sensor_type_categories = ["Mainline", "HOV", "Fwy-Fwy"]
    sensor_type_map = {c: i + 1 for i, c in enumerate(sensor_type_categories)}
    surface_categories = sorted(region_meta["Surface"].dropna().astype(str).unique().tolist())
    surface_map = {c: i + 1 for i, c in enumerate(surface_categories)}
    roadway_use_categories = sorted(region_meta["Roadway Use"].dropna().astype(str).unique().tolist())
    roadway_use_map = {c: i + 1 for i, c in enumerate(roadway_use_categories)}

    sensor_type = region_meta["Sensor Type"].fillna("").astype(str).map(lambda v: sensor_type_map.get(v, 0)).values
    if sensor_type.sum() == 0:
        sensor_type = region_meta["Type"].astype(str).map(lambda v: sensor_type_map.get(v, 0)).values
    surface = region_meta["Surface"].astype(str).map(lambda v: surface_map.get(v, 0)).values
    roadway_use = region_meta["Roadway Use"].astype(str).map(lambda v: roadway_use_map.get(v, 0)).values
    road_width = np.array([_strip_units(v, 0.0) for v in region_meta["Road Width"]], dtype=np.float32)
    speed_limit = np.array([_strip_units(v, 0.0) for v in region_meta["Design Speed Limit"]], dtype=np.float32)
    rw_max = float(road_width.max() or 1.0)
    sl_max = float(speed_limit.max() or 1.0)
    road_width = road_width / rw_max
    speed_limit = speed_limit / sl_max
    return {
        "sensor_type": sensor_type.astype(np.int64),
        "surface": surface.astype(np.int64),
        "roadway_use": roadway_use.astype(np.int64),
        "road_width": road_width.astype(np.float32),
        "speed_limit": speed_limit.astype(np.float32),
        "_meta": {
            "sensor_type_map": sensor_type_map,
            "surface_map": surface_map,
            "roadway_use_map": roadway_use_map,
            "road_width_max": rw_max,
            "speed_limit_max": sl_max,
        },
    }


def stitch_traffic(archive: Path, region_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_flow shape (T, N, 1), valid_mask shape (T, N))."""
    flows = []
    cumulative = 0
    months = []
    for m in range(1, 13):
        a = np.load(archive / f"p{m:02d}_done.npy", mmap_mode="r")
        sub = a[:, region_idx, 0:1].astype(np.float32)
        flows.append(sub)
        months.append((cumulative, cumulative + sub.shape[0]))
        cumulative += sub.shape[0]
    X = np.concatenate(flows, axis=0)
    # Valid: positive flow values (XTraffic uses -1 / nan for missing)
    valid = (X[..., 0] >= 0) & np.isfinite(X[..., 0])
    X = np.where(valid[..., None], X, 0.0)
    return X, valid


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_incidents(archive: Path, year: int, region_meta: pd.DataFrame, total_steps: int, match_radius_km: float):
    csv_path = archive / f"incidents_y{year}.csv"
    inc = pd.read_csv(csv_path, sep="\t", low_memory=False)
    inc = inc.dropna(subset=["dt", "Latitude", "Longitude"]).copy()
    inc["dt"] = pd.to_datetime(inc["dt"], errors="coerce")
    inc = inc.dropna(subset=["dt"])
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
    secs = (inc["dt"] - year_start).dt.total_seconds().values
    inc["t_idx"] = (secs // 300).astype(int)
    inc = inc[(inc["t_idx"] >= 0) & (inc["t_idx"] < total_steps)].copy()

    node_lat = region_meta["Lat"].values.astype(np.float64)
    node_lon = region_meta["Lng"].values.astype(np.float64)
    node_pm = region_meta["Abs PM"].values.astype(np.float64)
    node_fwy_name = region_meta["Fwy Name"].fillna("").astype(str).values

    inc_lat = inc["Latitude"].values.astype(np.float64)
    inc_lon = inc["Longitude"].values.astype(np.float64)
    inc_pm = inc["Abs PM"].values.astype(np.float64)
    inc_fwy = inc["Fwy"].fillna(0).astype(float).values
    inc_dir = inc["Freeway_direction"].fillna("").astype(str).values

    # For each incident, find nearest region node by haversine
    block = 256
    nearest_idx = np.full(len(inc), -1, dtype=np.int64)
    nearest_dist = np.full(len(inc), np.inf)
    for start in range(0, len(inc), block):
        end = min(len(inc), start + block)
        d = haversine_km(
            inc_lat[start:end, None], inc_lon[start:end, None],
            node_lat[None, :], node_lon[None, :],
        )
        nearest_idx[start:end] = d.argmin(axis=1)
        nearest_dist[start:end] = d.min(axis=1)
    keep = nearest_dist <= match_radius_km
    inc = inc.iloc[keep].reset_index(drop=True)
    nearest_idx = nearest_idx[keep]
    nearest_dist = nearest_dist[keep]
    return inc, nearest_idx, nearest_dist


def compute_distances(region_meta: pd.DataFrame, dis_matrix: np.ndarray, region_idx: np.ndarray,
                      anchor_node_idx: int, inc_lat: float, inc_lon: float, inc_pm: float, inc_fwy: float,
                      sigma_e: float, sigma_r: float, radius_km: float) -> np.ndarray:
    """Return (N, 3): [Euclidean Gaussian, road-network Gaussian, upstream indicator]."""
    n = len(region_meta)
    node_lat = region_meta["Lat"].values.astype(np.float64)
    node_lon = region_meta["Lng"].values.astype(np.float64)
    node_pm = region_meta["Abs PM"].values.astype(np.float64)
    node_fwy_raw = region_meta["Fwy"].fillna(-1).astype(float).values  # numeric Fwy

    eu_km = haversine_km(inc_lat, inc_lon, node_lat, node_lon)
    rn_km = dis_matrix[region_idx[anchor_node_idx], region_idx]
    rn_km = np.where(rn_km < 0, np.inf, rn_km / 1000.0)  # convert m -> km if needed; the matrix appears to be in m

    eu_kernel = np.exp(-(eu_km ** 2) / (2 * sigma_e ** 2))
    rn_kernel = np.exp(-(rn_km ** 2) / (2 * sigma_r ** 2))
    eu_kernel = np.where(eu_km <= radius_km, eu_kernel, 0.0)
    rn_kernel = np.where(rn_km <= radius_km, rn_kernel, 0.0)

    same_fwy = (node_fwy_raw == inc_fwy) & (inc_fwy > 0)
    upstream = ((node_pm > inc_pm) & same_fwy).astype(np.float32)

    dist = np.stack([eu_kernel.astype(np.float32),
                     rn_kernel.astype(np.float32),
                     upstream.astype(np.float32)], axis=-1)
    return dist


def is_holiday(ts: pd.Timestamp) -> int:
    return 1 if ts.strftime("%Y-%m-%d") in US_FEDERAL_HOLIDAYS_2023 else 0


def build_samples(args, X_flow, valid_mask, region_meta, region_idx, dis_matrix, inc, nearest_idx,
                  desc_map, type_map, year: int, mean: float = 0.0, std: float = 1.0):
    T, N, _ = X_flow.shape
    in_steps = args.input_steps
    h = args.horizon_steps
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")

    # build per-step time features (broadcast to N at sample time)
    t_arr = np.arange(T)
    tod = (t_arr % 288).astype(np.float32) / 288.0
    start_dow = year_start.weekday()  # 6 for 2023 Sun
    dow = ((t_arr // 288 + start_dow) % 7).astype(np.float32) / 7.0

    samples = []
    skipped_window = 0
    skipped_validity = 0
    for k in range(len(inc)):
        t = int(inc["t_idx"].iloc[k])
        if t < in_steps or t + h > T:
            skipped_window += 1
            continue
        # Anchor incident at last input step: input window = [t-in_steps+1, t]
        x_start = t - in_steps + 1
        x_end = t + 1
        y_start = t + 1
        y_end = t + 1 + h
        if y_end > T:
            skipped_window += 1
            continue
        x_flow = X_flow[x_start:x_end, :, :]
        y_flow = X_flow[y_start:y_end, :, :]
        x_valid = valid_mask[x_start:x_end].mean()
        y_valid_window = valid_mask[y_start:y_end].mean()
        if x_valid < 0.30 or y_valid_window < 0.30:
            skipped_validity += 1
            continue

        # Build x_data with 3 channels: normalized flow, tod, dow
        tod_block = np.broadcast_to(tod[x_start:x_end, None, None], (in_steps, N, 1))
        dow_block = np.broadcast_to(dow[x_start:x_end, None, None], (in_steps, N, 1))
        x_flow_norm = ((x_flow - mean) / std).astype(np.float32)
        x_data = np.concatenate([x_flow_norm, tod_block.astype(np.float32), dow_block.astype(np.float32)], axis=-1)
        y_data = ((y_flow - mean) / std).astype(np.float32)

        anchor = int(nearest_idx[k])
        inc_row = inc.iloc[k]
        distances = compute_distances(
            region_meta, dis_matrix, region_idx, anchor,
            float(inc_row["Latitude"]), float(inc_row["Longitude"]),
            float(inc_row["Abs PM"]) if not pd.isna(inc_row["Abs PM"]) else -1.0,
            float(inc_row["Fwy"]) if not pd.isna(inc_row["Fwy"]) else -1.0,
            args.sigma_e_km, args.sigma_r_km, args.match_radius_km * 2,
        )

        desc = desc_map.get(str(inc_row["DESCRIPTION"]).strip(), 0)
        typ = type_map.get(str(inc_row["Type"]).strip(), 0)
        holiday = is_holiday(inc_row["dt"])
        delta_time_min = 0.0  # incident occurs at last input step; horizon minutes since occurrence handled by TIID

        sample = {
            "x_data": x_data,
            "y_data": y_data,
            "event_features": {
                "Event Time": float(delta_time_min),
                "Description": int(desc),
                "Type": int(typ),
                "Holiday": int(holiday),
            },
            "event_position": int(in_steps - 1),
            "event_distances": distances.astype(np.float32),
            "durations": float(inc_row["duration"]) if not pd.isna(inc_row["duration"]) else 0.0,
            "_t_idx": int(t),
        }
        samples.append(sample)
    return samples, {"skipped_window": skipped_window, "skipped_validity": skipped_validity}


def main():
    args = parse_args()
    county = REGION_TO_COUNTY[args.region]
    out_dir = args.out_root / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading region meta ({county})")
    region_idx, region_meta = load_region_meta(
        args.archive / "sensor_meta_feature.csv",
        args.archive / "node_order.npy",
        county,
    )
    print(f"  region_meta: {len(region_meta)} mainline nodes (region_idx in {region_idx.min()}..{region_idx.max()})")

    print("[2/7] Slicing adjacency")
    adj_full = np.load(args.archive / "adj_matrix.npy", mmap_mode="r")
    adj = np.array(adj_full[region_idx][:, region_idx], dtype=np.float32)
    np.save(out_dir / "adj_matrix.npy", adj)
    print(f"  adj saved {adj.shape}")

    print("[3/7] Loading distance matrix")
    dis_matrix = np.load(args.archive / "dis_matrix.npy", mmap_mode="r")

    print("[4/7] Stitching traffic flow (12 monthly partitions)")
    X_flow, valid_mask = stitch_traffic(args.archive, region_idx)
    T = X_flow.shape[0]
    print(f"  X_flow: {X_flow.shape} valid_frac={valid_mask.mean():.4f}")

    print("[5/7] Parsing incidents")
    inc, nearest_idx, nearest_dist = parse_incidents(args.archive, args.year, region_meta, T, args.match_radius_km)
    if args.max_incidents > 0:
        inc = inc.iloc[:args.max_incidents].reset_index(drop=True)
        nearest_idx = nearest_idx[:args.max_incidents]
        nearest_dist = nearest_dist[:args.max_incidents]
    print(f"  matched {len(inc)} incidents within {args.match_radius_km}km of region nodes")
    print(f"  median match dist: {float(np.median(nearest_dist)):.3f} km")
    print(f"  Type counts: {inc['Type'].value_counts().to_dict()}")

    train_end_t = int(T * args.train_frac)
    val_end_t = int(T * (args.train_frac + args.val_frac))
    train_mask = inc["t_idx"] < train_end_t
    val_mask = (inc["t_idx"] >= train_end_t) & (inc["t_idx"] < val_end_t)
    test_mask = inc["t_idx"] >= val_end_t
    print(f"  split incidents: train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}")

    print("[6/7] Building mappings (from training incidents only)")
    train_descs = sorted(inc.loc[train_mask, "DESCRIPTION"].astype(str).str.strip().unique().tolist())
    train_types = sorted(inc.loc[train_mask, "Type"].astype(str).str.strip().unique().tolist())
    desc_map = {d: i for i, d in enumerate(train_descs)}
    type_map = {t: i for i, t in enumerate(train_types)}
    print(f"  num descriptions: {len(desc_map)}, num types: {len(type_map)}")
    with open(out_dir / "desc_mapping.json", "w") as f:
        json.dump(desc_map, f)
    with open(out_dir / "type_mapping.json", "w") as f:
        json.dump(type_map, f)

    print("[7a/7] Computing flow mean/std on train period (for normalization)")
    train_flow = X_flow[:train_end_t, :, 0]
    train_valid = valid_mask[:train_end_t]
    flat = train_flow[train_valid]
    mean = float(flat.mean())
    std = float(flat.std())
    np.savez(out_dir / "incident_data_stats.npz", mean=mean, std=std)
    print(f"  mean={mean:.4f} std={std:.4f}")

    print("[7b/7] Building samples per split")
    for cat, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub_inc = inc.loc[mask].reset_index(drop=True)
        sub_near = nearest_idx[mask.values]
        samples, stats = build_samples(
            args, X_flow, valid_mask, region_meta, region_idx, dis_matrix,
            sub_inc, sub_near, desc_map, type_map, args.year, mean=mean, std=std,
        )
        out_path = out_dir / f"incident_data_{cat}.npy"
        np.save(out_path, np.array(samples, dtype=object), allow_pickle=True)
        print(f"  {cat}: kept {len(samples)} samples (skipped {stats['skipped_window']} window / {stats['skipped_validity']} validity)")

    print("[sensor] Building per-node sensor info")
    sensor_info = build_sensor_info(region_meta)
    sensor_meta_save = sensor_info.pop("_meta")
    np.savez(out_dir / "sensor_info.npz", **sensor_info)
    with open(out_dir / "sensor_info_meta.json", "w") as f:
        json.dump(sensor_meta_save, f, indent=2)
    print(f"  saved sensor_info.npz with keys: {list(sensor_info.keys())}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
