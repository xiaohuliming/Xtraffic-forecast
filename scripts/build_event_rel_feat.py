#!/usr/bin/env python3
"""Precompute per-event incident↔sensor relational features for DIST-Net.

For each event in events.csv of a region, compute a (N, 4) feature tensor
encoding spatial geometry between the event and every sensor in the region:

  rel_feat[:, 0] = log_euclid     log(haversine_km + 1)
  rel_feat[:, 1] = log_road       log(road_network_km + 1), inf → log(MAX_ROAD + 1)
  rel_feat[:, 2] = up_down        -1 upstream, +1 downstream, 0 different/no fwy
  rel_feat[:, 3] = same_freeway   0/1

Output per region:
  outputs/dist_net/region_data/{region}_event_rel_feat.npz
    rel_feat   (n_events, N, 4) float32

This addresses the rel_feat deferred in build_full_county_cache.py and lets
DIST-Net's incident branch learn the spatial inductive bias that IGSTGNN
gets for free via its preset distance kernels (compute_distances in
export_to_igstgnn.py).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_impact_labels import (
    load_incidents_2023,
    load_region_traffic,  # noqa: F401  — kept for parity; not used here
    load_sensor_meta,
    region_specs,
)

REGIONS_DEFAULT = ["Alameda", "ContraCosta", "Orange"]

# Cap for "unreachable" road distances. log(MAX_ROAD_KM + 1) becomes a finite
# value that the model can learn to ignore. Setting MAX_ROAD_KM at 200 km
# means log_road maxes at ~log(201) ≈ 5.30.
MAX_ROAD_KM = 200.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in km. Inputs in degrees, broadcastable."""
    R = 6371.0
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def direction_sign(event_dir: str, event_pm: float, sensor_pm: np.ndarray,
                   same_fwy: np.ndarray) -> np.ndarray:
    """Returns -1 if sensor is upstream of event, +1 if downstream, 0 otherwise.

    Caltrans postmiles increase N/E. Vehicles travel N/E from low PM → high PM.
    So for N/E direction: sensor.pm > event.pm = downstream (+1).
    For S/W direction: sensor.pm < event.pm = downstream (+1).
    """
    delta = sensor_pm - event_pm                                # (N,)
    side = np.zeros_like(delta, dtype=np.float32)
    if event_dir in ("N", "E"):
        side[delta < -1e-6] = -1.0   # sensor PM smaller = upstream
        side[delta > 1e-6]  = +1.0
    elif event_dir in ("S", "W"):
        side[delta > 1e-6]  = -1.0
        side[delta < -1e-6] = +1.0
    # zero out where not on same fwy
    side[~same_fwy] = 0.0
    return side


def compute_event_rel_feat(events: pd.DataFrame, region_meta: pd.DataFrame,
                           region_idx: np.ndarray, dis_matrix: np.ndarray
                           ) -> np.ndarray:
    """Returns (n_events, N, 4) float32."""
    N = len(region_meta)
    sensor_lat = region_meta["Lat"].values.astype(np.float64)
    sensor_lon = region_meta["Lng"].values.astype(np.float64)
    sensor_pm = region_meta["Abs PM"].values.astype(np.float64)
    sensor_fwy = pd.to_numeric(region_meta["Fwy"], errors="coerce").fillna(-1).values
    n_events = len(events)
    out = np.zeros((n_events, N, 4), dtype=np.float32)

    log_max_road = float(np.log(MAX_ROAD_KM + 1.0))

    # Pull event fields as columns for fast row access
    ev_lat = pd.to_numeric(events["latitude"], errors="coerce").values
    ev_lon = pd.to_numeric(events["longitude"], errors="coerce").values
    ev_pm  = pd.to_numeric(events["incident_abs_pm"], errors="coerce").values
    ev_fwy = pd.to_numeric(events["fwy"], errors="coerce").fillna(-1).values
    ev_dir = events["direction"].fillna("").astype(str).str.strip().str.upper().str[:1].values
    ev_anchor_region_idx = events["anchor_region_idx"].astype(np.int64).values

    last_print = -1
    for i in range(n_events):
        lat, lon = ev_lat[i], ev_lon[i]
        pm = ev_pm[i]
        fwy = ev_fwy[i]
        d = ev_dir[i] if i < len(ev_dir) else ""

        # 1. log_euclid (defensive against NaN)
        if np.isfinite(lat) and np.isfinite(lon):
            eu_km = haversine_km(np.full(N, lat), np.full(N, lon), sensor_lat, sensor_lon)
            log_eu = np.log(np.clip(eu_km, 0.0, None) + 1.0).astype(np.float32)
        else:
            log_eu = np.full(N, log_max_road, dtype=np.float32)

        # 2. log_road via dis_matrix anchor row
        anchor = int(ev_anchor_region_idx[i])
        if 0 <= anchor < N:
            rn_m = dis_matrix[region_idx[anchor], region_idx]      # (N,) meters
            rn_km = np.where(rn_m < 0, np.inf, rn_m.astype(np.float64) / 1000.0)
            rn_km = np.where(np.isfinite(rn_km), np.minimum(rn_km, MAX_ROAD_KM), MAX_ROAD_KM)
            log_rn = np.log(rn_km + 1.0).astype(np.float32)
        else:
            log_rn = np.full(N, log_max_road, dtype=np.float32)

        # 3. same freeway + 4. upstream/downstream
        if np.isfinite(fwy) and fwy > 0:
            same_fwy = sensor_fwy == fwy
        else:
            same_fwy = np.zeros(N, dtype=bool)
        same_fwy_f = same_fwy.astype(np.float32)
        if np.isfinite(pm):
            up_dn = direction_sign(d, float(pm), sensor_pm, same_fwy)
        else:
            up_dn = np.zeros(N, dtype=np.float32)

        out[i, :, 0] = log_eu
        out[i, :, 1] = log_rn
        out[i, :, 2] = up_dn
        out[i, :, 3] = same_fwy_f

        if i // 1000 != last_print:
            print(f"    event {i}/{n_events}", flush=True)
            last_print = i // 1000
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("archive"))
    p.add_argument("--event-root", type=Path,
                   default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/dist_net/region_data"))
    p.add_argument("--regions", nargs="+", default=REGIONS_DEFAULT)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading global sensor metadata + dis_matrix ...")
    meta = load_sensor_meta(args.archive)
    dis_matrix = np.load(args.archive / "dis_matrix.npy", mmap_mode="r")
    print(f"  dis_matrix shape: {dis_matrix.shape}")

    specs = region_specs()
    for region_name in args.regions:
        if region_name not in specs:
            raise ValueError(f"unknown region {region_name}")
        region = specs[region_name]
        print(f"\n=== {region_name} ({region.county}) ===", flush=True)

        region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
        region_meta = region_meta.reset_index(drop=True)
        region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int64)
        N = region_node_idx.size
        print(f"  N = {N}")

        events_path = args.event_root / region_name / "event_labels.csv"
        events = pd.read_csv(events_path)
        events = events.sort_values("start_idx").reset_index(drop=True)
        n_events = len(events)
        print(f"  events (sorted by start_idx): {n_events}")

        t0 = time.time()
        rel_feat = compute_event_rel_feat(events, region_meta, region_node_idx, dis_matrix)
        print(f"  rel_feat shape: {rel_feat.shape}  (mem={rel_feat.nbytes / 1e6:.0f} MB)  "
              f"in {time.time() - t0:.1f}s")

        out_path = args.out_dir / f"{region_name}_event_rel_feat.npz"
        np.savez_compressed(
            out_path,
            rel_feat=rel_feat,
            n_events=np.int32(n_events),
            max_road_km=np.float32(MAX_ROAD_KM),
        )
        sz = out_path.stat().st_size / (1024 ** 2)
        print(f"  saved -> {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
