#!/usr/bin/env python3
"""Precompute per-event incident↔sensor relational features for DIST-Net.

For each event in events.csv of a region, compute a (N, 4) feature tensor
encoding spatial geometry between the event and every sensor in the region:

  rel_feat[:, 0] = log_euclid     log(haversine_km + 1)
  rel_feat[:, 1] = log_road       log(road_network_km + 1), inf → log(MAX_ROAD + 1)
  rel_feat[:, 2] = up_down        -1 upstream, +1 downstream, 0 different/no fwy
  rel_feat[:, 3] = same_freeway   0/1

Output per region:
  outputs/dist_net/region_data_v3_noleak_dedup/{region}_event_rel_feat.npz
    rel_feat   (n_events, N, 4) float32

This addresses the rel_feat deferred in build_full_county_cache.py and lets
DIST-Net's incident branch learn the spatial inductive bias that IGSTGNN
gets for free via its preset distance kernels (compute_distances in
export_to_igstgnn.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dist_net.protocol import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_REGION_DATA_DIR,
    PROTOCOL_VERSION,
    git_worktree_provenance,
)
from scripts.build_full_county_cache import (
    canonicalize_events,
    resolve_cache_mode,
    sha256_file,
    source_manifest,
)
from scripts.build_impact_labels import (
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
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_REGION_DATA_DIR,
    )
    p.add_argument("--regions", nargs="+", default=REGIONS_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty Git worktree for local smoke/debug builds only.",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    git_info = git_worktree_provenance(ROOT)
    if git_info["git_dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "formal relation builds require a clean Git worktree"
        )

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
        events = canonicalize_events(pd.read_csv(events_path))
        n_events = len(events)
        print(f"  events (canonical identity order): {n_events}")
        event_start_idx = events["start_idx"].to_numpy(dtype=np.int64)
        event_id = events["_event_id"].astype(str).to_numpy()

        samples_path = args.out_dir / f"{region_name}_samples.h5"
        with h5py.File(samples_path, "r") as samples:
            protocol = str(samples.attrs.get("protocol_version", "legacy"))
            schema = int(samples.attrs.get("cache_schema_version", -1))
            if (
                protocol != PROTOCOL_VERSION
                or schema != CACHE_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    f"{samples_path} uses protocol/schema "
                    f"{protocol!r}/{schema}; required "
                    f"{PROTOCOL_VERSION!r}/{CACHE_SCHEMA_VERSION}"
                )
            cached_starts = samples["event_start_idx"][:]
            if "event_id" not in samples:
                raise RuntimeError(f"{samples_path} is missing event_id")
            cached_event_id = samples["event_id"].asstr()[:]
            bundle_id = str(samples.attrs.get("bundle_id", ""))
            traffic_cache_id = str(
                samples.attrs.get("traffic_cache_id", "")
            )
            samples_cache_id = str(
                samples.attrs.get("samples_cache_id", "")
            )
            cache_mode = str(samples.attrs.get("cache_mode", ""))
        if not np.array_equal(event_start_idx, cached_starts) or not np.array_equal(
            event_id, cached_event_id
        ):
            raise RuntimeError(
                f"event ordering mismatch between {events_path} and {samples_path}"
            )
        if not bundle_id or not traffic_cache_id or not samples_cache_id:
            raise RuntimeError(f"{samples_path} is missing v3 cache identities")

        t0 = time.time()
        rel_feat = compute_event_rel_feat(events, region_meta, region_node_idx, dis_matrix)
        print(f"  rel_feat shape: {rel_feat.shape}  (mem={rel_feat.nbytes / 1e6:.0f} MB)  "
              f"in {time.time() - t0:.1f}s")

        out_path = args.out_dir / f"{region_name}_event_rel_feat.npz"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite {out_path}; pass --overwrite"
            )
        relation_cache_id = f"relation-{uuid.uuid4().hex}"
        relation_cache_mode = (
            cache_mode
            if cache_mode != "formal"
            else resolve_cache_mode(
                smoke_events=0,
                git_dirty=bool(git_info["git_dirty"]),
            )
        )
        relation_manifest = source_manifest([events_path, samples_path])
        tmp_path = out_path.with_name(
            f".{out_path.stem}.{uuid.uuid4().hex}.tmp.npz"
        )
        try:
            np.savez_compressed(
                tmp_path,
                rel_feat=rel_feat,
                n_events=np.int32(n_events),
                max_road_km=np.float32(MAX_ROAD_KM),
                event_start_idx=event_start_idx,
                event_id=event_id,
                protocol_version=np.array(PROTOCOL_VERSION),
                cache_schema_version=np.int32(CACHE_SCHEMA_VERSION),
                bundle_id=np.array(bundle_id),
                traffic_cache_id=np.array(traffic_cache_id),
                samples_cache_id=np.array(samples_cache_id),
                relation_cache_id=np.array(relation_cache_id),
                artifact_id=np.array(relation_cache_id),
                cache_role=np.array("relation"),
                cache_mode=np.array(relation_cache_mode),
                samples_sha256=np.array(sha256_file(samples_path)),
                source_manifest_json=np.array(
                    json.dumps(relation_manifest, sort_keys=True)
                ),
                git_sha=np.array(git_info["git_sha"]),
                git_branch=np.array(git_info["git_branch"]),
                git_dirty=np.bool_(git_info["git_dirty"]),
            )
            os.replace(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        sz = out_path.stat().st_size / (1024 ** 2)
        print(f"  saved -> {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
