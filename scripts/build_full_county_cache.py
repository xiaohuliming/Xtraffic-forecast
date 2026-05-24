#!/usr/bin/env python3
"""Build DIST-Net data cache per design doc §3.

For each region produces 2 HDF5 files under outputs/dist_net/region_data/:
  {region}_traffic.h5  - continuous regional time series + statistics
  {region}_samples.h5  - per-sample metadata (event-anchored)

Continuous traffic file (loaded once per region at training time):
  /flow_series_imputed (T, N, 3)  float32  linear-interpolated raw flow
  /flow_mask           (T, N, 3)  bool     1=originally observed
  /time_enc            (T, 5)     float32  sin/cos(tod)+sin/cos(dow)+holiday
  /static_meta         (N, C_meta) float32 sensor static features (z-normalized)
  /region_idx          (N,)       int64    index in global 16972 space
  /baseline_median     (2, 288, N, 3) float32  per (day_kind, tod)
  /baseline_scale      (2, 288, N, 3) float32  robust MAD-based scale
  /times_ns            (T,)       int64    nanosecond timestamps

Sample metadata file (one entry per (event, offset)):
  /sample_start        (S,)       int64    anchor timestep (= event.start_idx + offset)
  /split               (S,)       int8     0=train, 1=val, 2=test
  /event_idx           (S,)       int32    primary event index for this sample
  /anchor_region_idx   (S,)       int32    primary event's anchor node
  /sample_offset       (S,)       int8     which offset (0/6/12) within the event
  /incident_feat       (S, M_max, C_e) float32  active incidents in [t-24, t]
  /incident_mask       (S, M_max) bool
  /n_active_incidents  (S,)       int32
  /affected_mask       (S, N)     bool     union over active events' affected nodes
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from build_impact_labels import (
    CHANNEL_FLOW,
    CHANNEL_OCC,
    CHANNEL_SPEED,
    build_baseline_valid_mask,
    build_matches,
    build_robust_baseline,
    load_incidents_2023,
    load_region_traffic,
    load_sensor_meta,
    region_specs,
)

REGIONS_DEFAULT = ["Alameda", "ContraCosta", "Orange"]

INCIDENT_TYPES = ["1141", "Fire", "NoInj", "UnknInj",
                  "Hazard", "AHazard", "CarFire", "Other"]
TYPE_TO_IDX = {t: i for i, t in enumerate(INCIDENT_TYPES)}
N_TYPES = len(INCIDENT_TYPES)
C_E = N_TYPES + 5  # one-hot type + [duration, severity, lat, lon, abs_pm]

# US federal holidays in 2023 (matches export_to_igstgnn.py)
US_FEDERAL_HOLIDAYS_2023 = {
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29", "2023-06-19",
    "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-10", "2023-11-23",
    "2023-12-25",
}

DEFAULT_M_MAX = 32
DEFAULT_T_H = 12
DEFAULT_T_P = 12
DEFAULT_SAMPLE_OFFSETS = (0, 6, 12)
DEFAULT_ACTIVE_WINDOW = 24  # 2 hours = 24 5-min steps
DEFAULT_META_COLUMNS = [
    "Lat", "Lng", "Abs PM", "Fwy",
    # numerics from sensor_meta_feature
]

META_NUMERIC_DEFAULT = ["Lat", "Lng", "Abs PM", "Fwy"]
META_CATEGORICAL_DEFAULT = ["Direction"]


# ---------- helpers ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("archive"))
    p.add_argument("--event-root", type=Path,
                   default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"))
    p.add_argument("--node-label-dir", type=Path,
                   default=Path("outputs/impact_labels"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/dist_net/region_data"))
    p.add_argument("--regions", nargs="+", default=REGIONS_DEFAULT)
    p.add_argument("--t-h", type=int, default=DEFAULT_T_H)
    p.add_argument("--t-p", type=int, default=DEFAULT_T_P)
    p.add_argument("--m-max", type=int, default=DEFAULT_M_MAX)
    p.add_argument("--active-window", type=int, default=DEFAULT_ACTIVE_WINDOW,
                   help="Active-incident lookback in 5-min steps (default 24 = 2 hours)")
    p.add_argument("--sample-offsets", nargs="+", type=int, default=list(DEFAULT_SAMPLE_OFFSETS))
    p.add_argument("--candidate-pm-radius", type=float, default=5.0)
    p.add_argument("--anchor-pm-radius", type=float, default=2.0)
    p.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    p.add_argument("--baseline-train-frac", type=float, default=0.70,
                   help="Use first this fraction of year to fit baseline statistics")
    p.add_argument("--min-baseline-count", type=int, default=8)
    p.add_argument("--smoke-events", type=int, default=0,
                   help=">0: process only first N events per region (for testing)")
    p.add_argument("--samples-only", action="store_true",
                   help="Skip traffic.h5 build, only regenerate samples.h5 "
                        "(traffic must already exist).")
    p.add_argument("--split-train-end", type=float, default=0.70,
                   help="Temporal fraction at which train -> val")
    p.add_argument("--split-val-end", type=float, default=0.85,
                   help="Temporal fraction at which val -> test")
    return p.parse_args()


def split_code(sample_start: int, total_steps: int,
               train_end: float, val_end: float) -> int:
    frac = sample_start / total_steps
    if frac < train_end:
        return 0  # train
    if frac < val_end:
        return 1  # val
    return 2      # test


def compute_time_enc(times: pd.DatetimeIndex) -> np.ndarray:
    tod = times.hour.to_numpy() * 60 + times.minute.to_numpy()
    tod = tod / (24 * 60)  # [0,1)
    dow = times.dayofweek.to_numpy() / 7.0
    holiday = np.array([1.0 if d in US_FEDERAL_HOLIDAYS_2023 else 0.0
                        for d in times.strftime("%Y-%m-%d")], dtype=np.float32)
    enc = np.stack([
        np.sin(2 * np.pi * tod),
        np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow),
        np.cos(2 * np.pi * dow),
        holiday,
    ], axis=-1).astype(np.float32)
    return enc  # (T, 5)


def impute_flow(traffic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation along time axis per (node, channel).
    Returns (imputed, mask) where mask is True for originally-observed values."""
    T, N, C = traffic.shape
    mask = np.isfinite(traffic)
    imputed = traffic.copy()
    t_arange = np.arange(T)
    for n in range(N):
        for c in range(C):
            col = imputed[:, n, c]
            m = mask[:, n, c]
            if m.all():
                continue
            if not m.any():
                # node-channel entirely missing; fill 0 (caller relies on mask=0)
                imputed[:, n, c] = 0.0
                continue
            # linear interpolation over time
            imputed[:, n, c] = np.interp(t_arange, t_arange[m], col[m])
    return imputed.astype(np.float32), mask


def build_static_meta(region_meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Pack a subset of static sensor features into (N, C_meta)."""
    numeric_cols = list(META_NUMERIC_DEFAULT)
    feats = []
    feat_names: list[str] = []
    for col in numeric_cols:
        x = pd.to_numeric(region_meta[col], errors="coerce").to_numpy(dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.nanmean(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0.0)
        if x.std() > 1e-6:
            x = (x - x.mean()) / x.std()
        feats.append(x)
        feat_names.append(col)
    # Categorical: Direction one-hot (N, E, S, W)
    direction_chars = ["N", "E", "S", "W"]
    direction_series = region_meta["Direction"].fillna("").astype(str).str.strip().str.upper().str[:1]
    for ch in direction_chars:
        feats.append((direction_series == ch).to_numpy(dtype=np.float32))
        feat_names.append(f"dir_{ch}")
    static_meta = np.stack(feats, axis=-1).astype(np.float32)  # (N, C_meta)
    return static_meta, feat_names


def normalize_type(type_str: object) -> str:
    if pd.isna(type_str):
        return "Other"
    t = str(type_str).strip()
    return t if t in TYPE_TO_IDX else "Other"


def _safe_float(value, default: float = 0.0) -> float:
    """NaN-safe float coercion. NaN is truthy in Python so `x or default` fails."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def event_incident_feature(row: pd.Series) -> np.ndarray:
    """Pack one event's row into a fixed (C_e,) vector. All entries finite."""
    feat = np.zeros(C_E, dtype=np.float32)
    feat[TYPE_TO_IDX[normalize_type(row.get("primary_type"))]] = 1.0
    feat[N_TYPES + 0] = _safe_float(row.get("duration_min")) / 60.0  # hours
    feat[N_TYPES + 1] = _safe_float(row.get("severity_any_z_auc_topk"))
    feat[N_TYPES + 2] = _safe_float(row.get("latitude"))
    feat[N_TYPES + 3] = _safe_float(row.get("longitude"))
    feat[N_TYPES + 4] = _safe_float(row.get("incident_abs_pm"))
    return feat


def build_affected_lookup(node_labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map incident_id -> array of region_node_idx where affected==1."""
    affected = node_labels[node_labels["affected"] == 1]
    lookup: dict[str, np.ndarray] = {}
    for inc_id, grp in affected.groupby("incident_id"):
        lookup[str(inc_id)] = grp["region_node_idx"].to_numpy(dtype=np.int32)
    return lookup


def event_affected_mask(N: int, incident_ids: list[str],
                        affected_lookup: dict[str, np.ndarray]) -> np.ndarray:
    """OR-aggregate affected node indices across all incident_ids of this event."""
    mask = np.zeros(N, dtype=bool)
    for inc_id in incident_ids:
        idx = affected_lookup.get(str(inc_id))
        if idx is not None and idx.size:
            mask[idx] = True
    return mask


def parse_incident_ids_field(s: object) -> list[str]:
    """event_labels.csv stores incident_ids as a comma-separated string."""
    if pd.isna(s) or s is None:
        return []
    text = str(s).strip()
    if not text:
        return []
    return [tok.strip() for tok in text.split(",") if tok.strip()]


# ---------- main per-region build ----------

def build_traffic_h5(region_name: str, region_meta: pd.DataFrame,
                     traffic: np.ndarray, times: pd.DatetimeIndex,
                     baseline: np.ndarray, scale: np.ndarray,
                     region_node_idx: np.ndarray, out_path: Path) -> None:
    print(f"  [{region_name}] imputing flow (linear interp) ...", flush=True)
    t0 = time.time()
    imputed, mask = impute_flow(traffic)
    print(f"    imputed in {time.time() - t0:.1f}s; "
          f"missing fraction (original): {(~mask).mean():.3%}")

    print(f"  [{region_name}] computing time_enc ...", flush=True)
    time_enc = compute_time_enc(times)

    print(f"  [{region_name}] building static_meta ...", flush=True)
    static_meta, feat_names = build_static_meta(region_meta)
    print(f"    C_meta = {static_meta.shape[1]} ({feat_names})")

    print(f"  [{region_name}] writing {out_path} ...", flush=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("flow_series_imputed", data=imputed,
                         compression="lzf", chunks=(min(288, imputed.shape[0]), imputed.shape[1], 3))
        f.create_dataset("flow_mask", data=mask,
                         compression="lzf", chunks=(min(288, mask.shape[0]), mask.shape[1], 3))
        f.create_dataset("time_enc", data=time_enc, compression="lzf")
        f.create_dataset("static_meta", data=static_meta, compression="lzf")
        f.create_dataset("region_idx", data=region_node_idx.astype(np.int64))
        f.create_dataset("baseline_median", data=baseline, compression="lzf")
        f.create_dataset("baseline_scale", data=scale, compression="lzf")
        f.create_dataset("times_ns", data=times.asi8.astype(np.int64))
        f.attrs["region_name"] = region_name
        f.attrs["meta_feature_names"] = np.array(feat_names, dtype="S32")
        f.attrs["N"] = int(static_meta.shape[0])
        f.attrs["T"] = int(traffic.shape[0])
        f.attrs["channels"] = np.array(["flow", "occupancy", "speed"], dtype="S16")
    sz = out_path.stat().st_size / (1024 ** 3)
    print(f"    saved {sz:.2f} GB")


def build_samples_h5(region_name: str, region_node_idx: np.ndarray,
                     events: pd.DataFrame, affected_lookup: dict[str, np.ndarray],
                     total_steps: int, args: argparse.Namespace,
                     out_path: Path) -> None:
    N = region_node_idx.size
    n_events = len(events)
    if args.smoke_events > 0:
        events = events.head(args.smoke_events).copy()
        n_events = len(events)
        print(f"  [{region_name}] SMOKE: limited to first {n_events} events")

    # Sort events by start_idx for efficient active-event lookup
    events = events.sort_values("start_idx").reset_index(drop=True)
    starts_sorted = events["start_idx"].to_numpy(dtype=np.int64)

    # Pre-build per-event feature vec + affected mask
    print(f"  [{region_name}] pre-computing per-event features + affected_mask...", flush=True)
    t0 = time.time()
    event_feats = np.zeros((n_events, C_E), dtype=np.float32)
    event_affected = np.zeros((n_events, N), dtype=bool)
    for i, row in enumerate(events.itertuples(index=False)):
        event_feats[i] = event_incident_feature(events.iloc[i])
        inc_ids = parse_incident_ids_field(getattr(row, "incident_ids"))
        event_affected[i] = event_affected_mask(N, inc_ids, affected_lookup)
    print(f"    pre-compute in {time.time() - t0:.1f}s; "
          f"mean affected/event: {event_affected.sum(axis=1).mean():.1f}")

    # Build samples
    print(f"  [{region_name}] generating samples ({len(args.sample_offsets)} offsets)...", flush=True)
    t0 = time.time()
    sample_start_list: list[int] = []
    split_list: list[int] = []
    event_idx_list: list[int] = []
    anchor_idx_list: list[int] = []
    offset_list: list[int] = []
    inc_feat_list: list[np.ndarray] = []
    inc_mask_list: list[np.ndarray] = []
    n_active_list: list[int] = []
    affected_list: list[np.ndarray] = []
    active_event_idx_list: list[np.ndarray] = []   # (M_max,) int32 per sample,
                                                    # -1 marks padding

    # Bounds: need T_h history and T_p future
    min_start = args.t_h            # sample_start at t means hist is [t-T_h+1..t]
    max_start = total_steps - args.t_p - 1  # need [t+1..t+T_p] all in-range

    max_active_seen = 0
    n_truncated = 0

    for ev_idx in range(n_events):
        ev_start = int(starts_sorted[ev_idx])
        anchor_idx = int(events.iloc[ev_idx]["anchor_region_idx"])
        for offset in args.sample_offsets:
            sample_start = ev_start + int(offset)
            if sample_start < min_start or sample_start > max_start:
                continue

            # Find all events with start_idx in [sample_start - active_window + 1, sample_start]
            lo = bisect.bisect_left(starts_sorted, sample_start - args.active_window + 1)
            hi = bisect.bisect_right(starts_sorted, sample_start)
            active = np.arange(lo, hi, dtype=np.int64)
            n_active = active.size
            max_active_seen = max(max_active_seen, n_active)

            # Pack into (M_max, C_e). If too many, keep most recent.
            inc_feat = np.zeros((args.m_max, C_E), dtype=np.float32)
            inc_mask = np.zeros(args.m_max, dtype=bool)
            if n_active > args.m_max:
                # Keep the most recent (highest start_idx)
                chosen = active[-args.m_max:]
                n_truncated += 1
            else:
                chosen = active
            inc_feat[:chosen.size] = event_feats[chosen]
            inc_mask[:chosen.size] = True

            # Affected mask: union across all active events
            if chosen.size:
                aff = np.any(event_affected[chosen], axis=0)
            else:
                aff = np.zeros(N, dtype=bool)

            sample_start_list.append(sample_start)
            split_list.append(split_code(sample_start, total_steps,
                                          args.split_train_end, args.split_val_end))
            event_idx_list.append(ev_idx)
            anchor_idx_list.append(anchor_idx)
            offset_list.append(int(offset))
            inc_feat_list.append(inc_feat)
            inc_mask_list.append(inc_mask)
            n_active_list.append(int(chosen.size))
            affected_list.append(aff)

            active_idx = np.full(args.m_max, -1, dtype=np.int32)
            active_idx[:chosen.size] = chosen.astype(np.int32)
            active_event_idx_list.append(active_idx)

    n_samples = len(sample_start_list)
    print(f"    generated {n_samples} samples in {time.time() - t0:.1f}s")
    print(f"    max active events in any window: {max_active_seen}, "
          f"truncated samples (>M_max={args.m_max}): {n_truncated}")

    if n_samples == 0:
        print(f"    WARNING: no samples generated for {region_name}")
        return

    sample_start_arr = np.array(sample_start_list, dtype=np.int64)
    split_arr = np.array(split_list, dtype=np.int8)
    event_idx_arr = np.array(event_idx_list, dtype=np.int32)
    anchor_idx_arr = np.array(anchor_idx_list, dtype=np.int32)
    offset_arr = np.array(offset_list, dtype=np.int8)
    inc_feat_arr = np.stack(inc_feat_list, axis=0)             # (S, M_max, C_e)
    inc_mask_arr = np.stack(inc_mask_list, axis=0)             # (S, M_max)
    n_active_arr = np.array(n_active_list, dtype=np.int32)
    affected_arr = np.stack(affected_list, axis=0)             # (S, N)
    active_event_idx_arr = np.stack(active_event_idx_list, axis=0)  # (S, M_max) int32

    split_counts = {0: int((split_arr == 0).sum()),
                    1: int((split_arr == 1).sum()),
                    2: int((split_arr == 2).sum())}
    print(f"    split counts: train={split_counts[0]} val={split_counts[1]} test={split_counts[2]}")

    print(f"  [{region_name}] writing {out_path} ...", flush=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("sample_start", data=sample_start_arr)
        f.create_dataset("split", data=split_arr)
        f.create_dataset("event_idx", data=event_idx_arr)
        f.create_dataset("anchor_region_idx", data=anchor_idx_arr)
        f.create_dataset("sample_offset", data=offset_arr)
        f.create_dataset("incident_feat", data=inc_feat_arr, compression="lzf")
        f.create_dataset("incident_mask", data=inc_mask_arr, compression="lzf")
        f.create_dataset("n_active_incidents", data=n_active_arr)
        f.create_dataset("affected_mask", data=affected_arr, compression="lzf")
        f.create_dataset("active_event_idx", data=active_event_idx_arr,
                         compression="lzf")
        f.attrs["region_name"] = region_name
        f.attrs["N"] = int(N)
        f.attrs["M_max"] = int(args.m_max)
        f.attrs["C_e"] = int(C_E)
        f.attrs["T_h"] = int(args.t_h)
        f.attrs["T_p"] = int(args.t_p)
        f.attrs["active_window"] = int(args.active_window)
        f.attrs["incident_types"] = np.array(INCIDENT_TYPES, dtype="S16")
        f.attrs["n_events"] = int(n_events)
        f.attrs["n_samples"] = int(n_samples)
        f.attrs["n_truncated_active"] = int(n_truncated)
        f.attrs["max_active_seen"] = int(max_active_seen)
        f.attrs["split_train_end"] = float(args.split_train_end)
        f.attrs["split_val_end"] = float(args.split_val_end)
    sz = out_path.stat().st_size / (1024 ** 2)
    print(f"    saved {sz:.1f} MB")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading global sensor metadata + incidents...")
    meta = load_sensor_meta(args.archive)
    inc = load_incidents_2023(args.archive)
    print(f"  global: {len(meta)} sensors, {len(inc)} incidents")

    specs = region_specs()

    for region_name in args.regions:
        if region_name not in specs:
            raise ValueError(f"Unknown region {region_name}; choices: {list(specs)}")
        region = specs[region_name]
        print(f"\n=== {region_name} ({region.county}) ===", flush=True)

        region_meta_full = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
        region_meta_full = region_meta_full.reset_index(drop=True)
        region_node_idx = region_meta_full["node_idx"].to_numpy(dtype=np.int64)
        N = region_node_idx.size
        print(f"  N = {N}")

        if args.samples_only:
            # Just need T (=total_steps) for split coding; pull from existing h5.
            out_traffic_pre = args.out_dir / f"{region_name}_traffic.h5"
            if not out_traffic_pre.exists():
                raise FileNotFoundError(
                    f"--samples-only requires {out_traffic_pre} to exist already"
                )
            with h5py.File(out_traffic_pre, "r") as _f:
                T = int(_f.attrs["T"])
            print(f"  [samples-only] T={T} from existing traffic.h5")
            traffic = None
            times = pd.date_range("2023-01-01", "2024-01-01", freq="5min", inclusive="left")
            baseline = scale = counts = matches = baseline_valid = train_valid = None
        else:
            print(f"  loading traffic time series...", flush=True)
            t0 = time.time()
            traffic, times = load_region_traffic(args.archive, region_node_idx)
            T = traffic.shape[0]
            print(f"    traffic shape {traffic.shape} loaded in {time.time() - t0:.1f}s")
            print(f"    traffic NaN fraction (raw): {np.isnan(traffic).mean():.3%}")

            # Build baseline statistics (fitting on first split_train_end of year, with incident-window masking)
            print(f"  building baseline statistics (training portion only) ...", flush=True)
            t0 = time.time()
            matches = build_matches(
                inc=inc, region_meta=region_meta_full, times=times,
                candidate_pm_radius=args.candidate_pm_radius,
                anchor_pm_radius=args.anchor_pm_radius,
                baseline_mask_extra_steps=args.baseline_mask_extra_steps,
            )
            baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)
            train_cutoff = int(T * args.baseline_train_frac)
            train_valid = baseline_valid.copy()
            train_valid[train_cutoff:, :] = False
            baseline, scale, counts = build_robust_baseline(
                traffic=traffic, times=times, baseline_valid=train_valid,
                min_count=args.min_baseline_count,
            )
            nan_in_baseline = np.isnan(baseline).mean()
            print(f"    baseline median computed in {time.time() - t0:.1f}s; "
                  f"slots with NaN: {nan_in_baseline:.3%}")
            # Replace NaN baseline with global median per (node, channel)
            for c in range(baseline.shape[-1]):
                for n in range(baseline.shape[2]):
                    slab = baseline[:, :, n, c]
                    if np.any(np.isfinite(slab)):
                        fill = np.nanmedian(slab)
                    else:
                        fill = 0.0
                    slab[~np.isfinite(slab)] = fill
                    sslab = scale[:, :, n, c]
                    if np.any(np.isfinite(sslab)):
                        sfill = np.nanmedian(sslab)
                    else:
                        sfill = 1.0
                    sslab[~np.isfinite(sslab)] = sfill

            out_traffic = args.out_dir / f"{region_name}_traffic.h5"
            build_traffic_h5(
                region_name=region_name, region_meta=region_meta_full,
                traffic=traffic, times=times, baseline=baseline, scale=scale,
                region_node_idx=region_node_idx, out_path=out_traffic,
            )

        # set path for meta_json regardless of branch
        out_traffic = args.out_dir / f"{region_name}_traffic.h5"

        print(f"  loading event_labels + node_labels...", flush=True)
        events = pd.read_csv(args.event_root / region_name / "event_labels.csv")
        node_labels = pd.read_csv(args.node_label_dir / region_name / "node_labels.csv")
        print(f"    events: {len(events)}, node_label rows: {len(node_labels)}")
        affected_lookup = build_affected_lookup(node_labels)
        print(f"    affected_lookup: {len(affected_lookup)} unique incident_ids")

        out_samples = args.out_dir / f"{region_name}_samples.h5"
        build_samples_h5(
            region_name=region_name, region_node_idx=region_node_idx,
            events=events, affected_lookup=affected_lookup,
            total_steps=T, args=args, out_path=out_samples,
        )

        # Write a brief config note for this region
        meta_json = {
            "region_name": region_name,
            "county": region.county,
            "N": int(N),
            "T": int(T),
            "C_e": int(C_E),
            "M_max": int(args.m_max),
            "T_h": int(args.t_h),
            "T_p": int(args.t_p),
            "active_window": int(args.active_window),
            "sample_offsets": list(args.sample_offsets),
            "split_train_end": float(args.split_train_end),
            "split_val_end": float(args.split_val_end),
            "baseline_train_frac": float(args.baseline_train_frac),
            "incident_types": INCIDENT_TYPES,
            "traffic_file": str(out_traffic),
            "samples_file": str(out_samples),
        }
        (args.out_dir / f"{region_name}_config.json").write_text(
            json.dumps(meta_json, indent=2), encoding="utf-8"
        )
        if not args.samples_only:
            del traffic, baseline, scale, matches, baseline_valid, train_valid

    print(f"\nall regions built. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
