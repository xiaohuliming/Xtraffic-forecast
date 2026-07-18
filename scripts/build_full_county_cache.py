#!/usr/bin/env python3
"""Build the versioned no-leakage XTraffic cache.

For each region produces 2 HDF5 files under
outputs/dist_net/region_data_v2_noleak/:
  {region}_traffic.h5  - continuous regional time series + statistics
  {region}_samples.h5  - per-sample metadata (event-anchored)

Continuous traffic file (loaded once per region at training time):
  /flow_series_imputed (T, N, 3)  float32  causal forward-filled raw flow
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
  /primary_incident_slot (S,)     int16    anchor event's slot in incident_feat
  /analysis_labels/affected_mask  (S, N) bool post-event evaluation label
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_impact_labels import (
    build_robust_baseline,
    load_region_traffic,
    load_sensor_meta,
    region_specs,
)
from dist_net.protocol import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    build_deployable_incident_feature,
    build_training_fit_mask,
    causal_forward_fill,
    classify_event_group,
    classify_forecast_window,
    git_worktree_provenance,
    select_active_events,
)

REGIONS_DEFAULT = ["Alameda", "ContraCosta", "Orange"]

INCIDENT_TYPES = ["1141", "Fire", "NoInj", "UnknInj",
                  "Hazard", "AHazard", "CarFire", "Other"]
TYPE_TO_IDX = {t: i for i, t in enumerate(INCIDENT_TYPES)}
N_TYPES = len(INCIDENT_TYPES)
C_E = N_TYPES + 4  # one-hot type + [elapsed_hours, lat, lon, abs_pm]

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
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/dist_net/region_data_v2_noleak"),
    )
    p.add_argument("--regions", nargs="+", default=REGIONS_DEFAULT)
    p.add_argument("--t-h", type=int, default=DEFAULT_T_H)
    p.add_argument("--t-p", type=int, default=DEFAULT_T_P)
    p.add_argument("--m-max", type=int, default=DEFAULT_M_MAX)
    p.add_argument("--active-window", type=int, default=DEFAULT_ACTIVE_WINDOW,
                   help="Active-incident lookback in 5-min steps (default 24 = 2 hours)")
    p.add_argument("--sample-offsets", nargs="+", type=int, default=list(DEFAULT_SAMPLE_OFFSETS))
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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in the v2 output directory if they already exist.",
    )
    return p.parse_args()


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
    """Causally fill missing values and retain the original observation mask."""
    return causal_forward_fill(traffic)


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


def event_incident_feature(row: pd.Series, sample_start: int) -> np.ndarray:
    """Pack only fields known by the forecast origin."""
    return build_deployable_incident_feature(
        row,
        anchor_step=sample_start,
        type_to_idx=TYPE_TO_IDX,
    )


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
    """Parse comma- or semicolon-separated incident identifiers."""
    if pd.isna(s) or s is None:
        return []
    text = str(s).strip()
    if not text:
        return []
    normalized = text.replace(";", ",")
    return [tok.strip() for tok in normalized.split(",") if tok.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(paths: list[Path]) -> list[dict[str, object]]:
    manifest = []
    for path in paths:
        resolved = path.resolve()
        try:
            display = str(resolved.relative_to(ROOT))
        except ValueError:
            display = str(resolved)
        manifest.append(
            {
                "path": display,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return manifest


def write_protocol_attrs(
    h5: h5py.File,
    train_end: int,
    val_end: int,
    baseline_fit_end: int,
    manifest: list[dict[str, object]],
    git_info: dict[str, object],
) -> None:
    h5.attrs["cache_schema_version"] = CACHE_SCHEMA_VERSION
    h5.attrs["protocol_version"] = PROTOCOL_VERSION
    h5.attrs["train_end"] = int(train_end)
    h5.attrs["val_end"] = int(val_end)
    h5.attrs["baseline_fit_start"] = 0
    h5.attrs["baseline_fit_end"] = int(baseline_fit_end)
    h5.attrs["imputation_method"] = "causal_forward_fill_zero_prefix"
    h5.attrs["baseline_method"] = "label_free_train_only_robust_median"
    h5.attrs["source_manifest_json"] = json.dumps(manifest, sort_keys=True)
    for key, value in git_info.items():
        h5.attrs[key] = value


# ---------- main per-region build ----------

def build_traffic_h5(region_name: str, region_meta: pd.DataFrame,
                     traffic: np.ndarray, times: pd.DatetimeIndex,
                     baseline: np.ndarray, scale: np.ndarray,
                     region_node_idx: np.ndarray, out_path: Path,
                     train_end: int, val_end: int, baseline_fit_end: int,
                     manifest: list[dict[str, object]],
                     git_info: dict[str, object]) -> None:
    print(f"  [{region_name}] imputing flow (causal forward fill) ...", flush=True)
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
        write_protocol_attrs(
            f,
            train_end=train_end,
            val_end=val_end,
            baseline_fit_end=baseline_fit_end,
            manifest=manifest,
            git_info=git_info,
        )
    sz = out_path.stat().st_size / (1024 ** 3)
    print(f"    saved {sz:.2f} GB")


def build_samples_h5(region_name: str, region_node_idx: np.ndarray,
                     events: pd.DataFrame, affected_lookup: dict[str, np.ndarray],
                     total_steps: int, args: argparse.Namespace,
                     out_path: Path, train_end: int, val_end: int,
                     baseline_fit_end: int,
                     manifest: list[dict[str, object]],
                     git_info: dict[str, object]) -> None:
    N = region_node_idx.size
    n_events = len(events)
    if args.smoke_events > 0:
        events = events.head(args.smoke_events).copy()
        n_events = len(events)
        print(f"  [{region_name}] SMOKE: limited to first {n_events} events")

    # Sort events by start_idx for efficient active-event lookup
    events = events.sort_values("start_idx").reset_index(drop=True)
    starts_sorted = events["start_idx"].to_numpy(dtype=np.int64)

    # Pre-build analysis-only affected masks. Deployable incident features are
    # built per sample because elapsed time changes with the forecast origin.
    print(f"  [{region_name}] pre-computing analysis affected masks...", flush=True)
    t0 = time.time()
    event_affected = np.zeros((n_events, N), dtype=bool)
    for i, row in enumerate(events.itertuples(index=False)):
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
    primary_slot_list: list[int] = []

    # Bounds: need T_h history and T_p future
    min_start = args.t_h - 1        # sample_start at t means hist is [t-T_h+1..t]
    max_start = total_steps - args.t_p - 1  # need [t+1..t+T_p] all in-range

    max_active_seen = 0
    n_truncated = 0
    n_mixed_events_dropped = 0

    for ev_idx in range(n_events):
        ev_start = int(starts_sorted[ev_idx])
        anchor_idx = int(events.iloc[ev_idx]["anchor_region_idx"])
        group_split = classify_event_group(
            event_start=ev_start,
            offsets=args.sample_offsets,
            history=args.t_h,
            horizon=args.t_p,
            train_end=train_end,
            val_end=val_end,
            total_steps=total_steps,
        )
        if group_split is None:
            n_mixed_events_dropped += 1
            continue

        for offset in args.sample_offsets:
            sample_start = ev_start + int(offset)
            if sample_start < min_start or sample_start > max_start:
                continue
            sample_split = classify_forecast_window(
                sample_start,
                args.t_p,
                train_end,
                val_end,
                total_steps,
            )
            if sample_split != group_split:
                continue

            # Find all events with start_idx in [sample_start - active_window + 1, sample_start]
            lo = bisect.bisect_left(starts_sorted, sample_start - args.active_window + 1)
            hi = bisect.bisect_right(starts_sorted, sample_start)
            active = np.arange(lo, hi, dtype=np.int64)
            if ev_idx not in active:
                active = np.sort(np.append(active, ev_idx))
            n_active = active.size
            max_active_seen = max(max_active_seen, n_active)

            # Pack into (M_max, C_e), retaining the sample's primary event even
            # when a dense incident window must be truncated.
            inc_feat = np.zeros((args.m_max, C_E), dtype=np.float32)
            inc_mask = np.zeros(args.m_max, dtype=bool)
            chosen, primary_slot = select_active_events(
                active, primary_index=ev_idx, max_events=args.m_max
            )
            if n_active > args.m_max:
                n_truncated += 1
            for slot, chosen_idx in enumerate(chosen):
                inc_feat[slot] = event_incident_feature(
                    events.iloc[int(chosen_idx)], sample_start
                )
            inc_mask[:chosen.size] = True

            # Affected mask: union across all active events
            if chosen.size:
                aff = np.any(event_affected[chosen], axis=0)
            else:
                aff = np.zeros(N, dtype=bool)

            sample_start_list.append(sample_start)
            split_list.append(group_split)
            event_idx_list.append(ev_idx)
            anchor_idx_list.append(anchor_idx)
            offset_list.append(int(offset))
            inc_feat_list.append(inc_feat)
            inc_mask_list.append(inc_mask)
            n_active_list.append(int(chosen.size))
            affected_list.append(aff)
            primary_slot_list.append(primary_slot)

            active_idx = np.full(args.m_max, -1, dtype=np.int32)
            active_idx[:chosen.size] = chosen.astype(np.int32)
            active_event_idx_list.append(active_idx)

    n_samples = len(sample_start_list)
    print(f"    generated {n_samples} samples in {time.time() - t0:.1f}s")
    print(f"    max active events in any window: {max_active_seen}, "
          f"truncated samples (>M_max={args.m_max}): {n_truncated}")
    print(f"    events dropped to keep event groups disjoint: {n_mixed_events_dropped}")

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
    primary_slot_arr = np.asarray(primary_slot_list, dtype=np.int16)

    split_counts = {0: int((split_arr == 0).sum()),
                    1: int((split_arr == 1).sum()),
                    2: int((split_arr == 2).sum())}
    print(f"    split counts: train={split_counts[0]} val={split_counts[1]} test={split_counts[2]}")

    print(f"  [{region_name}] writing {out_path} ...", flush=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("sample_start", data=sample_start_arr)
        f.create_dataset("split", data=split_arr)
        f.create_dataset("event_idx", data=event_idx_arr)
        f.create_dataset("event_start_idx", data=starts_sorted)
        f.create_dataset("anchor_region_idx", data=anchor_idx_arr)
        f.create_dataset("sample_offset", data=offset_arr)
        f.create_dataset("incident_feat", data=inc_feat_arr, compression="lzf")
        f.create_dataset("incident_mask", data=inc_mask_arr, compression="lzf")
        f.create_dataset("n_active_incidents", data=n_active_arr)
        f.create_dataset("active_event_idx", data=active_event_idx_arr,
                         compression="lzf")
        f.create_dataset("primary_incident_slot", data=primary_slot_arr)
        analysis = f.create_group("analysis_labels")
        analysis.create_dataset("affected_mask", data=affected_arr, compression="lzf")
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
        f.attrs["n_mixed_events_dropped"] = int(n_mixed_events_dropped)
        f.attrs["split_train_end"] = float(args.split_train_end)
        f.attrs["split_val_end"] = float(args.split_val_end)
        f.attrs["incident_feature_names"] = np.array(
            [f"type_{name}" for name in INCIDENT_TYPES]
            + ["elapsed_hours", "latitude", "longitude", "incident_abs_pm"],
            dtype="S32",
        )
        f.attrs["affected_mask_role"] = "analysis_only_post_event_label"
        write_protocol_attrs(
            f,
            train_end=train_end,
            val_end=val_end,
            baseline_fit_end=baseline_fit_end,
            manifest=manifest,
            git_info=git_info,
        )
    sz = out_path.stat().st_size / (1024 ** 2)
    print(f"    saved {sz:.1f} MB")


def main() -> None:
    args = parse_args()
    if not 0 < args.split_train_end < args.split_val_end < 1:
        raise ValueError("split fractions must satisfy 0 < train < val < 1")
    if not 0 < args.baseline_train_frac <= args.split_train_end:
        raise ValueError(
            "baseline_train_frac must be positive and cannot exceed split_train_end"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing: list[Path] = []
        for region_name in args.regions:
            candidates = [
                args.out_dir / f"{region_name}_samples.h5",
                args.out_dir / f"{region_name}_config.json",
            ]
            if not args.samples_only:
                candidates.append(
                    args.out_dir / f"{region_name}_traffic.h5"
                )
            existing.extend(path for path in candidates if path.exists())
        if existing:
            raise FileExistsError(
                "refusing to overwrite v2 cache files: "
                + ", ".join(str(path) for path in existing)
                + "; pass --overwrite"
            )

    print("loading global sensor metadata...")
    meta = load_sensor_meta(args.archive)
    print(f"  global: {len(meta)} sensors")
    git_info = git_worktree_provenance(ROOT)
    common_sources = [
        args.archive / "sensor_meta_feature.csv",
        args.archive / "node_order.npy",
        *[args.archive / f"p{month:02d}_done.npy" for month in range(1, 13)],
    ]
    print("hashing source traffic files for cache provenance...", flush=True)
    common_manifest = source_manifest(common_sources)

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

        event_path = args.event_root / region_name / "event_labels.csv"
        node_label_path = args.node_label_dir / region_name / "node_labels.csv"
        sample_manifest = common_manifest + source_manifest(
            [event_path, node_label_path]
        )
        out_traffic = args.out_dir / f"{region_name}_traffic.h5"
        out_samples = args.out_dir / f"{region_name}_samples.h5"
        out_config = args.out_dir / f"{region_name}_config.json"

        if args.samples_only:
            if not out_traffic.exists():
                raise FileNotFoundError(
                    f"--samples-only requires {out_traffic} to exist already"
                )
            with h5py.File(out_traffic, "r") as _f:
                schema = int(_f.attrs.get("cache_schema_version", -1))
                protocol = str(_f.attrs.get("protocol_version", ""))
                if schema != CACHE_SCHEMA_VERSION or protocol != PROTOCOL_VERSION:
                    raise RuntimeError(
                        "--samples-only refuses a legacy or mismatched traffic cache"
                    )
                cached_manifest = json.loads(
                    str(_f.attrs.get("source_manifest_json", "[]"))
                )
                if cached_manifest != common_manifest:
                    raise RuntimeError(
                        "--samples-only source hashes differ from the existing "
                        "traffic cache; rebuild both files"
                    )
                T = int(_f.attrs["T"])
            print(f"  [samples-only] T={T} from existing traffic.h5")
            traffic = None
            baseline = scale = counts = train_valid = None
        else:
            print(f"  loading traffic time series...", flush=True)
            t0 = time.time()
            traffic, times = load_region_traffic(args.archive, region_node_idx)
            T = traffic.shape[0]
            print(f"    traffic shape {traffic.shape} loaded in {time.time() - t0:.1f}s")
            print(f"    traffic NaN fraction (raw): {np.isnan(traffic).mean():.3%}")

            print("  building label-free baseline statistics (training only) ...", flush=True)
            t0 = time.time()
            train_end = int(T * args.split_train_end)
            val_end = int(T * args.split_val_end)
            baseline_fit_end = int(T * args.baseline_train_frac)
            train_valid = build_training_fit_mask(
                traffic.shape[:2], train_end=baseline_fit_end
            )
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

            build_traffic_h5(
                region_name=region_name, region_meta=region_meta_full,
                traffic=traffic, times=times, baseline=baseline, scale=scale,
                region_node_idx=region_node_idx, out_path=out_traffic,
                train_end=train_end, val_end=val_end,
                baseline_fit_end=baseline_fit_end,
                manifest=common_manifest, git_info=git_info,
            )

        train_end = int(T * args.split_train_end)
        val_end = int(T * args.split_val_end)
        baseline_fit_end = int(T * args.baseline_train_frac)

        print(f"  loading event_labels + node_labels...", flush=True)
        events = pd.read_csv(event_path)
        node_labels = pd.read_csv(node_label_path)
        print(f"    events: {len(events)}, node_label rows: {len(node_labels)}")
        affected_lookup = build_affected_lookup(node_labels)
        print(f"    affected_lookup: {len(affected_lookup)} unique incident_ids")

        build_samples_h5(
            region_name=region_name, region_node_idx=region_node_idx,
            events=events, affected_lookup=affected_lookup,
            total_steps=T, args=args, out_path=out_samples,
            train_end=train_end, val_end=val_end,
            baseline_fit_end=baseline_fit_end,
            manifest=sample_manifest, git_info=git_info,
        )

        # Write a brief config note for this region
        meta_json = {
            "region_name": region_name,
            "county": region.county,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
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
            "train_end": train_end,
            "val_end": val_end,
            "baseline_train_frac": float(args.baseline_train_frac),
            "baseline_fit_start": 0,
            "baseline_fit_end": baseline_fit_end,
            "imputation_method": "causal_forward_fill_zero_prefix",
            "baseline_method": "label_free_train_only_robust_median",
            "incident_types": INCIDENT_TYPES,
            "incident_feature_names": [
                *[f"type_{name}" for name in INCIDENT_TYPES],
                "elapsed_hours",
                "latitude",
                "longitude",
                "incident_abs_pm",
            ],
            "traffic_file": str(out_traffic),
            "samples_file": str(out_samples),
            "traffic_source_manifest": common_manifest,
            "sample_source_manifest": sample_manifest,
            **git_info,
        }
        out_config.write_text(
            json.dumps(meta_json, indent=2), encoding="utf-8"
        )
        if not args.samples_only:
            del traffic, baseline, scale, train_valid

    print(f"\nall regions built. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
