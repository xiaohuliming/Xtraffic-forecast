"""Convert our .h5 cache → IGSTGNN .npy dict format for fair comparison.

IGSTGNN's data format per sample (dict):
  x_data           (T_h, N, 3) float32  — z-normalized traffic [flow, occ, speed]
  x_mask           (T_h, N, 1) bool     — raw flow observation mask
  y_data           (T_p, N, 1) float32  — z-normalized flow only
  y_mask           (T_p, N, 1) bool     — raw flow observation mask
  event_features   dict {Event Time, Description, Type, Holiday}
  event_position   int   — sensor index nearest event
  event_distances  (N, 3) float32 in [0, 1] — [euclid_kernel, road_kernel, downstream]
  durations        float — elapsed incident time at forecast origin, in 5-min steps
  _t_idx           int   — anchor time index

Mapping from our data (see scripts/build_full_county_cache.py + build_event_rel_feat.py):
  incident_feat[i]   (M_max=32, C_e=12):
    dims 0..7: one-hot type
    dim 8: elapsed time at forecast origin (hours)
    dims 9..11: lat / lon / abs_postmile
  event_rel_feat[k]  (N, 4):  global per-event distances
    dim 0: log(euclid_km + 1)
    dim 1: log(road_km + 1)
    dim 2: -1/0/+1  upstream/different/downstream
    dim 3: same_freeway 0/1
  active_event_idx[i] (M_max,) int32 — which global event each slot maps to

Type re-order (our index → IGSTGNN index):
  ours:    1141=0, Fire=1, NoInj=2, UnknInj=3, Hazard=4, AHazard=5, CarFire=6, Other=7
  theirs:  1141=0, AHazard=1, CarFire=2, Fire=3, Hazard=4, NoInj=5, Other=6, UnknInj=7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dist_net.data import RegionData
from dist_net.protocol import (
    DEFAULT_REGION_DATA_DIR,
    PROTOCOL_VERSION,
    forecast_window_bounds,
)

# Order in our incident_feat one-hot (dims 0..7)
OUR_TYPES = ["1141", "Fire", "NoInj", "UnknInj", "Hazard", "AHazard", "CarFire", "Other"]
# IGSTGNN's type_mapping.json
IGS_TYPE_CODE = {"1141": 0, "AHazard": 1, "CarFire": 2, "Fire": 3,
                 "Hazard": 4, "NoInj": 5, "Other": 6, "UnknInj": 7}

# Map our type → IGSTGNN's code
OUR_IDX_TO_IGS = {i: IGS_TYPE_CODE[t] for i, t in enumerate(OUR_TYPES)}

# Map our type → an IGSTGNN description code (rough best-effort heuristic)
# IGSTGNN desc_mapping has 34 categories. We pick a representative for each type.
# Description codes are limited to those present in ALL 3 regions' desc_mapping
# (Alameda has 34, CC has 31, Orange has 33 — pick safe set within min=31)
DESC_CODE_BY_OUR_TYPE = {
    "1141":    5,   # 1179-Trfc Collision-1141 Enrt
    "Fire":   24,   # FIRE-Report of Fire (in all 3)
    "NoInj":   9,   # 1182-Trfc Collision-No Inj
    "UnknInj":10,   # 1183-Trfc Collision-Unkn Inj
    "Hazard":  1,   # 1125-Traffic Hazard
    "AHazard": 2,   # 1125A-Animal Hazard
    "CarFire":18,   # CFIRE-Car Fire
    "Other":   1,   # fallback to "Traffic Hazard" — code 31 missing in CC (only 31 entries)
}


def gaussian_kernel(log_dist: np.ndarray, sigma_log: float, cutoff_log: float) -> np.ndarray:
    """Apply Gaussian kernel to log-distances; zero out beyond cutoff."""
    val = np.exp(-(log_dist ** 2) / (2.0 * sigma_log ** 2))
    val = np.where(log_dist > cutoff_log, 0.0, val)
    val = np.where(log_dist <= 1e-6, 0.0, val)  # zero out self (where dist == 0)
    return val.astype(np.float32)


def build_split(region: str, data_dir: Path, out_dir: Path,
                sigma_euc_log: float, sigma_road_log: float,
                cutoff_euc_log: float, cutoff_road_log: float,
                normalization: dict | None = None,
                overwrite: bool = False,
                *,
                graph_dir: Path | None = None,
                validate_source: bool = True) -> dict:
    """Convert one region's .h5 to IGSTGNN .npy. Returns stats for downstream use."""
    samples_path = data_dir / f"{region}_samples.h5"
    traffic_path = data_dir / f"{region}_traffic.h5"
    rel_path = data_dir / f"{region}_event_rel_feat.npz"
    validated_cache = None
    if validate_source:
        if graph_dir is None:
            raise ValueError(
                "graph_dir is required for strict v3 source validation"
            )
        validated_cache = RegionData(
            region,
            data_dir,
            graph_dir,
            lazy=True,
            require_rel_feat=True,
            include_analysis_labels=False,
            verify_artifact_hashes=True,
        )

    with h5py.File(samples_path, "r") as f:
        protocol = str(f.attrs.get("protocol_version", "legacy"))
        if protocol != PROTOCOL_VERSION:
            raise RuntimeError(
                f"{samples_path} uses {protocol!r}; required {PROTOCOL_VERSION!r}"
            )
        split = f["split"][:]
        sample_start = f["sample_start"][:]
        incident_feat = f["incident_feat"][:]                  # (S, M_max, C_e)
        affected_mask = f["analysis_labels/affected_mask"][:]  # (S, N)
        n_active = f["n_active_incidents"][:]
        T_h = int(f.attrs["T_h"]); T_p = int(f.attrs["T_p"])
        N = int(f.attrs["N"]); M_max = int(f.attrs["M_max"])
        train_end = int(f.attrs["train_end"])
        active_event_idx = f["active_event_idx"][:]            # (S, M_max) int32
        primary_incident_slot = f["primary_incident_slot"][:]
        primary_event_idx = f["event_idx"][:]
    with h5py.File(traffic_path, "r") as f:
        traffic_protocol = str(f.attrs.get("protocol_version", "legacy"))
        if traffic_protocol != PROTOCOL_VERSION:
            raise RuntimeError(
                f"{traffic_path} uses {traffic_protocol!r}; "
                f"required {PROTOCOL_VERSION!r}"
            )
        flow = f["flow_series_imputed"][:]                     # (T, N, 3)
        flow_mask = f["flow_mask"][:]                          # (T, N, 3) bool
        time_enc = f["time_enc"][:]                            # (T, 5)
        times_ns = f["times_ns"][:]                            # (T,) int64 nanoseconds
    import pandas as pd
    ts = pd.to_datetime(times_ns)
    tod_idx = ((ts.hour.to_numpy() * 60 + ts.minute.to_numpy()) // 5).astype(np.float32)  # 0..287
    dow_idx = ts.dayofweek.to_numpy().astype(np.float32)                                    # 0..6
    # IGSTGNN expects x_data[..., 1] = tod / tpd  and  x_data[..., 2] = dow / 7
    tod_norm = tod_idx / 288.0      # (T,)
    dow_norm = dow_idx / 7.0        # (T,)
    with np.load(rel_path, allow_pickle=False) as rel:
        event_rel_feat = rel["rel_feat"].astype(np.float32)

    # Stats are computed from raw observed training timestamps only.
    if normalization is None:
        mean_per_ch = np.empty(3, dtype=np.float32)
        std_per_ch = np.empty(3, dtype=np.float32)
        for channel in range(3):
            valid = flow_mask[:train_end, :, channel].astype(bool)
            values = flow[:train_end, :, channel][valid]
            if values.size == 0:
                raise RuntimeError(
                    f"{region} has no observed training values in channel {channel}"
                )
            mean_per_ch[channel] = values.mean()
            std_per_ch[channel] = max(float(values.std()), 1e-6)
        # IGSTGNN stats_npz stores only flow mean/std (single value each)
        flow_mean = float(mean_per_ch[0])
        flow_std = float(std_per_ch[0])
        normalization = {
            "mean_per_ch": mean_per_ch.astype(np.float32),
            "std_per_ch":  std_per_ch.astype(np.float32),
            "flow_mean": flow_mean,
            "flow_std": flow_std,
        }
        print(f"  [{region}] flow stats: mean={flow_mean:.2f}  std={flow_std:.2f}")

    mean_per_ch = normalization["mean_per_ch"]
    std_per_ch  = normalization["std_per_ch"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_region_dir = out_dir / region
    out_region_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        out_region_dir / "incident_data_stats.npz",
        *[
            out_region_dir / f"incident_data_{name}.npy"
            for name in ("train", "val", "test")
        ],
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite converted v2 data: "
            + ", ".join(str(path) for path in existing)
        )

    # Save stats in IGSTGNN-compatible format
    np.savez(out_region_dir / "incident_data_stats.npz",
             mean=normalization["flow_mean"], std=normalization["flow_std"])

    # Build per-split sample lists
    split_names = {0: "train", 1: "val", 2: "test"}
    per_split_n = {}
    for split_code, split_name in split_names.items():
        idxs = np.where(split == split_code)[0]
        # Keep only samples with at least 1 active incident
        keep = idxs[n_active[idxs] >= 1]
        per_split_n[split_name] = len(keep)
        samples_out = []
        for i in keep:
            anchor = int(sample_start[i])
            hist_start, hist_stop, target_start, target_stop = (
                forecast_window_bounds(anchor, T_h, T_p)
            )
            x_raw = flow[hist_start:hist_stop, :, 0:1]      # (T_h, N, 1)
            y = flow[target_start:target_stop, :, 0:1]      # (T_p, N, 1)
            x_mask = flow_mask[hist_start:hist_stop, :, 0:1]
            y_mask = flow_mask[target_start:target_stop, :, 0:1]

            # z-normalize flow
            x_flow_norm = (x_raw - mean_per_ch[None, None, 0:1]) / std_per_ch[None, None, 0:1]
            y_norm = (y - mean_per_ch[None, None, 0:1]) / std_per_ch[None, None, 0:1]

            # IGSTGNN expects x_data[..., 0] = flow_norm, [..., 1] = tod/tpd, [..., 2] = dow/7
            tod_window = tod_norm[hist_start:hist_stop]  # (T_h,)
            dow_window = dow_norm[hist_start:hist_stop]  # (T_h,)
            tod_chan = np.broadcast_to(tod_window[:, None, None], (T_h, N, 1)).astype(np.float32)
            dow_chan = np.broadcast_to(dow_window[:, None, None], (T_h, N, 1)).astype(np.float32)
            x_norm = np.concatenate([x_flow_norm, tod_chan, dow_chan], axis=-1)  # (T_h, N, 3)

            primary_slot = int(primary_incident_slot[i])
            primary_event = int(active_event_idx[i, primary_slot])
            if primary_event != int(primary_event_idx[i]):
                raise RuntimeError(
                    f"sample {i} primary event mismatch: slot has {primary_event}, "
                    f"sample declares {int(primary_event_idx[i])}"
                )

            # Type: argmax over one-hot dims [0..7]
            type_onehot = incident_feat[i, primary_slot, :8]
            our_type_idx = int(type_onehot.argmax()) if type_onehot.sum() > 0 else 7  # Other
            our_type_name = OUR_TYPES[our_type_idx]
            igs_type_code = IGS_TYPE_CODE[our_type_name]
            desc_code = DESC_CODE_BY_OUR_TYPE[our_type_name]
            elapsed = float(incident_feat[i, primary_slot, 8])  # hours
            elapsed_steps = float(elapsed * 12.0)

            # event_position: IGSTGNN treats this as a 0..11 categorical code (not a sensor index!)
            # Their position_embedding = nn.Embedding(12, 8). We don't have this categorical info,
            # so default to 0. (NOT used as a sensor lookup; only as a learnable embedding key.)
            ev_rel = event_rel_feat[primary_event]   # (N, 4)
            log_euc = ev_rel[:, 0]
            log_road = ev_rel[:, 1]
            up_down = ev_rel[:, 2]
            # Keep an exact zero-distance hit as the closest sensor.  Excluding
            # it would incorrectly zero the second-nearest sensor as well.
            closest_sensor = int(np.argmin(log_euc))
            event_position = 0  # default categorical position code

            # event_distances: gaussian kernel of log distances, 3 dims [euc_kernel, road_kernel, downstream]
            euc_kernel = gaussian_kernel(log_euc, sigma_euc_log, cutoff_euc_log)
            road_kernel = gaussian_kernel(log_road, sigma_road_log, cutoff_road_log)
            downstream = (up_down > 0.5).astype(np.float32)  # +1 → 1, else 0
            event_distances = np.stack([euc_kernel, road_kernel, downstream], axis=-1)
            # Zero out the sensor closest to the event (IGSTGNN convention — they zero event's own row)
            event_distances[closest_sensor] = 0.0

            samples_out.append({
                "x_data": x_norm.astype(np.float32),
                "x_mask": x_mask.astype(np.bool_),
                "y_data": y_norm.astype(np.float32),
                "y_mask": y_mask.astype(np.bool_),
                "event_features": {
                    "Event Time": elapsed_steps,
                    "Description": int(desc_code),
                    "Type": int(igs_type_code),
                    "Holiday": int(time_enc[anchor, 4] > 0.5),
                },
                "event_position": event_position,
                "event_distances": event_distances,
                "durations": elapsed_steps,
                "_t_idx": anchor,
                "_history_start": hist_start,
                "_forecast_origin": anchor,
                "_protocol_version": PROTOCOL_VERSION,
                "_affected_mask": affected_mask[i].astype(np.bool_),   # carried for our eval (NOT used by IGSTGNN)
            })

        out_file = out_region_dir / f"incident_data_{split_name}.npy"
        np.save(out_file, np.array(samples_out, dtype=object), allow_pickle=True)
        print(f"  [{region}] {split_name}: {len(samples_out)} samples → {out_file}")

    print(f"  [{region}] total kept: train={per_split_n['train']} val={per_split_n['val']} test={per_split_n['test']}")
    (out_region_dir / "conversion_meta.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "source_data_dir": str(data_dir),
                "window_semantics": {
                    "history": "[anchor-T_h+1, anchor]",
                    "target": "[anchor+1, anchor+T_p]",
                },
                "train_end": train_end,
                "normalization": {
                    "fit_start": 0,
                    "fit_end": train_end,
                    "mean_per_ch": mean_per_ch.tolist(),
                    "std_per_ch": std_per_ch.tolist(),
                },
                "observation_mask_semantics": {
                    "required": True,
                    "source": "flow_mask",
                    "x_mask": "x_data[..., 0:1]",
                    "y_mask": "y_data",
                    "meaning": "True iff the raw source flow value was observed",
                },
                "samples": per_split_n,
                "source_cache_provenance": (
                    validated_cache.result_metadata()["cache_provenance"]
                    if validated_cache is not None
                    else {"validation": "disabled_for_synthetic_fixture"}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return normalization


REGION_TO_GRAPH_KEY = {"Alameda": "alameda", "ContraCosta": "contra_costa", "Orange": "orange"}


def build_adj_matrix(region: str, graph_dir: Path, out_dir: Path):
    """Build (N, N) binary adj_matrix from our sparse edge_index."""
    g = np.load(graph_dir / f"{REGION_TO_GRAPH_KEY[region]}_sparse_adj.npz")
    edge_index = g["edge_index"]
    N = int(g["N"])
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    out_path = out_dir / region / "adj_matrix.npy"
    np.save(out_path, A)
    print(f"  [{region}] adj_matrix: ({N},{N}) nonzero={(A > 0).sum()}")


def build_sensor_info(region: str, data_dir: Path, out_dir: Path):
    """Build sensor_info.npz from our static_meta if IGSTGNN expects it."""
    with h5py.File(data_dir / f"{region}_traffic.h5", "r") as f:
        static_meta = f["static_meta"][:]   # (N, C_meta)
    # IGSTGNN's sensor_info keys (from their loader): sensor_type, surface, roadway_use, road_width, speed_limit
    # We don't have exact mapping. Use placeholders.
    N = static_meta.shape[0]
    out_path = out_dir / region / "sensor_info.npz"
    np.savez(out_path,
             sensor_type=np.ones(N, dtype=np.int64),     # all mainline
             surface=np.ones(N, dtype=np.int64),
             roadway_use=np.full(N, 5, dtype=np.int64),  # "No Special Features"
             road_width=np.full(N, 30.0, dtype=np.float32),
             speed_limit=np.full(N, 65.0, dtype=np.float32))
    print(f"  [{region}] saved placeholder sensor_info.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="all", help="region or 'all'")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_REGION_DATA_DIR,
    )
    p.add_argument("--graph_dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/igstgnn_data_from_ours_v3_noleak_dedup"),
    )
    p.add_argument("--sigma_euc_log", type=float, default=1.5,
                   help="sigma for gaussian kernel on log-euclidean km")
    p.add_argument("--sigma_road_log", type=float, default=1.5,
                   help="sigma for gaussian kernel on log-road-network km")
    p.add_argument("--cutoff_euc_log", type=float, default=4.0,
                   help="zero out log_euclidean > this (≈54 km euclid)")
    p.add_argument("--cutoff_road_log", type=float, default=5.0,
                   help="zero out log_road > this (≈148 km road)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    regions = ["Alameda", "ContraCosta", "Orange"] if args.region == "all" else [args.region]

    for region in regions:
        print(f"\n=== {region} ===")
        build_split(region, args.data_dir, args.out_dir,
                    args.sigma_euc_log, args.sigma_road_log,
                    args.cutoff_euc_log, args.cutoff_road_log,
                    overwrite=args.overwrite,
                    graph_dir=args.graph_dir)
        build_sensor_info(region, args.data_dir, args.out_dir)
        build_adj_matrix(region, args.graph_dir, args.out_dir)


if __name__ == "__main__":
    main()
