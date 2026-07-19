#!/bin/bash
#SBATCH -p cpu
#SBATCH --job-name=ndr_v2_cache_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/student/t330026045/traffic_weather/ndr_logs/%x_%j.out
#SBATCH --error=/home/student/t330026045/traffic_weather/ndr_logs/%x_%j.err

# Formal, read-only preflight for the v2 causal event cache.  The three GPU
# jobs are submitted with an afterok dependency on this job.
set -euo pipefail
trap 'echo "FAILED at line ${LINENO}, time $(date)" >&2' ERR
umask 027

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
PYTHON=$BASE/dcrnn_env/bin/python
DATA_DIR=$BASE/outputs/dist_net/region_data_causal_v2
GRAPH_DIR=$BASE/outputs/region_graphs

[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repository is missing: $REPO" >&2; exit 1; }

cd "$REPO"

echo "============================================================"
echo "NDR-Net formal causal-v2 cache audit"
echo "START: $(date)"
echo "HOST: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "DATA_DIR: $DATA_DIR"
echo "GRAPH_DIR: $GRAPH_DIR"
echo "============================================================"

"$PYTHON" - "$DATA_DIR" "$GRAPH_DIR" <<'PY'
from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import h5py
import numpy as np

from dist_net.data import REGION_NAME_TO_KEY, RegionData


DATA_DIR = Path(sys.argv[1]).resolve()
GRAPH_DIR = Path(sys.argv[2]).resolve()
REGIONS = ("Alameda", "ContraCosta", "Orange")
EXPECTED_N = {"Alameda": 521, "ContraCosta": 496, "Orange": 990}
BUILD_SCRIPT = Path("scripts/build_full_county_cache.py").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def text_attr(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"causal-v2 data directory is missing: {DATA_DIR}")
if not GRAPH_DIR.is_dir():
    raise FileNotFoundError(f"graph directory is missing: {GRAPH_DIR}")
if not BUILD_SCRIPT.is_file():
    raise FileNotFoundError(f"cache builder source is missing: {BUILD_SCRIPT}")
current_builder_sha256 = sha256(BUILD_SCRIPT)

global_build_ids: set[str] = set()
for region in REGIONS:
    traffic_path = DATA_DIR / f"{region}_traffic.h5"
    samples_path = DATA_DIR / f"{region}_samples.h5"
    graph_path = GRAPH_DIR / f"{REGION_NAME_TO_KEY[region]}_sparse_adj.npz"
    for path in (traffic_path, samples_path, graph_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty formal input: {path}")

    # This performs the authoritative schema, causal-imputation, 0.70 baseline,
    # cache-pair identity, shape, unique-sample, split-window, and graph checks.
    data = RegionData(
        region,
        DATA_DIR,
        GRAPH_DIR,
        lazy=True,
        strict_causal_cache=True,
        expected_baseline_train_frac=0.70,
        expected_t_h=12,
        expected_t_p=12,
        expected_split_train_end=0.70,
        expected_split_val_end=0.85,
    )
    if data.N != EXPECTED_N[region]:
        raise RuntimeError(
            f"{region}: expected {EXPECTED_N[region]} nodes, found {data.N}"
        )

    with h5py.File(traffic_path, "r") as traffic, h5py.File(samples_path, "r") as samples:
        traffic_build_id = text_attr(traffic.attrs["build_id"])
        samples_build_id = text_attr(samples.attrs["build_id"])
        if samples_build_id != traffic_build_id:
            raise RuntimeError(
                f"{region}: samples and traffic were not produced by the same full "
                f"build ({samples_build_id} != {traffic_build_id}); do not use "
                "--samples-only for this formal cache"
            )
        recorded_builder_hashes = {
            text_attr(traffic.attrs["builder_source_sha256"]),
            text_attr(samples.attrs["traffic_builder_source_sha256"]),
            text_attr(samples.attrs["samples_builder_source_sha256"]),
        }
        if recorded_builder_hashes != {current_builder_sha256}:
            raise RuntimeError(
                f"{region}: cache builder source mismatch; recorded="
                f"{sorted(recorded_builder_hashes)}, current={current_builder_sha256}"
            )
        global_build_ids.add(traffic_build_id)

        split = samples["split"][...]
        counts = np.bincount(split.astype(np.int64), minlength=3)
        if counts.shape != (3,) or np.any(counts <= 0):
            raise RuntimeError(f"{region}: empty or invalid split counts {counts.tolist()}")

        # Bounded-memory numerical audit.  Valid physical traffic and all
        # imputed values must be finite and non-negative.
        flow = traffic["flow_series_imputed"]
        mask = traffic["flow_mask"]
        for lo in range(0, flow.shape[0], 288):
            values = flow[lo:lo + 288]
            observed = mask[lo:lo + 288].astype(bool)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{region}: imputed traffic contains NaN/Inf")
            if np.any(values < 0):
                raise RuntimeError(f"{region}: imputed traffic contains negative values")
            if not np.isfinite(values[observed]).all():
                raise RuntimeError(f"{region}: an observed traffic cell is non-finite")

        for name in ("baseline_median", "baseline_scale", "time_enc", "static_meta"):
            if not np.isfinite(traffic[name][...]).all():
                raise RuntimeError(f"{region}: {name} contains NaN/Inf")
        if np.any(traffic["baseline_scale"][...] < 0):
            raise RuntimeError(f"{region}: baseline_scale contains negative values")
        if not np.isfinite(samples["incident_feat"][...]).all():
            raise RuntimeError(f"{region}: incident_feat contains NaN/Inf")

        print(
            f"CACHE_OK: {region} N={data.N} T={data.T} "
            f"train/val/test={counts.tolist()} "
            f"duplicates_dropped={int(samples.attrs['n_duplicate_sample_start_dropped'])} "
            f"build_id={traffic_build_id}"
        )

if len(global_build_ids) != 1:
    raise RuntimeError(
        "the three regions were not produced by one complete cache build: "
        f"{sorted(global_build_ids)}"
    )

print(f"CAUSAL_V2_CACHE_AUDIT_COMPLETE: build_id={next(iter(global_build_ids))}")
PY

echo "END: $(date)"
