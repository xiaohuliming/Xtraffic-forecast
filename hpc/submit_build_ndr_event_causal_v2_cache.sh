#!/bin/bash
# Run this on the HPC login node.  It validates the direct-under-BASE archive
# layout and creates Slurm's log directory before submitting the build job.
set -euo pipefail
umask 027

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
BUILD_SCRIPT=$REPO/hpc/run_build_ndr_event_causal_v2_cache_cpu.sh

mkdir -p "$BASE/cache_logs"

REQUIRED_RAW=(
  "$BASE/sensor_meta_feature.csv"
  "$BASE/node_order.npy"
  "$BASE/incidents_y2023.csv"
)
for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
  REQUIRED_RAW+=("$BASE/p${MONTH}_done.npy")
done

for FILE in "${REQUIRED_RAW[@]}"; do
  [[ -s "$FILE" ]] || {
    echo "ERROR: expected raw-data file is missing or empty: $FILE" >&2
    echo "The formal script expects raw data directly under $BASE." >&2
    exit 1
  }
done

[[ -s "$BUILD_SCRIPT" ]] || {
  echo "ERROR: build job script is missing: $BUILD_SCRIPT" >&2
  exit 1
}
bash -n "$BUILD_SCRIPT"

JOB_ID=$(sbatch --parsable "$BUILD_SCRIPT")
JOB_ID=${JOB_ID%%;*}

echo "Formal causal-v2 cache build submitted: $JOB_ID"
echo "STDOUT: $BASE/cache_logs/build_ndr_causal_v2_${JOB_ID}.out"
echo "STDERR: $BASE/cache_logs/build_ndr_causal_v2_${JOB_ID}.err"
squeue -j "$JOB_ID"
