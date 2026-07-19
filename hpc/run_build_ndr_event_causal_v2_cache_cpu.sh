#!/bin/bash
#SBATCH -p cpu
#SBATCH --job-name=build_ndr_causal_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/student/t330026045/traffic_weather/cache_logs/%x_%j.out
#SBATCH --error=/home/student/t330026045/traffic_weather/cache_logs/%x_%j.err

# Full rebuild only.  This deliberately refuses a non-empty destination and
# never uses --samples-only, so traffic and samples share one formal build_id.
set -euo pipefail
trap 'echo "FAILED at line ${LINENO}, time $(date)" >&2' ERR
umask 027

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
PYTHON=$BASE/dcrnn_env/bin/python
OUT_DIR=$BASE/outputs/dist_net/region_data_causal_v2
EVENT_ROOT=$BASE/outputs/impact_labels_aggregated/region_area_sensor_window
NODE_LABEL_DIR=$BASE/outputs/impact_labels

[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
[[ -s "$REPO/scripts/build_full_county_cache.py" ]] || {
  echo "ERROR: cache builder is missing" >&2
  exit 1
}
for REGION in Alameda ContraCosta Orange; do
  [[ -s "$EVENT_ROOT/$REGION/event_labels.csv" ]] || {
    echo "ERROR: missing event labels for $REGION" >&2
    exit 1
  }
  [[ -s "$NODE_LABEL_DIR/$REGION/node_labels.csv" ]] || {
    echo "ERROR: missing node labels for $REGION" >&2
    exit 1
  }
done

if [[ -d "$OUT_DIR" ]] && \
   [[ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: refusing to overwrite non-empty cache directory: $OUT_DIR" >&2
  echo "Move it aside or choose a new versioned directory." >&2
  exit 1
fi

mkdir -p "$BASE/cache_logs" "$OUT_DIR"
cd "$REPO"

echo "START: $(date)"
echo "Full formal cache build: Alameda ContraCosta Orange"
echo "Destination: $OUT_DIR"
sha256sum scripts/build_full_county_cache.py

"$PYTHON" scripts/build_full_county_cache.py \
  --archive "$BASE" \
  --event-root "$EVENT_ROOT" \
  --node-label-dir "$NODE_LABEL_DIR" \
  --out-dir "$OUT_DIR" \
  --regions Alameda ContraCosta Orange \
  --t-h 12 \
  --t-p 12 \
  --m-max 32 \
  --active-window 24 \
  --sample-offsets 0 6 12 \
  --candidate-pm-radius 5.0 \
  --anchor-pm-radius 2.0 \
  --baseline-mask-extra-steps 12 \
  --baseline-train-frac 0.70 \
  --min-baseline-count 8 \
  --split-train-end 0.70 \
  --split-val-end 0.85

echo "CAUSAL_V2_CACHE_BUILD_COMPLETE"
echo "END: $(date)"
