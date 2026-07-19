#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=ndr_v2_region
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/student/t330026045/traffic_weather/ndr_logs/%x_%j.out
#SBATCH --error=/home/student/t330026045/traffic_weather/ndr_logs/%x_%j.err

# One formal region per Slurm job; seeds 42, 1, and 2 run sequentially on the
# allocated GPU.  Submit through submit_ndr_event_causal_v2.sh so that Orange
# receives an A100 and every region depends on the all-region cache audit.
set -euo pipefail
trap 'echo "FAILED at line ${LINENO}, time $(date)" >&2' ERR
umask 027

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
PYTHON=$BASE/dcrnn_env/bin/python
DATA_DIR=$BASE/outputs/dist_net/region_data_causal_v2
GRAPH_DIR=$BASE/outputs/region_graphs
OUT_ROOT=$BASE/outputs/ndrnet_2023_event_causal_v2

: "${REGION:?REGION must be Alameda, ContraCosta, or Orange}"
case "$REGION" in
  Alameda) GRAPH_FILE=$GRAPH_DIR/alameda_sparse_adj.npz ;;
  ContraCosta) GRAPH_FILE=$GRAPH_DIR/contra_costa_sparse_adj.npz ;;
  Orange) GRAPH_FILE=$GRAPH_DIR/orange_sparse_adj.npz ;;
  *) echo "ERROR: unsupported REGION=$REGION" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
REQUIRED_FILES=(
  "$REPO/scripts/train_staeformer_deseason.py"
  "$REPO/scripts/summarize_ndr_horizon_metrics.py"
  "$REPO/scripts/build_full_county_cache.py"
  "$REPO/baselines/STAEformer/STAEformer.py"
  "$REPO/dist_net/data.py"
  "$REPO/fourier_dual_net/rgdn.py"
  "$REPO/fourier_dual_net/deseason.py"
  "$DATA_DIR/${REGION}_traffic.h5"
  "$DATA_DIR/${REGION}_samples.h5"
  "$GRAPH_FILE"
)
for FILE in "${REQUIRED_FILES[@]}"; do
  [[ -s "$FILE" ]] || { echo "ERROR: missing or empty file: $FILE" >&2; exit 1; }
done

# Formal jobs never skip or reuse a seed.  Fail before doing any training if
# the region output already contains artifacts from another attempt.
REGION_OUT=$OUT_ROOT/$REGION
if [[ -d "$REGION_OUT" ]] && \
   [[ -n "$(find "$REGION_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: refusing to reuse non-empty formal output: $REGION_OUT" >&2
  echo "Choose a new versioned OUT_ROOT; do not silently skip old seeds." >&2
  exit 1
fi
mkdir -p "$OUT_ROOT"

cd "$REPO"

echo "============================================================"
echo "NDR-Net formal event causal-v2 training"
echo "START: $(date)"
echo "HOST: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "REGION: $REGION"
echo "SEEDS: 42 1 2 (sequential)"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_ROOT: $OUT_ROOT"
echo "PARAMETERS: epochs=30 batch=16 lr=0.001 wd=0.0001 gate=1.0 spec=0.3"
echo "============================================================"

echo "=== Source SHA256 ==="
sha256sum \
  scripts/train_staeformer_deseason.py \
  scripts/summarize_ndr_horizon_metrics.py \
  scripts/build_full_county_cache.py \
  baselines/STAEformer/STAEformer.py \
  dist_net/data.py \
  fourier_dual_net/rgdn.py \
  fourier_dual_net/deseason.py

# Re-run the strict region-level cache contract in the GPU job.  This protects
# direct/accidental submission even though the formal wrapper also adds an
# afterok dependency on the all-region audit job.
"$PYTHON" - "$REGION" "$DATA_DIR" "$GRAPH_DIR" <<'PY'
import sys

from dist_net.data import RegionData

region, data_dir, graph_dir = sys.argv[1:]
data = RegionData(
    region,
    data_dir,
    graph_dir,
    lazy=True,
    strict_causal_cache=True,
    expected_baseline_train_frac=0.70,
    expected_t_h=12,
    expected_t_p=12,
    expected_split_train_end=0.70,
    expected_split_val_end=0.85,
)
print(
    f"CACHE_OK: {region} causal_forward_fill "
    f"N={data.N} T={data.T} cache_id={data.cache_provenance['traffic_cache_id']}"
)
PY

"$PYTHON" - "$REGION" <<'PY'
import sys
import torch

region = sys.argv[1]
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job")
name = torch.cuda.get_device_name(0)
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {name}; memory={total_gib:.2f} GiB")
if region == "Orange" and "A100" not in name.upper():
    raise RuntimeError(
        f"Orange formal training requires an A100, but Slurm provided {name!r}"
    )
PY

nvidia-smi

for SEED in 42 1 2; do
  echo "============================================================"
  echo "$REGION seed $SEED START: $(date)"
  echo "============================================================"

  "$PYTHON" scripts/train_staeformer_deseason.py \
    --region "$REGION" \
    --variant s0g \
    --data_dir "$DATA_DIR" \
    --graph_dir "$GRAPH_DIR" \
    --out_dir "$OUT_ROOT" \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --seed "$SEED" \
    --num_workers 0 \
    --lambda_gate 1.0 \
    --lambda_spec 0.3 \
    --device cuda \
    --strict-causal-cache

  SUMMARY=$OUT_ROOT/$REGION/s0g_seed${SEED}/summary.json
  [[ -s "$SUMMARY" ]] || {
    echo "ERROR: training did not produce $SUMMARY" >&2
    exit 1
  }
  echo "$REGION seed $SEED COMPLETE: $(date)"
done

# No legacy override and no hash-verification bypass: this requires all 12
# horizons, finite/count-complete metrics, identical provenance across seeds,
# an unchanged causal cache/graph, and exact agreement with cache masks.
"$PYTHON" scripts/summarize_ndr_horizon_metrics.py \
  --root "$OUT_ROOT" \
  --region "$REGION" \
  --seeds 42 1 2 \
  --method "NDR-Net" \
  --output-dir "$OUT_ROOT/$REGION"

echo "=== Final table row ==="
cat "$OUT_ROOT/$REGION/ndrnet_horizon_table.csv"
echo "NDR_V2_DONE: $REGION"
echo "END: $(date)"
