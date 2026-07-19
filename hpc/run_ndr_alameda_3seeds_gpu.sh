#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=ndr_ala_3seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/student/t330026045/traffic_weather/ndr_logs/ndr_alameda_3seeds_%j.out
#SBATCH --error=/home/student/t330026045/traffic_weather/ndr_logs/ndr_alameda_3seeds_%j.err

set -euo pipefail
trap 'echo "FAILED at line ${LINENO}, time $(date)" >&2' ERR

# Historical preliminary script: it points at the pre-v2 cache and trains only
# Alameda.  Require an explicit override so it cannot be mistaken for the
# formal three-region causal-v2 workflow.
if [[ "${ALLOW_LEGACY_NDR:-0}" != "1" ]]; then
  echo "ERROR: this legacy Alameda-only script is disabled for formal results." >&2
  echo "Use hpc/submit_ndr_event_causal_v2.sh instead." >&2
  exit 2
fi

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
PYTHON=$BASE/traffic_env/bin/python
DATA_DIR=$BASE/outputs/dist_net/region_data
GRAPH_DIR=$BASE/outputs/region_graphs
OUT_DIR=$BASE/outputs/baselines/staeformer_deseason
REGION=Alameda

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

cd "$REPO"
mkdir -p "$BASE/ndr_logs" "$OUT_DIR"

echo "============================================================"
echo "NDR-Net formal training with horizon metrics"
echo "START: $(date)"
echo "HOST: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "REGION: $REGION"
echo "SEEDS: 42 1 2"
echo "============================================================"

REQUIRED_FILES=(
  "$REPO/scripts/train_staeformer_deseason.py"
  "$REPO/scripts/summarize_ndr_horizon_metrics.py"
  "$REPO/baselines/STAEformer/STAEformer.py"
  "$REPO/dist_net/data.py"
  "$REPO/fourier_dual_net/rgdn.py"
  "$REPO/fourier_dual_net/deseason.py"
  "$DATA_DIR/Alameda_traffic.h5"
  "$DATA_DIR/Alameda_samples.h5"
  "$GRAPH_DIR/alameda_sparse_adj.npz"
)
for FILE in "${REQUIRED_FILES[@]}"; do
  [[ -s "$FILE" ]] || { echo "ERROR: missing or empty file: $FILE" >&2; exit 1; }
  echo "FOUND: $FILE"
done

echo "=== Source SHA256 ==="
sha256sum \
  scripts/train_staeformer_deseason.py \
  scripts/summarize_ndr_horizon_metrics.py \
  baselines/STAEformer/STAEformer.py \
  dist_net/data.py \
  fourier_dual_net/rgdn.py \
  fourier_dual_net/deseason.py

"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job")
props = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(props.total_memory / 1024**3, 2))
PY
nvidia-smi

summary_is_complete() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import math
import sys

path, expected_seed = sys.argv[1], int(sys.argv[2])
try:
    result = json.load(open(path, encoding="utf-8"))
    assert result["variant"] == "s0g"
    assert int(result["seed"]) == expected_seed
    assert int(result["params"]) == 1_582_308
    selected = result["selected_horizons"]
    for horizon in ("3", "6", "12"):
        for metric in ("mae", "rmse", "mape_pct"):
            assert math.isfinite(float(selected[horizon][metric]))
except Exception:
    raise SystemExit(1)
PY
}

for SEED in 42 1 2; do
  SUMMARY="$OUT_DIR/$REGION/s0g_seed${SEED}/summary.json"
  echo "============================================================"
  echo "SEED $SEED START: $(date)"
  echo "============================================================"

  if [[ -s "$SUMMARY" ]] && summary_is_complete "$SUMMARY" "$SEED"; then
    echo "SKIP seed $SEED: compatible completed result already exists"
    continue
  fi

  "$PYTHON" scripts/train_staeformer_deseason.py \
    --region "$REGION" \
    --variant s0g \
    --data_dir "$DATA_DIR" \
    --graph_dir "$GRAPH_DIR" \
    --out_dir "$OUT_DIR" \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --seed "$SEED" \
    --num_workers 0 \
    --lambda_gate 1.0 \
    --lambda_spec 0.3 \
    --device cuda

  summary_is_complete "$SUMMARY" "$SEED" || {
    echo "ERROR: seed $SEED did not produce complete horizon metrics" >&2
    exit 1
  }
  echo "SEED $SEED COMPLETE: $(date)"
done

"$PYTHON" scripts/summarize_ndr_horizon_metrics.py \
  --root "$OUT_DIR" \
  --region "$REGION" \
  --seeds 42 1 2 \
  --method "NDR-Net" \
  --output-dir "$OUT_DIR/$REGION"

echo "=== Final table row ==="
cat "$OUT_DIR/$REGION/ndrnet_horizon_table.csv"
echo "NDR_DONE"
echo "END: $(date)"
