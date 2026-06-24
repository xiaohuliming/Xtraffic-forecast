#!/usr/bin/env bash
# Full-window protocol queue: 3 models x 3 regions, IDENTICAL windows.
# All runs share --protocol full_window --stride $STRIDE --train_frac/--val_frac,
# so sample_start/split are element-wise identical across models (deterministic
# loader + fixed seed => reproducible, windows aligned by construction).
#
# Usage:  STRIDE=1 SEED=42 bash scripts/run_fullwindow_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STRIDE="${STRIDE:-1}"
SEED="${SEED:-42}"
TF="${TRAIN_FRAC:-0.7}"
VF="${VAL_FRAC:-0.1}"
PATIENCE="${PATIENCE:-6}"
COMMON="--protocol full_window --stride $STRIDE --train_frac $TF --val_frac $VF --seed $SEED --patience $PATIENCE"
LOG=outputs/fullwindow_run.log
mkdir -p outputs
echo "=== full-window queue  stride=$STRIDE seed=$SEED frac=$TF/$VF ===" | tee -a "$LOG"

run() { echo ">>> $*" | tee -a "$LOG"; python "$@" 2>&1 | tee -a "$LOG"; }

# Region-first ordering: a full 3-model comparison lands per region (lightest N first).
for R in Alameda ContraCosta Orange; do
  run scripts/train_graphwavenet.py     --region "$R" $COMMON
  run scripts/train_fourier_dual_net.py --region "$R" --decomp_mode learnable --K 3 $COMMON
  run scripts/train_staeformer_xtraffic.py --region "$R" $COMMON
done

echo "=== done. summaries: ===" | tee -a "$LOG"
for R in Alameda ContraCosta Orange; do
  for D in graphwavenet_fullwindow fourier_dual_net/learnable_K3_fullwindow staeformer_fullwindow; do
    f="outputs/baselines/$D/$R/summary.json"; [ -f "$f" ] || f="outputs/$D/$R/summary.json"
    [ -f "$f" ] && echo "$R $D: $(cat "$f")" | tee -a "$LOG"
  done
done
