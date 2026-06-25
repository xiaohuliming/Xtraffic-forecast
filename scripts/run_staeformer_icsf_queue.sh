#!/usr/bin/env bash
# Decisive experiment 2: STAEformer +/- ICSF on event-anchored protocol.
# A/B per region (base then icsf) so the label-gain delta lands per region.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
PATIENCE="${PATIENCE:-6}"
EPOCHS="${EPOCHS:-30}"
COMMON="--seed $SEED --patience $PATIENCE --epochs $EPOCHS --device cuda:0"
LOG=outputs/staeformer_icsf_run.log
mkdir -p outputs
echo "=== STAEformer +/- ICSF queue  seed=$SEED patience=$PATIENCE epochs=$EPOCHS ===" | tee -a "$LOG"

run() { echo ">>> $*" | tee -a "$LOG"; python "$@" 2>&1 | tee -a "$LOG"; }

for R in Alameda ContraCosta Orange; do
  run scripts/train_staeformer_icsf.py --region "$R" $COMMON
  run scripts/train_staeformer_icsf.py --region "$R" --use_icsf $COMMON
done

echo "=== done. A/B summaries: ===" | tee -a "$LOG"
for R in Alameda ContraCosta Orange; do
  for T in base icsf; do
    f="outputs/baselines/staeformer_icsf/$R/$T/summary.json"
    [ -f "$f" ] && echo "$R $T: $(cat "$f")" | tee -a "$LOG"
  done
done
