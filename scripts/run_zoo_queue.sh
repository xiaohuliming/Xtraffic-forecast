#!/usr/bin/env bash
# Baseline-zoo worker pool. Runs on the GPU box from the repo root's parent.
#   RUNS_FILE=zoo_tasks.txt WORKERS=6 bash scripts/run_zoo_queue.sh
# Task file lines: <model> <region> <seed> [batch_size] [attempt]
# Seed 42 writes outputs/baselines/zoo_<model>/, other seeds zoo_<model>_seed<seed>/.
# A failed run (no summary.json) is re-queued once at half batch (max 2 attempts).
set -u
cd "$(dirname "$0")/.."
RUNS_FILE=${RUNS_FILE:-zoo_tasks.txt}
WORKERS=${WORKERS:-6}
LOCK=/tmp/zoo_queue.lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
touch "$LOCK"

worker() {
  local wid=$1
  while true; do
    local line
    line=$(flock "$LOCK" bash -c "head -n1 '$RUNS_FILE' 2>/dev/null; sed -i 1d '$RUNS_FILE' 2>/dev/null")
    [ -z "$line" ] && break
    set -- $line
    local model=$1 region=$2 seed=$3 bs=${4:-32} att=${5:-0}
    local out="outputs/baselines/zoo_${model}"
    [ "$seed" != "42" ] && out="${out}_seed${seed}"
    echo "[w$wid] START $model $region s$seed bs$bs att$att $(date +%F_%T)"
    python3 scripts/train_baseline_zoo.py --model "$model" --region "$region" \
      --seed "$seed" --batch-size "$bs" --out-dir "$out" \
      > "logs/zoo_${model}_${region}_s${seed}.log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ] && [ ! -s "$out/$region/summary.json" ] && [ "$att" -lt 2 ]; then
      local nbs=$(( bs > 8 ? bs / 2 : bs ))
      flock "$LOCK" bash -c "echo '$model $region $seed $nbs $((att + 1))' >> '$RUNS_FILE'"
      echo "[w$wid] RETRY-QUEUED $model $region s$seed bs$nbs $(date +%F_%T)"
    fi
    echo "[w$wid] END $model $region s$seed rc=$rc $(date +%F_%T)"
  done
}

for i in $(seq 1 "$WORKERS"); do worker "$i" & done
wait
echo ALL_DONE
