#!/usr/bin/env bash
# Recover the known 24 GB GPU failures, resume the canonical host queues, and finalize.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
HOST_ID=${HOST_ID:?HOST_ID is required}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=${MANIFEST_SHA256:-d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78}
SPLITS_ROOT=${SPLITS_ROOT:-/root/autodl-tmp/table4/data/official_splits}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/results}
CONTROL_DIR=${CONTROL_DIR:-/root/autodl-tmp/table4/control/$HOST_ID}
HOST_RUNNER=${HOST_RUNNER:-$REPO/scripts/remote/linux/igstgnn_table4_run_host.sh}
LOG=${LOG:-$CONTROL_DIR/recovery_v2.log}

mkdir -p "$CONTROL_DIR"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "RECOVERY_V2_BLOCKED $*"
  exit 1
}

run_recovery_task() {
  local task_id=$1
  local shard_id=$2
  local adapter=$3
  local visible_devices=$4
  local microbatches=${5:-1}
  local result_root="$RESULTS_BASE/shards/$shard_id"
  local task_dir="$result_root/tasks/$task_id"

  [[ -d "$result_root" && ! -L "$result_root" ]] \
    || fail "result root is missing or unsafe for $task_id"
  if [[ -d "$task_dir" && ! -L "$task_dir" ]]; then
    log "RECOVERY_TASK_ALREADY_PRESENT task=$task_id shard=$shard_id"
    return
  fi
  [[ ! -e "$task_dir" && ! -L "$task_dir" ]] \
    || fail "task target is unsafe for $task_id"

  log "RECOVERY_TASK_START task=$task_id adapter=$adapter devices=$visible_devices microbatches=$microbatches"
  set +e
  CUDA_VISIBLE_DEVICES="$visible_devices" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONPATH="$REPO" \
  "$PYTHON" -X utf8 -m \
    scripts.reproduction.igstgnn_table4.recovery_task_runner \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --task-id "$task_id" \
    --splits-root "$SPLITS_ROOT" \
    --results-root "$result_root" \
    --device cuda:0 \
    --adapter "$adapter" \
    --microbatches "$microbatches" >> "$LOG" 2>&1
  local task_rc=$?
  set -e
  [[ "$task_rc" == "0" ]] \
    || fail "task=$task_id adapter=$adapter exited rc=$task_rc"
  [[ -d "$task_dir" && ! -L "$task_dir" ]] \
    || fail "task=$task_id completed without a formal task directory"
  log "RECOVERY_TASK_OK task=$task_id adapter=$adapter"
}

[[ -d "$REPO" && ! -L "$REPO" ]] || fail "repository checkout is missing"
[[ -x "$PYTHON" ]] || fail "Python runtime is missing"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail "manifest is missing"
[[ -d "$SPLITS_ROOT" && ! -L "$SPLITS_ROOT" ]] || fail "split root is missing"
[[ -x "$HOST_RUNNER" && ! -L "$HOST_RUNNER" ]] || fail "host runner is missing"
if pgrep -af \
  'igstgnn_table4\.(distributed_host_runner|matrix_runner|task_runner|recovery_task_runner)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "another Table 4 training process is active"
fi

log "RECOVERY_V2_START host=$HOST_ID"
case "$HOST_ID" in
  table4_4090d4)
    run_recovery_task \
      005_alameda_stgcn table4_4090d4_gpu0 \
      stgcn-idempotent-self-loop-removal 0
    run_recovery_task \
      018_contra_costa_stgcn table4_4090d4_gpu0 \
      stgcn-idempotent-self-loop-removal 0
    run_recovery_task \
      011_alameda_d2stgnn table4_4090d4_gpu2 \
      multi-gpu-full-batch 0,1,2,3
    run_recovery_task \
      024_contra_costa_d2stgnn table4_4090d4_gpu2 \
      multi-gpu-full-batch 0,1,2,3
    run_recovery_task \
      013_alameda_igstgnn table4_4090d4_gpu3 \
      multi-gpu-full-batch 0,1,2,3
    run_recovery_task \
      026_contra_costa_igstgnn table4_4090d4_gpu3 \
      multi-gpu-full-batch 0,1,2,3
    ;;
  table4_4090d5)
    run_recovery_task \
      031_orange_stgcn table4_4090d5_gpu0 \
      stgcn-idempotent-self-loop-removal 0
    run_recovery_task \
      036_orange_dgcrn table4_4090d5_gpu1 \
      multi-gpu-full-batch 0,1,2,3,4
    run_recovery_task \
      037_orange_d2stgnn table4_4090d5_gpu2 \
      multi-gpu-full-batch 0,1,2,3,4 2
    run_recovery_task \
      039_orange_igstgnn table4_4090d5_gpu4 \
      multi-gpu-full-batch 0,1,2,3,4
    ;;
  *)
    fail "unsupported host ID: $HOST_ID"
    ;;
esac

log "CANONICAL_HOST_RESUME_START host=$HOST_ID"
HOST_ID="$HOST_ID" \
REPO="$REPO" \
PYTHON="$PYTHON" \
MANIFEST="$MANIFEST" \
MANIFEST_SHA256="$MANIFEST_SHA256" \
SPLITS_ROOT="$SPLITS_ROOT" \
RESULTS_BASE="$RESULTS_BASE" \
CONTROL_DIR="$CONTROL_DIR" \
bash "$HOST_RUNNER" >> "$LOG" 2>&1
