#!/usr/bin/env bash
# Recover Orange IGSTGNN on two GPUs, then finalize the complete source host.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
DP_REPO=${DP_REPO:-/root/autodl-tmp/table4/repo_dp}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
HOST_ID=${HOST_ID:-table4_4090d5}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=${MANIFEST_SHA256:-d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78}
SPLITS_ROOT=${SPLITS_ROOT:-/root/autodl-tmp/table4/data/official_splits}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/results}
TASK_ID=${TASK_ID:-039_orange_igstgnn}
DP_VISIBLE_DEVICES=${DP_VISIBLE_DEVICES:-4,3}
RESOURCE_QUEUE=${RESOURCE_QUEUE:-$RESULTS_BASE/shards/table4_4090d5_gpu3/queue_status.json}
TASK_RESULT_ROOT=${TASK_RESULT_ROOT:-$RESULTS_BASE/shards/table4_4090d5_gpu4}
WAIT_QUEUE_GLOB=${WAIT_QUEUE_GLOB:-$RESULTS_BASE/shards/table4_4090d5_gpu}
POLL_SECONDS=${POLL_SECONDS:-60}
FINALIZER=${FINALIZER:-$REPO/scripts/remote/linux/igstgnn_table4_finalize_host.sh}
LOG=${LOG:-/root/autodl-tmp/table4/control/$HOST_ID/task039_dp_recovery.log}

mkdir -p "$(dirname "$LOG")"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "RECOVERY_BLOCKED $*"
  exit 1
}

queue_state() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state = value.get("state")
if not isinstance(state, str):
    raise SystemExit("queue state is invalid")
print(state)
PY
}

wait_for_completed_queue() {
  local queue=$1
  local label=$2
  local state
  while true; do
    [[ -s "$queue" && ! -L "$queue" ]] || fail "$label queue is missing"
    state=$(queue_state "$queue") || fail "$label queue cannot be parsed"
    case "$state" in
      completed)
        log "QUEUE_COMPLETED label=$label"
        return
        ;;
      running)
        log "WAIT_QUEUE label=$label state=$state"
        sleep "$POLL_SECONDS"
        ;;
      *)
        fail "$label queue reached state=$state"
        ;;
    esac
  done
}

[[ "$DP_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] \
  || fail "DP_VISIBLE_DEVICES must name exactly two GPUs"
[[ -d "$REPO" && -d "$DP_REPO" ]] || fail "repository checkout is missing"
[[ -x "$PYTHON" ]] || fail "Python runtime is missing"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail "manifest is missing"
[[ -d "$SPLITS_ROOT" && ! -L "$SPLITS_ROOT" ]] || fail "split root is missing"
[[ -d "$TASK_RESULT_ROOT" && ! -L "$TASK_RESULT_ROOT" ]] \
  || fail "task result root is missing"

log "RECOVERY_START host=$HOST_ID task=$TASK_ID devices=$DP_VISIBLE_DEVICES"
wait_for_completed_queue "$RESOURCE_QUEUE" "gpu3_resource"

task_dir="$TASK_RESULT_ROOT/tasks/$TASK_ID"
if [[ ! -d "$task_dir" ]]; then
  log "DATA_PARALLEL_TASK_START task=$TASK_ID devices=$DP_VISIBLE_DEVICES"
  set +e
  CUDA_VISIBLE_DEVICES="$DP_VISIBLE_DEVICES" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONPATH="$DP_REPO" \
  "$PYTHON" -X utf8 -m scripts.reproduction.igstgnn_table4.task_runner \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --task-id "$TASK_ID" \
    --splits-root "$SPLITS_ROOT" \
    --results-root "$TASK_RESULT_ROOT" \
    --device cuda:0 \
    --igstgnn-data-parallel >> "$LOG" 2>&1
  task_rc=$?
  set -e
  [[ "$task_rc" == "0" ]] || fail "data parallel task exited rc=$task_rc"
  log "DATA_PARALLEL_TASK_OK task=$TASK_ID"
else
  log "DATA_PARALLEL_TASK_ALREADY_PRESENT task=$TASK_ID"
fi

log "QUEUE_REVALIDATION_START task=$TASK_ID"
set +e
CUDA_VISIBLE_DEVICES="${DP_VISIBLE_DEVICES%%,*}" \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTHONPATH="$REPO" \
"$PYTHON" -X utf8 -m scripts.reproduction.igstgnn_table4.matrix_runner \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --splits-root "$SPLITS_ROOT" \
  --results-root "$TASK_RESULT_ROOT" \
  --device cuda:0 \
  --task-id "$TASK_ID" >> "$LOG" 2>&1
queue_rc=$?
set -e
[[ "$queue_rc" == "0" ]] || fail "task queue revalidation exited rc=$queue_rc"
log "QUEUE_REVALIDATION_OK task=$TASK_ID"

for index in 0 1 2 3 4; do
  wait_for_completed_queue \
    "${WAIT_QUEUE_GLOB}${index}/queue_status.json" \
    "gpu${index}"
done

log "FINALIZER_START host=$HOST_ID"
HOST_ID="$HOST_ID" \
REPO="$REPO" \
PYTHON="$PYTHON" \
MANIFEST="$MANIFEST" \
MANIFEST_SHA256="$MANIFEST_SHA256" \
RESULTS_BASE="$RESULTS_BASE" \
bash "$FINALIZER" >> "$LOG" 2>&1
