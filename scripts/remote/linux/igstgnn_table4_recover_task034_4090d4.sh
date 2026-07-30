#!/usr/bin/env bash
# Recover only Orange STTN task034 on the locked four-GPU host.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/task034_4gpu_recovery_v1.json}
PLAN_SHA256=022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5
PLAN_ID=igstgnn_table4_task034_4gpu_recovery_v1
TASK_ID=034_orange_sttn
HOST_ID=table4_4090d4
EXPECTED_HOSTNAME=autodl-container-51fa4087de-14120c99
GPU_SKU="NVIDIA GeForce RTX 4090 D"
VISIBLE_DEVICES=0,1,2,3
SPLITS_ROOT=${SPLITS_ROOT:-/root/autodl-tmp/table4/data/official_splits}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/task034_recovery/results}
RESULTS_RELATIVE_ROOT=shards/task034_4gpu_recovery
CONTROL_DIR=${CONTROL_DIR:-/root/autodl-tmp/table4/task034_recovery/control}
HARDWARE_INVENTORY=${HARDWARE_INVENTORY:-$CONTROL_DIR/hardware_inventory.json}
GPU_INVENTORY_RAW=${GPU_INVENTORY_RAW:-$CONTROL_DIR/nvidia_smi.csv}
SMOKE_REPORT=${SMOKE_REPORT:-$CONTROL_DIR/smoke.json}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-$CONTROL_DIR/postflight.json}
FINALIZER=${FINALIZER:-$REPO/scripts/remote/linux/igstgnn_table4_finalize_host.sh}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-/root/autodl-tmp/table4/control/table4_4090d4/relay_ed25519}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=/root/xtraffic_results/igstgnn_table4_official_v1/task034_4gpu_recovery_v1/hosts
REMOTE_NAME=table4_4090d4_task034
LOG=${LOG:-$CONTROL_DIR/controller.log}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}

mkdir -p "$CONTROL_DIR" "$RESULTS_BASE/$RESULTS_RELATIVE_ROOT"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "TASK034_RECOVERY_BLOCKED $*"
  exit 1
}

[[ -d "$REPO" && ! -L "$REPO" ]] || fail "repository directory is missing or unsafe"
[[ -x "$PYTHON" ]] || fail "Python runtime is missing"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail "manifest is missing or unsafe"
[[ -f "$PLAN" && ! -L "$PLAN" ]] || fail "recovery plan is missing or unsafe"
[[ -d "$SPLITS_ROOT" && ! -L "$SPLITS_ROOT" ]] || fail "split root is missing or unsafe"
[[ -x "$FINALIZER" && ! -L "$FINALIZER" ]] || fail "finalizer is missing or unsafe"
[[ -x "$NVIDIA_SMI_BIN" && ! -L "$NVIDIA_SMI_BIN" ]] \
  || fail "nvidia-smi is missing or unsafe"
[[ -f "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
  || fail "relay identity is missing or unsafe"
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] \
  || fail "manifest SHA-256 mismatch"
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] \
  || fail "recovery plan SHA-256 mismatch"
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || fail "hostname identity mismatch"
if pgrep -af \
  'igstgnn_table4\.(distributed_host_runner|matrix_runner|task_runner|recovery_task_runner)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "another Table 4 training process is active"
fi

"$NVIDIA_SMI_BIN" \
  --query-gpu=index,name,uuid \
  --format=csv,noheader,nounits > "$GPU_INVENTORY_RAW.tmp" \
  || fail "cannot read GPU inventory"
mv "$GPU_INVENTORY_RAW.tmp" "$GPU_INVENTORY_RAW"

export PYTHONPATH="$REPO"
"$PYTHON" -X utf8 - \
  "$PLAN" "$MANIFEST" "$MANIFEST_SHA256" "$GPU_INVENTORY_RAW" \
  "$HARDWARE_INVENTORY" "$EXPECTED_HOSTNAME" <<'PY' \
  || fail "locked plan or hardware inventory verification failed"
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.reproduction.igstgnn_table4.task034_recovery_postflight import (
    load_task034_recovery_plan,
)

plan_name, manifest_name, manifest_sha, raw_name, output_name, hostname = sys.argv[1:]
plan = load_task034_recovery_plan(
    plan_name,
    manifest_path=manifest_name,
    expected_manifest_sha256=manifest_sha,
)
rows = []
for line in Path(raw_name).read_text(encoding="utf-8").splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise SystemExit("invalid nvidia-smi inventory row")
    index, name, gpu_uuid = parts
    rows.append((int(index), name, gpu_uuid))
rows.sort()
if [index for index, _, _ in rows] != list(range(plan["gpu_count"])):
    raise SystemExit("GPU index inventory mismatch")
if [name for _, name, _ in rows] != [plan["gpu_sku"]] * plan["gpu_count"]:
    raise SystemExit("GPU SKU inventory mismatch")
if [gpu_uuid for _, _, gpu_uuid in rows] != plan["gpu_uuids"]:
    raise SystemExit("GPU UUID inventory mismatch")
if hostname != plan["recovery_hostname"]:
    raise SystemExit("hostname differs from the locked recovery plan")
value = {
    "schema_version": 1,
    "status": "verified",
    "host_id": plan["recovery_host_id"],
    "hostname": hostname,
    "gpu_sku": plan["gpu_sku"],
    "gpu_count": plan["gpu_count"],
    "gpu_uuids": plan["gpu_uuids"],
    "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
target = Path(output_name)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
PY

TASK_RESULT_ROOT="$RESULTS_BASE/$RESULTS_RELATIVE_ROOT"
TASK_DIR="$TASK_RESULT_ROOT/tasks/$TASK_ID"
if [[ ! -e "$TASK_DIR" && ! -L "$TASK_DIR" ]]; then
  log "SMOKE_START task=$TASK_ID devices=$VISIBLE_DEVICES"
  set +e
  CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PYTHON" -X utf8 -m \
    scripts.reproduction.igstgnn_table4.recovery_task_runner \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --task-id "$TASK_ID" \
    --splits-root "$SPLITS_ROOT" \
    --device cuda:0 \
    --adapter multi-gpu-full-batch \
    --smoke > "$SMOKE_REPORT.tmp" 2>> "$LOG"
  smoke_rc=$?
  set -e
  [[ "$smoke_rc" == "0" ]] || fail "four-GPU smoke exited rc=$smoke_rc"
  mv "$SMOKE_REPORT.tmp" "$SMOKE_REPORT"
  "$PYTHON" -X utf8 - "$SMOKE_REPORT" <<'PY' \
    || fail "four-GPU smoke report failed its contract"
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("status") != "verified"
    or value.get("task_id") != "034_orange_sttn"
    or value.get("method") != "sttn"
    or value.get("adapter") != "multi-gpu-full-batch"
    or value.get("device_ids") != [0, 1, 2, 3]
    or value.get("batch_size") != 64
    or value.get("optimizer_steps") != 1
    or value.get("prediction_shape", [])[:2] != [64, 12]
):
    raise SystemExit("invalid task034 smoke report")
peaks = value.get("peak_allocated_bytes")
if (
    not isinstance(peaks, dict)
    or set(peaks) != {"0", "1", "2", "3"}
    or any(type(item) is not int or item <= 0 for item in peaks.values())
):
    raise SystemExit("invalid task034 smoke memory report")
PY
  log "SMOKE_OK task=$TASK_ID report=$SMOKE_REPORT"

  log "TRAIN_START task=$TASK_ID devices=$VISIBLE_DEVICES"
  set +e
  CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PYTHON" -X utf8 -m \
    scripts.reproduction.igstgnn_table4.recovery_task_runner \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --task-id "$TASK_ID" \
    --splits-root "$SPLITS_ROOT" \
    --results-root "$TASK_RESULT_ROOT" \
    --device cuda:0 \
    --adapter multi-gpu-full-batch >> "$LOG" 2>&1
  train_rc=$?
  set -e
  [[ "$train_rc" == "0" ]] || fail "formal task034 training exited rc=$train_rc"
  log "TRAIN_OK task=$TASK_ID"
else
  [[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]] \
    || fail "existing task034 target is unsafe"
  log "TASK_ALREADY_PRESENT task=$TASK_ID"
fi

[[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]] \
  || fail "formal task034 directory is missing"
actual_allowlist="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' \
    | LC_ALL=C sort
)"
expected_allowlist="$(
  printf '%s\n' checkpoint.pt predictions.npz provenance.json summary.json train.log \
    | LC_ALL=C sort
)"
[[ "$actual_allowlist" == "$expected_allowlist" ]] \
  || fail "formal task034 artifact allowlist mismatch"

POSTFLIGHT_CMD="$PYTHON -X utf8 -m scripts.reproduction.igstgnn_table4.task034_recovery_postflight --plan $PLAN --manifest $MANIFEST --manifest-sha256 $MANIFEST_SHA256 --results-base $RESULTS_BASE --hardware-inventory $HARDWARE_INVENTORY --report $POSTFLIGHT_REPORT"
log "FINALIZER_START task=$TASK_ID"
HOST_ID="$HOST_ID" \
RESULTS_BASE="$TASK_RESULT_ROOT" \
PLAN="$PLAN" \
MANIFEST="$MANIFEST" \
MANIFEST_SHA256="$MANIFEST_SHA256" \
EXPECTED_PLAN_ID="$PLAN_ID" \
EXPECTED_EXECUTION_LAYOUT="cross_host_single_task_recovery" \
POSTFLIGHT_CMD="$POSTFLIGHT_CMD" \
POSTFLIGHT_REPORT="$POSTFLIGHT_REPORT" \
TRANSFER_FILE_LIST="$CONTROL_DIR/transfer.files" \
TRANSFER_MANIFEST="$CONTROL_DIR/transfer.sha256" \
TRANSFER_MARKER="$CONTROL_DIR/transfer_complete.json" \
SSH_IDENTITY_FILE="$SSH_IDENTITY_FILE" \
RELAY="$RELAY" \
RELAY_BASE="$RELAY_BASE" \
REMOTE_NAME="$REMOTE_NAME" \
SHUTDOWN_DEFER_FILE="$CONTROL_DIR/no_shutdown_defer" \
SHUTDOWN_BIN="/usr/bin/shutdown" \
LOG="$CONTROL_DIR/finalizer.log" \
PYTHON="$PYTHON" \
bash "$FINALIZER" >> "$LOG" 2>&1
