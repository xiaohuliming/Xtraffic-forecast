#!/usr/bin/env bash
# Export the twelve completed host5 tasks without training, then halt.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/results}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/host5_partial_export_12_v1.json}
PLAN_SHA256=cd81652493690d1d329f5eecd45fe541c9de7db26c93650fc490a2507616845d
PLAN_ID=igstgnn_table4_host5_partial_export_12_v1
HOST_ID=table4_4090d5
EXPECTED_HOSTNAME=autodl-container-caf44cbe32-9a5094c3
CONTROL_DIR=${CONTROL_DIR:-/root/autodl-tmp/table4/host5_partial_export_12_v1}
GPU_INVENTORY=${GPU_INVENTORY:-$CONTROL_DIR/nvidia_smi.csv}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-$CONTROL_DIR/postflight.json}
FINALIZER=${FINALIZER:-$REPO/scripts/remote/linux/igstgnn_table4_finalize_host.sh}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-/root/autodl-tmp/table4/control/table4_4090d5/relay_ed25519}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=/root/xtraffic_results/igstgnn_table4_official_v1/original_host5_partial_12_v1/hosts
REMOTE_NAME=table4_4090d5_partial12
LOG=${LOG:-$CONTROL_DIR/controller.log}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}

mkdir -p "$CONTROL_DIR"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "HOST5_PARTIAL_EXPORT_BLOCKED $*"
  exit 1
}

[[ -d "$REPO" && ! -L "$REPO" ]] || fail "repository directory is missing or unsafe"
[[ -x "$PYTHON" ]] || fail "Python runtime is missing"
[[ -d "$RESULTS_BASE" && ! -L "$RESULTS_BASE" ]] \
  || fail "results base is missing or unsafe"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail "manifest is missing or unsafe"
[[ -f "$PLAN" && ! -L "$PLAN" ]] || fail "partial export plan is missing or unsafe"
[[ -x "$FINALIZER" && ! -L "$FINALIZER" ]] || fail "finalizer is missing or unsafe"
[[ -f "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
  || fail "relay identity is missing or unsafe"
[[ -x "$NVIDIA_SMI_BIN" ]] || fail "nvidia-smi is missing"
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || fail "hostname identity mismatch"
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] \
  || fail "manifest SHA-256 mismatch"
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] \
  || fail "partial export plan SHA-256 mismatch"
if pgrep -af \
  'igstgnn_table4\.(distributed_host_runner|matrix_runner|task_runner|recovery_task_runner)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "a Table 4 training process is active"
fi
if pgrep -af \
  'igstgnn_table4_(finalize_host|original_export_shutdown_waiter|ndr_extension_handoff)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "another Table 4 finalizer or waiter is active"
fi

task034="$RESULTS_BASE/shards/table4_4090d5_gpu0/tasks/034_orange_sttn"
[[ ! -e "$task034" && ! -L "$task034" ]] \
  || fail "task034 must remain absent from the host5 partial export"
summary_count="$(
  find "$RESULTS_BASE" -path '*/tasks/*/summary.json' -type f | wc -l | tr -d ' '
)"
[[ "$summary_count" == "12" ]] || fail "host5 summary count is not exactly 12"

"$NVIDIA_SMI_BIN" \
  --query-gpu=index,name,uuid \
  --format=csv,noheader,nounits > "$GPU_INVENTORY.tmp" \
  || fail "cannot read GPU inventory"
mv "$GPU_INVENTORY.tmp" "$GPU_INVENTORY"
export PYTHONPATH="$REPO"
"$PYTHON" -X utf8 - \
  "$PLAN" "$MANIFEST" "$MANIFEST_SHA256" "$GPU_INVENTORY" <<'PY' \
  || fail "live GPU inventory differs from the locked export plan"
import sys
from pathlib import Path

from scripts.reproduction.igstgnn_table4.host5_partial_export_postflight import (
    load_host5_partial_export_plan,
)

plan_name, manifest_name, manifest_sha, inventory_name = sys.argv[1:]
plan = load_host5_partial_export_plan(
    plan_name,
    manifest_path=manifest_name,
    expected_manifest_sha256=manifest_sha,
)
rows = []
for line in Path(inventory_name).read_text(encoding="utf-8").splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise SystemExit("invalid nvidia-smi row")
    rows.append((int(parts[0]), parts[1], parts[2]))
rows.sort()
if [row[0] for row in rows] != list(range(plan["gpu_count"])):
    raise SystemExit("GPU index mismatch")
if [row[1] for row in rows] != [plan["gpu_sku"]] * plan["gpu_count"]:
    raise SystemExit("GPU SKU mismatch")
if [row[2] for row in rows] != plan["gpu_uuids"]:
    raise SystemExit("GPU UUID mismatch")
PY

POSTFLIGHT_CMD="$PYTHON -X utf8 -m scripts.reproduction.igstgnn_table4.host5_partial_export_postflight --plan $PLAN --manifest $MANIFEST --manifest-sha256 $MANIFEST_SHA256 --results-base $RESULTS_BASE --report $POSTFLIGHT_REPORT"
log "HOST5_PARTIAL_EXPORT_START tasks=12"
HOST_ID="$HOST_ID" \
RESULTS_BASE="$RESULTS_BASE" \
PLAN="$PLAN" \
MANIFEST="$MANIFEST" \
MANIFEST_SHA256="$MANIFEST_SHA256" \
EXPECTED_PLAN_ID="$PLAN_ID" \
EXPECTED_EXECUTION_LAYOUT="single_host_partial_original_export" \
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
