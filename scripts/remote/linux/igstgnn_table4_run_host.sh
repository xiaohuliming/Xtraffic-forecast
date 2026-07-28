#!/usr/bin/env bash
# Run all shards assigned to one Table 4 host, then finalize and halt on success.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
HOST_ID=${HOST_ID:?HOST_ID is required}
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/distributed_4090d_9gpu_v1.json}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=${MANIFEST_SHA256:-d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78}
SPLITS_ROOT=${SPLITS_ROOT:-/root/autodl-tmp/table4/data/official_splits}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/results}
CONTROL_DIR=${CONTROL_DIR:-/root/autodl-tmp/table4/control/$HOST_ID}
FINALIZER=${FINALIZER:-$REPO/scripts/remote/linux/igstgnn_table4_finalize_host.sh}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-$CONTROL_DIR/supervisor.log}

mkdir -p "$CONTROL_DIR"
printf '%s HOST_RUN_START host=%s\n' "$(date -u +%FT%TZ)" "$HOST_ID" \
  >> "$SUPERVISOR_LOG"

set +e
PYTHONPATH="$REPO" "$PYTHON" -X utf8 -m \
  scripts.reproduction.igstgnn_table4.distributed_host_runner \
  --plan "$PLAN" \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --host-id "$HOST_ID" \
  --splits-root "$SPLITS_ROOT" \
  --results-base "$RESULTS_BASE" \
  --control-dir "$CONTROL_DIR" \
  --python "$PYTHON" >> "$SUPERVISOR_LOG" 2>&1
runner_rc=$?
set -e

printf '%s HOST_RUN_EXIT host=%s rc=%s\n' \
  "$(date -u +%FT%TZ)" "$HOST_ID" "$runner_rc" >> "$SUPERVISOR_LOG"
if [[ "$runner_rc" != "0" ]]; then
  printf '%s FINALIZE_NOT_STARTED host=%s reason=runner_failed\n' \
    "$(date -u +%FT%TZ)" "$HOST_ID" >> "$SUPERVISOR_LOG"
  exit "$runner_rc"
fi

HOST_ID="$HOST_ID" \
REPO="$REPO" \
PYTHON="$PYTHON" \
PLAN="$PLAN" \
MANIFEST="$MANIFEST" \
MANIFEST_SHA256="$MANIFEST_SHA256" \
RESULTS_BASE="$RESULTS_BASE" \
bash "$FINALIZER" >> "$SUPERVISOR_LOG" 2>&1
