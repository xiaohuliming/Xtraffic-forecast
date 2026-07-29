#!/usr/bin/env bash
# Finalize the four-GPU NDR rebalance without changing its six prior task artifacts.
set -euo pipefail

HOST_ID=${HOST_ID:-table4_4090d4}
ROOT=${ROOT:-/root/autodl-tmp/table4}
REPO=${REPO:-$ROOT/ndr_repo}
RESULTS_BASE=${RESULTS_BASE:-$ROOT/ndr_extension/results}
CONTROL_DIR=${CONTROL_DIR:-$ROOT/ndr_extension/control/table4_4090d4_rebalance_v2}
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/distributed_ndr_extension_4gpu_rebalance_v2.json}
PLAN_SHA256=${PLAN_SHA256:-56a71fe2a3b8101ae282dc6f9b9efab07874108a40922276285206e99cbf2612}
HOST_STATUS=${HOST_STATUS:-$CONTROL_DIR/host_status.json}
PRIOR_TASK_ARTIFACTS=${PRIOR_TASK_ARTIFACTS:-$CONTROL_DIR/prior_task_artifacts.sha256}
PRIOR_TASK_ARTIFACT_COUNT=${PRIOR_TASK_ARTIFACT_COUNT:-30}
SHUTDOWN_GUARD=${SHUTDOWN_GUARD:-$CONTROL_DIR/shutdown_guard}
HANDOFF_BIN=${HANDOFF_BIN:-$REPO/scripts/remote/linux/igstgnn_table4_ndr_extension_handoff.sh}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/igstgnn_table4_official_v1/ndr_extension_4gpu_rebalance_v2/hosts}
WAIT_SECONDS=${WAIT_SECONDS:-60}
WAIT_ONCE=${WAIT_ONCE:-0}
LOG=${LOG:-$CONTROL_DIR/rebalance_finalize.log}

mkdir -p "$CONTROL_DIR"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "NDR_REBALANCE_FINALIZE_BLOCKED $*"
  exit 1
}

[[ "$HOST_ID" == "table4_4090d4" ]] || fail "invalid HOST_ID"
[[ -d "$REPO" && ! -L "$REPO" ]] || fail "repository is missing or unsafe"
[[ -d "$RESULTS_BASE" && ! -L "$RESULTS_BASE" ]] \
  || fail "results base is missing or unsafe"
[[ -f "$PLAN" && ! -L "$PLAN" ]] || fail "plan is missing or unsafe"
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] \
  || fail "plan hash mismatch"
[[ -f "$PRIOR_TASK_ARTIFACTS" && ! -L "$PRIOR_TASK_ARTIFACTS" ]] \
  || fail "prior task artifact manifest is missing or unsafe"
[[ -f "$SHUTDOWN_GUARD" && ! -L "$SHUTDOWN_GUARD" ]] \
  || fail "shutdown guard is missing or unsafe"
[[ -x "$HANDOFF_BIN" && ! -L "$HANDOFF_BIN" ]] \
  || fail "handoff controller is missing or unsafe"

log "NDR_REBALANCE_FINALIZE_WAIT status=$HOST_STATUS"
while true; do
  if [[ -s "$HOST_STATUS" && ! -L "$HOST_STATUS" ]]; then
    set +e
    python3 - "$HOST_STATUS" "$HOST_ID" "$PLAN_SHA256" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") == "running":
    raise SystemExit(3)
if (
    status.get("status") != "completed"
    or status.get("host_id") != sys.argv[2]
    or status.get("plan_sha256") != sys.argv[3]
):
    raise SystemExit(4)
shards = status.get("shards")
if (
    not isinstance(shards, list)
    or len(shards) != 4
    or any(shard.get("returncode") != 0 for shard in shards)
):
    raise SystemExit(5)
PY
    status_rc=$?
    set -e
    if [[ "$status_rc" == "0" ]]; then
      break
    fi
    if [[ "$status_rc" != "3" ]]; then
      fail "host status is failed or invalid"
    fi
  fi
  if [[ "$WAIT_ONCE" == "1" ]]; then
    fail "host run is not complete"
  fi
  sleep "$WAIT_SECONDS"
done

while pgrep -af \
  'scripts\.reproduction\.igstgnn_table4\.(ndr_extension_distributed|ndr_extension_matrix_runner|ndr_extension_runner)' \
  | grep -v 'pgrep -af' >/dev/null; do
  if [[ "$WAIT_ONCE" == "1" ]]; then
    fail "NDR training process remains active after completed status"
  fi
  sleep "$WAIT_SECONDS"
done

prior_count=$(wc -l < "$PRIOR_TASK_ARTIFACTS" | tr -d ' ')
[[ "$prior_count" == "$PRIOR_TASK_ARTIFACT_COUNT" ]] \
  || fail "prior task artifact count mismatch"
(
  cd "$RESULTS_BASE"
  sha256sum -c "$PRIOR_TASK_ARTIFACTS" >/dev/null
) || fail "a prior task artifact changed during rebalance"
log "NDR_REBALANCE_PRIOR_ARTIFACTS_OK count=$prior_count"

log "NDR_REBALANCE_HANDOFF_START destination=$RELAY_BASE/$HOST_ID"
exec env \
  HOST_ID="$HOST_ID" \
  ROOT="$ROOT" \
  REPO="$REPO" \
  RESULTS_BASE="$RESULTS_BASE" \
  CONTROL_DIR="$CONTROL_DIR" \
  PLAN="$PLAN" \
  SHUTDOWN_DEFER_FILE="$SHUTDOWN_GUARD" \
  RELAY_BASE="$RELAY_BASE" \
  HANDOFF_WAIT_ONCE=1 \
  "$HANDOFF_BIN"
