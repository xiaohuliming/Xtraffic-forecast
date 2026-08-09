#!/usr/bin/env bash
# Start formal extension shards as baseline GPUs become free, then aggregate them.
set -Eeuo pipefail

SHARD_PRIORITY="${SHARD_PRIORITY:-1,2,3}"

validate_shard_priority() {
  local value shard
  local -a parsed=()
  IFS=',' read -r -a parsed <<<"$SHARD_PRIORITY"
  [ "${#parsed[@]}" = "3" ] \
    || { printf 'invalid SHARD_PRIORITY: expected three shards\n' >&2; return 2; }
  for value in 1 2 3; do
    local matches=0
    for shard in "${parsed[@]}"; do
      [ "$shard" = "$value" ] && matches=$((matches + 1))
    done
    [ "$matches" = "1" ] \
      || { printf 'invalid SHARD_PRIORITY: expected permutation of 1,2,3\n' >&2; return 2; }
  done
  PRIORITY_SHARDS=("${parsed[@]}")
}

validate_shard_priority
if [ "${1:-}" = "--print-shard-priority" ]; then
  for shard in "${PRIORITY_SHARDS[@]}"; do
    printf 'gpu%02d\n' "$shard"
  done
  exit 0
fi

BASELINE_RESULTS="${BASELINE_RESULTS:?BASELINE_RESULTS is required}"
REPO="${REPO:?REPO is required}"
CONTROL_ROOT="${CONTROL_ROOT:?CONTROL_ROOT is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"
GRAPH_DIR="${GRAPH_DIR:?GRAPH_DIR is required}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT is required}"
NODE_ORDER="${NODE_ORDER:?NODE_ORDER is required}"
CALIBRATION="${CALIBRATION:?CALIBRATION is required}"
LOADING_AUDIT="${LOADING_AUDIT:?LOADING_AUDIT is required}"
CACHE_BUNDLE_ID="${CACHE_BUNDLE_ID:?CACHE_BUNDLE_ID is required}"
CACHE_BUNDLE_DIGEST="${CACHE_BUNDLE_DIGEST:?CACHE_BUNDLE_DIGEST is required}"
CODE_SHA="${CODE_SHA:?CODE_SHA is required}"
MATRIX_ID="${MATRIX_ID:?MATRIX_ID is required}"
HOST_ID="${HOST_ID:?HOST_ID is required}"
PLAN_DIGEST="${PLAN_DIGEST:?PLAN_DIGEST is required}"
PYTHON="${PYTHON:-python3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG="${LOG:-$CONTROL_ROOT/relay.log}"

MANIFEST="$REPO/config/protocol_v3/table4_event_extensions_v1.json"
SUPPORT_MANIFEST="$REPO/config/protocol_v3/event_extension_support_v1.json"
PLAN="$CONTROL_ROOT/plan.json"

mkdir -p "$CONTROL_ROOT/logs" "$RESULTS_ROOT"
: >>"$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$LOG"
}

blocked() {
  log "RELAY_BLOCKED $1"
  exit 1
}

json_get() {
  "$PYTHON" -c '
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value=value.get(key) if isinstance(value,dict) else None
print("" if value is None else value)
' "$1" "$2"
}

verify_identity() {
  [ -f "$PLAN" ] || blocked "formal plan is missing"
  [ "$(json_get "$PLAN" plan_digest)" = "$PLAN_DIGEST" ] \
    || blocked "formal plan digest changed"
  [ "$(json_get "$PLAN" matrix_id)" = "$MATRIX_ID" ] \
    || blocked "formal matrix identity changed"
  [ "$(json_get "$PLAN" host_id)" = "$HOST_ID" ] \
    || blocked "formal host identity changed"
  [ "$(json_get "$PLAN" code_sha)" = "$CODE_SHA" ] \
    || blocked "formal code identity changed"
  [ "$(json_get "$PLAN" gpu_count)" = "4" ] \
    || blocked "formal plan is not the reviewed four-GPU plan"
}

baseline_state() {
  local physical_gpu="$1"
  local queue="$BASELINE_RESULTS/gpu$(printf '%02d' "$physical_gpu")/queue_status.json"
  if [ ! -f "$queue" ]; then
    printf 'missing\n'
    return
  fi
  json_get "$queue" state
}

baseline_gpu_busy() {
  local physical_gpu="$1"
  local root="$BASELINE_RESULTS/gpu$(printf '%02d' "$physical_gpu")"
  pgrep -af 'protocol_v3_baseline_runner|train_baseline_zoo|train_graphwavenet' \
    | grep -F -- "$root" >/dev/null 2>&1
}

shard_exit_path() {
  printf '%s/manual_shard_gpu%02d.exitcode\n' "$CONTROL_ROOT" "$1"
}

shard_pid_path() {
  printf '%s/manual_shard_gpu%02d.pid\n' "$CONTROL_ROOT" "$1"
}

shard_physical_gpu_path() {
  printf '%s/manual_shard_gpu%02d.physical_gpu\n' "$CONTROL_ROOT" "$1"
}

shard_is_running() {
  local shard="$1" pid_path pid
  pid_path="$(shard_pid_path "$shard")"
  [ -f "$pid_path" ] || return 1
  pid="$(tr -d '[:space:]' <"$pid_path")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

shard_is_done() {
  local shard="$1" exit_path code
  exit_path="$(shard_exit_path "$shard")"
  [ -f "$exit_path" ] || return 1
  code="$(tr -d '[:space:]' <"$exit_path")"
  [ "$code" = "0" ] || blocked "formal shard gpu$(printf '%02d' "$shard") exited $code"
}

launch_shard() {
  local physical_gpu="$1" shard="$2"
  local shard_name manifest results log_path pid_path exit_path pid
  shard_name="gpu$(printf '%02d' "$shard")"
  manifest="$CONTROL_ROOT/manifests/shard_${shard_name}.json"
  results="$RESULTS_ROOT/$shard_name"
  log_path="$CONTROL_ROOT/logs/shard_${shard_name}.log"
  pid_path="$(shard_pid_path "$shard")"
  exit_path="$(shard_exit_path "$shard")"
  [ -f "$manifest" ] || blocked "missing formal shard manifest $manifest"
  if shard_is_done "$shard"; then
    return
  fi
  if shard_is_running "$shard"; then
    return
  fi
  [ ! -f "$exit_path" ] || blocked "stale nonzero exit file for $shard_name"
  log "LAUNCH physical_gpu=$physical_gpu formal_shard=$shard_name"
  (
    set +e
    cd "$REPO" || exit 97
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$REPO" \
      "$PYTHON" -X utf8 \
      -m scripts.remote.windows.protocol_v3_event_extension_runner \
      --allow-shard --manifest "$manifest" \
      --support-manifest "$SUPPORT_MANIFEST" \
      --repo-root "$REPO" --results-root "$results" \
      --cache-dir "$CACHE_DIR" --graph-dir "$GRAPH_DIR" \
      --support-root "$SUPPORT_ROOT" --node-order "$NODE_ORDER" \
      --cache-bundle-id "$CACHE_BUNDLE_ID" \
      --cache-bundle-digest "$CACHE_BUNDLE_DIGEST" \
      --code-sha "$CODE_SHA" --python "$PYTHON"
    code=$?
    printf '%s\n' "$code" >"$exit_path"
    exit "$code"
  ) >>"$log_path" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  printf '%s\n' "$physical_gpu" >"$(shard_physical_gpu_path "$shard")"
  log "LAUNCHED physical_gpu=$physical_gpu formal_shard=$shard_name pid=$pid"
}

physical_gpu_has_extension() {
  local physical_gpu="$1" shard assignment_path assigned
  for shard in 0 1 2 3; do
    shard_is_done "$shard" && continue
    shard_is_running "$shard" || continue
    assignment_path="$(shard_physical_gpu_path "$shard")"
    [ -f "$assignment_path" ] \
      || blocked "running formal shard gpu$(printf '%02d' "$shard") has no physical GPU record"
    assigned="$(tr -d '[:space:]' <"$assignment_path")"
    [ "$assigned" != "$physical_gpu" ] || return 0
  done
  return 1
}

physical_gpu_is_available() {
  local physical_gpu="$1" state
  physical_gpu_has_extension "$physical_gpu" && return 1
  state="$(baseline_state "$physical_gpu")"
  case "$state" in
    failed) blocked "baseline gpu$(printf '%02d' "$physical_gpu") failed" ;;
    completed)
      baseline_gpu_busy "$physical_gpu" && return 1
      return 0
      ;;
    *) return 1 ;;
  esac
}

next_unstarted_shard() {
  local shard
  for shard in "${PRIORITY_SHARDS[@]}"; do
    shard_is_done "$shard" && continue
    shard_is_running "$shard" && continue
    printf '%s\n' "$shard"
    return 0
  done
  return 1
}

verify_identity
log "RELAY_PRIORITY formal_shards=$(IFS=,; printf '%s' "${PRIORITY_SHARDS[*]}") plan_digest=$PLAN_DIGEST"

# gpu00 is launched before this relay because physical GPU 3 was already free.
if ! shard_is_running 0 && ! shard_is_done 0; then
  blocked "formal shard gpu00 is neither running nor completed"
fi
[ -f "$(shard_physical_gpu_path 0)" ] \
  || printf '3\n' >"$(shard_physical_gpu_path 0)"

while true; do
  for physical_gpu in 3 2 1 0; do
    physical_gpu_is_available "$physical_gpu" || continue
    if shard="$(next_unstarted_shard)"; then
      launch_shard "$physical_gpu" "$shard"
    fi
  done

  if shard_is_done 0 && shard_is_done 1 && shard_is_done 2 && shard_is_done 3; then
    break
  fi
  sleep "$POLL_SECONDS"
done

log "ALL_MANUAL_SHARDS_COMPLETED"
verify_identity
rm -f "$CONTROL_ROOT/controller.pid" "$CONTROL_ROOT/runner.exitcode"
set +e
cd "$REPO"
PYTHONPATH="$REPO" "$PYTHON" -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_extension_multi_gpu \
  --manifest "$MANIFEST" --support-manifest "$SUPPORT_MANIFEST" \
  --loading-audit "$LOADING_AUDIT" --calibration "$CALIBRATION" \
  --repo-root "$REPO" --control-root "$CONTROL_ROOT" \
  --results-root "$RESULTS_ROOT" --cache-dir "$CACHE_DIR" \
  --graph-dir "$GRAPH_DIR" --support-root "$SUPPORT_ROOT" \
  --node-order "$NODE_ORDER" --cache-bundle-id "$CACHE_BUNDLE_ID" \
  --cache-bundle-digest "$CACHE_BUNDLE_DIGEST" --code-sha "$CODE_SHA" \
  --matrix-id "$MATRIX_ID" --host-id "$HOST_ID" --gpu-count 4 \
  --python "$PYTHON" --expect-plan-digest "$PLAN_DIGEST" \
  >>"$CONTROL_ROOT/logs/aggregate.log" 2>&1
aggregate_code=$?
set -e
[ "$aggregate_code" = "0" ] || blocked "formal aggregate exited $aggregate_code"
[ "$(tr -d '[:space:]' <"$CONTROL_ROOT/runner.exitcode")" = "0" ] \
  || blocked "formal aggregate runner exit code is not zero"
log "RELAY_SUCCESS formal_tasks=9 plan_digest=$PLAN_DIGEST"
