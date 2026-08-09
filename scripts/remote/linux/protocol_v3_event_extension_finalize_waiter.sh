#!/usr/bin/env bash
# Publish extensions only after the baseline publication and formal relay succeed.
set -Eeuo pipefail

BASELINE_FINALIZER_EXITCODE="${BASELINE_FINALIZER_EXITCODE:?BASELINE_FINALIZER_EXITCODE is required}"
BASELINE_FINALIZER_PID="${BASELINE_FINALIZER_PID:-}"
EXTENSION_RELAY_EXITCODE="${EXTENSION_RELAY_EXITCODE:?EXTENSION_RELAY_EXITCODE is required}"
EXTENSION_RELAY_PID="${EXTENSION_RELAY_PID:-}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:?FINALIZER_SCRIPT is required}"
REPO="${REPO:?REPO is required}"
CONTROL_ROOT="${CONTROL_ROOT:?CONTROL_ROOT is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
PLAN="${PLAN:?PLAN is required}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?CANONICAL_MANIFEST is required}"
SUPPORT_MANIFEST="${SUPPORT_MANIFEST:?SUPPORT_MANIFEST is required}"
POSTFLIGHT_SCRIPT="${POSTFLIGHT_SCRIPT:?POSTFLIGHT_SCRIPT is required}"
NAS="${NAS:?NAS is required}"
NAS_BASE="${NAS_BASE:?NAS_BASE is required}"
HOST_ID="${HOST_ID:?HOST_ID is required}"
SSH_KEY="${SSH_KEY:?SSH_KEY is required}"
EXPECTED_TASKS="${EXPECTED_TASKS:-9}"
POST_PUBLISH_COMMAND="${POST_PUBLISH_COMMAND:?POST_PUBLISH_COMMAND is required}"
PYTHON="${PYTHON:-python3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG="${LOG:-$CONTROL_ROOT/finalize_waiter.log}"
FINALIZER_EXITCODE="${FINALIZER_EXITCODE:-$CONTROL_ROOT/finalizer.exitcode}"

mkdir -p "$CONTROL_ROOT"
: >>"$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$LOG"
}

blocked() {
  log "FINALIZE_WAITER_BLOCKED $1"
  exit 1
}

read_zero_exit() {
  local path="$1" label="$2" code
  [ -f "$path" ] || return 1
  code="$(tr -d '[:space:]' <"$path")"
  [ "$code" = "0" ] || blocked "$label exited $code"
}

pid_file_stopped_without_exit() {
  local pid_path="$1" exit_path="$2" pid
  [ -n "$pid_path" ] && [ -f "$pid_path" ] && [ ! -f "$exit_path" ] || return 1
  pid="$(tr -d '[:space:]' <"$pid_path")"
  [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null
}

for required in "$FINALIZER_SCRIPT" "$PLAN" "$CANONICAL_MANIFEST" \
  "$SUPPORT_MANIFEST" "$POSTFLIGHT_SCRIPT" "$SSH_KEY"; do
  [ -f "$required" ] || blocked "required file is missing: $required"
  [ ! -L "$required" ] || blocked "symbolic link is forbidden: $required"
done

while true; do
  relay_ok=0
  baseline_ok=0
  read_zero_exit "$EXTENSION_RELAY_EXITCODE" extension_relay && relay_ok=1
  read_zero_exit "$BASELINE_FINALIZER_EXITCODE" baseline_finalizer && baseline_ok=1
  if [ "$relay_ok" = "1" ] && [ "$baseline_ok" = "1" ]; then
    break
  fi
  if pid_file_stopped_without_exit "$EXTENSION_RELAY_PID" "$EXTENSION_RELAY_EXITCODE"; then
    blocked "extension relay stopped without an exit record"
  fi
  if pid_file_stopped_without_exit "$BASELINE_FINALIZER_PID" "$BASELINE_FINALIZER_EXITCODE"; then
    blocked "baseline finalizer stopped without an exit record"
  fi
  log "WAIT relay_ok=$relay_ok baseline_publish_ok=$baseline_ok"
  sleep "$POLL_SECONDS"
done

log "FINALIZER_START host_id=$HOST_ID expected_tasks=$EXPECTED_TASKS"
set +e
REPO="$REPO" CONTROL_ROOT="$CONTROL_ROOT" RESULTS_ROOT="$RESULTS_ROOT" \
PYTHON="$PYTHON" PLAN="$PLAN" CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
SUPPORT_MANIFEST="$SUPPORT_MANIFEST" POSTFLIGHT_SCRIPT="$POSTFLIGHT_SCRIPT" \
NAS="$NAS" NAS_BASE="$NAS_BASE" HOST_ID="$HOST_ID" SSH_KEY="$SSH_KEY" \
EXPECTED_TASKS="$EXPECTED_TASKS" POLL_SECONDS="$POLL_SECONDS" \
LOG="$CONTROL_ROOT/finalize.log" SHUTDOWN_ON_SUCCESS=1 \
POST_PUBLISH_COMMAND="$POST_PUBLISH_COMMAND" \
  bash "$FINALIZER_SCRIPT"
code=$?
set -e
printf '%s\n' "$code" >"$FINALIZER_EXITCODE"
[ "$code" = "0" ] || blocked "extension finalizer exited $code"
log "FINALIZE_WAITER_SUCCESS"
