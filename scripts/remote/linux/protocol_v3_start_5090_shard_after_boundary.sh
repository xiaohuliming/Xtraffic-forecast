#!/usr/bin/env bash
set -uo pipefail

PYTHON=/root/miniconda3/bin/python
BOUNDARY_STATUS=/root/autodl-tmp/protocol_v3_boundary_stop_status.json
BOUNDARY_PID_FILE=/root/autodl-tmp/protocol_v3_boundary_stop.pid
OLD_ORCHESTRATOR=/root/autodl-tmp/protocol_v3_5090_orchestrator.py
LAUNCHER=/root/autodl-tmp/protocol_v3_launch_5090_shard.sh
SHARD_PID_FILE=/root/autodl-tmp/protocol_v3_5090_shard.pid
SHARD_LOG=/root/autodl-tmp/protocol_v3_5090_shard.log
FINALIZER=/root/autodl-tmp/protocol_v3_finalize_5090_shard.sh
FINALIZER_PID_FILE=/root/autodl-tmp/protocol_v3_5090_shard_finalize.pid
FINALIZER_CONSOLE=/root/autodl-tmp/protocol_v3_5090_shard_finalize_console.log
TRANSITION_LOG=/root/autodl-tmp/protocol_v3_5090_transition.log
RESULTS=/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090
BOUNDARY_TASK=006_event_staeformer_s0c_orange_seed42

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$TRANSITION_LOG"
}

json_state() {
  "$PYTHON" - "$BOUNDARY_STATUS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("missing")
else:
    print(payload.get("state", "invalid"))
PY
}

log "WAIT_BOUNDARY task=$BOUNDARY_TASK"
while true; do
  state="$(json_state)"
  case "$state" in
    completed)
      break
      ;;
    failed)
      log "FAILED boundary watcher reported failure"
      exit 2
      ;;
  esac
  if [[ -f "$BOUNDARY_PID_FILE" ]]; then
    watcher_pid="$(tr -d '[:space:]' < "$BOUNDARY_PID_FILE")"
    if [[ -n "$watcher_pid" ]] && ! kill -0 "$watcher_pid" 2>/dev/null; then
      log "FAILED boundary watcher exited with state=$state"
      exit 3
    fi
  fi
  sleep 1
done

summary="$RESULTS/tasks/$BOUNDARY_TASK/summary.json"
task_state="$RESULTS/state/tasks/$BOUNDARY_TASK.json"
if [[ ! -s "$summary" || ! -s "$task_state" ]]; then
  log "FAILED boundary artifacts missing"
  exit 4
fi

for _ in $(seq 1 20); do
  if ! pgrep -f "$OLD_ORCHESTRATOR" >/dev/null \
    && ! pgrep -f 'scripts/train_.*task007' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f "$OLD_ORCHESTRATOR" >/dev/null; then
  log "FAILED old orchestrator is still running"
  exit 5
fi
if [[ -d "$RESULTS/tasks/007_event_staeformer_s0a_alameda_seed1" ]]; then
  log "FAILED obsolete queue promoted task007"
  exit 6
fi
if [[ -f "$SHARD_PID_FILE" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$SHARD_PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    log "FAILED 5090 shard is already running pid=$existing_pid"
    exit 7
  fi
fi

nohup setsid bash "$LAUNCHER" > "$SHARD_LOG" 2>&1 < /dev/null &
shard_pid=$!
printf '%s\n' "$shard_pid" > "$SHARD_PID_FILE"
sleep 2
if ! kill -0 "$shard_pid" 2>/dev/null; then
  log "FAILED 5090 shard exited during launch"
  tail -n 40 "$SHARD_LOG" | tee -a "$TRANSITION_LOG"
  exit 8
fi
log "STARTED 5090 shard pid=$shard_pid"

nohup setsid bash "$FINALIZER" > "$FINALIZER_CONSOLE" 2>&1 < /dev/null &
finalizer_pid=$!
printf '%s\n' "$finalizer_pid" > "$FINALIZER_PID_FILE"
sleep 1
if ! kill -0 "$finalizer_pid" 2>/dev/null; then
  log "FAILED 5090 shard finalizer exited during launch"
  tail -n 40 "$FINALIZER_CONSOLE" | tee -a "$TRANSITION_LOG"
  exit 9
fi
log "STARTED 5090 shard finalizer pid=$finalizer_pid"
