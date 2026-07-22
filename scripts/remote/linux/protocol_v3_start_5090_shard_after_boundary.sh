#!/usr/bin/env bash
# Transition the RTX 5090 from the old 21-task shard to the audited 18-task shard.
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/bin/python}
REPO=${REPO:-/root/xtraffic_v3_5090_clean}
OPS_ROOT=${OPS_ROOT:-/root/autodl-tmp/xtraffic_ops_27_18}
RESULTS=${RESULTS:-/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090}
MANIFEST=${MANIFEST:-$REPO/scripts/remote/windows/protocol_v3_manifest.json}
CACHE_DIR=${CACHE_DIR:-$REPO/outputs/dist_net/region_data_v3_noleak_dedup}
GRAPH_DIR=${GRAPH_DIR:-$REPO/outputs/region_graphs}
BOUNDARY_STATUS=${BOUNDARY_STATUS:-/root/autodl-tmp/protocol_v3_repartition_27_18_boundary_status.json}
BOUNDARY_TASK=${BOUNDARY_TASK:-039_fullwindow_staeformer_base_alameda_seed42}
FORBIDDEN_TASK=${FORBIDDEN_TASK:-040_fullwindow_graphwavenet_base_contracosta_seed42}
BOUNDARY_VALIDATE_CMD=${BOUNDARY_VALIDATE_CMD:-}
OLD_RUNNER_PID_FILE=${OLD_RUNNER_PID_FILE:-/root/autodl-tmp/protocol_v3_5090_shard.pid}
OLD_FINALIZER_PID_FILE=${OLD_FINALIZER_PID_FILE:-/root/autodl-tmp/protocol_v3_5090_shard_finalize.pid}
OLD_FINALIZER_COMMAND_MARKER=${OLD_FINALIZER_COMMAND_MARKER:-protocol_v3_finalize_5090_shard.sh}
LAUNCHER=${LAUNCHER:-/root/autodl-tmp/protocol_v3_launch_5090_shard18.sh}
SHARD_PID_FILE=${SHARD_PID_FILE:-/root/autodl-tmp/protocol_v3_5090_shard18.pid}
SHARD_LOG=${SHARD_LOG:-/root/autodl-tmp/protocol_v3_5090_shard18.log}
FINALIZER=${FINALIZER:-/root/autodl-tmp/protocol_v3_finalize_5090_shard18.sh}
FINALIZER_PID_FILE=${FINALIZER_PID_FILE:-/root/autodl-tmp/protocol_v3_5090_shard18_finalize.pid}
FINALIZER_CONSOLE=${FINALIZER_CONSOLE:-/root/autodl-tmp/protocol_v3_5090_shard18_finalize_console.log}
TRANSITION_LOG=${TRANSITION_LOG:-/root/autodl-tmp/protocol_v3_5090_shard18_transition.log}
READY_MARKER=${READY_MARKER:-/root/autodl-tmp/protocol_v3_repartition_27_18_boundary_ready.json}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_CONTROL_DIR=${RELAY_CONTROL_DIR:-/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/control}
READY_NAME=${READY_NAME:-repartition_27_18_boundary_ready.json}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
SETSID_BIN=${SETSID_BIN:-/usr/bin/setsid}
POLL_SECONDS=${POLL_SECONDS:-1}
MAX_WAIT_ITERATIONS=${MAX_WAIT_ITERATIONS:-0}
STOP_WAIT_SECONDS=${STOP_WAIT_SECONDS:-10}
LAUNCH_GRACE_SECONDS=${LAUNCH_GRACE_SECONDS:-2}
QUEUE_POLL_SECONDS=${QUEUE_POLL_SECONDS:-1}
QUEUE_MAX_WAIT_ITERATIONS=${QUEUE_MAX_WAIT_ITERATIONS:-300}

CACHE_BUNDLE_ID=bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4
CACHE_BUNDLE_DIGEST=854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5
CODE_SHA=3a4952837aaf6fbda1403d50b610b1bf40c6098a
SSH_OPTIONS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)

mkdir -p "$(dirname "$TRANSITION_LOG")"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$TRANSITION_LOG"
}

fail() {
  log "TRANSITION_BLOCKED $*"
  exit 1
}

process_alive() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local process_state
  process_state="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$process_state" && "$process_state" != Z* ]]
}

read_boundary_state() {
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

read_new_queue_state() {
  "$PYTHON" - "$RESULTS/queue_status.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    queue = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("stale missing null")
    raise SystemExit(0)
if not isinstance(queue, dict):
    print("invalid non_object null")
    raise SystemExit(0)
total = queue.get("total_tasks")
if type(total) is not int or total != 18:
    print("stale old_contract null")
    raise SystemExit(0)
state = queue.get("state")
current = queue.get("current_task")
allowed_current = {
    None,
    "043_fullwindow_graphwavenet_base_orange_seed42",
}
if state not in {"preflight", "running"} or current not in allowed_current:
    print(f"invalid {state or 'missing'} {current or 'null'}")
    raise SystemExit(0)
print(f"ready {state} {current or 'null'}")
PY
}

[[ "$RELAY_CONTROL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || fail "unsafe relay control directory"
[[ "$READY_NAME" =~ ^[A-Za-z0-9._-]+$ ]] \
  || fail "unsafe ready marker name"
[[ "$BOUNDARY_TASK" == "039_fullwindow_staeformer_base_alameda_seed42" ]] \
  || fail "unexpected boundary task: $BOUNDARY_TASK"
[[ "$FORBIDDEN_TASK" == "040_fullwindow_graphwavenet_base_contracosta_seed42" ]] \
  || fail "unexpected forbidden task: $FORBIDDEN_TASK"

log "WAIT_BOUNDARY task=$BOUNDARY_TASK status=$BOUNDARY_STATUS"
wait_iterations=0
while true; do
  boundary_state="$(read_boundary_state)"
  case "$boundary_state" in
    completed)
      break
      ;;
    failed)
      fail "boundary watcher reported failure"
      ;;
    waiting|missing|invalid)
      ;;
    *)
      fail "unexpected boundary watcher state: $boundary_state"
      ;;
  esac
  wait_iterations=$((wait_iterations + 1))
  if (( MAX_WAIT_ITERATIONS > 0 && wait_iterations >= MAX_WAIT_ITERATIONS )); then
    fail "boundary wait limit reached with state=$boundary_state"
  fi
  sleep "$POLL_SECONDS"
done

read -r old_runner_pid old_runner_pgid < <(
  "$PYTHON" - "$BOUNDARY_STATUS" "$BOUNDARY_TASK" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_task = sys.argv[2]
if payload.get("state") != "completed":
    raise SystemExit("boundary state is not completed")
if payload.get("task_id") != expected_task:
    raise SystemExit("boundary task identity mismatch")
if payload.get("stop_result") not in {
    "terminated",
    "killed_after_grace",
    "already_exited",
}:
    raise SystemExit("boundary stop result is invalid")
pid = payload.get("pid")
pgid = payload.get("pgid")
if (
    isinstance(pid, bool)
    or not isinstance(pid, int)
    or pid <= 0
    or isinstance(pgid, bool)
    or not isinstance(pgid, int)
    or pgid <= 0
):
    raise SystemExit("boundary process identity is invalid")
print(pid, pgid)
PY
) || fail "boundary status validation failed"

process_alive "$old_runner_pid" \
  && fail "old 5090 runner is still alive pid=$old_runner_pid"
if [[ -s "$OLD_RUNNER_PID_FILE" ]]; then
  recorded_old_runner="$(tr -d '[:space:]' < "$OLD_RUNNER_PID_FILE")"
  [[ "$recorded_old_runner" == "$old_runner_pid" ]] \
    || fail "old runner PID differs from boundary status"
  process_alive "$recorded_old_runner" \
    && fail "old runner PID file still refers to a live process"
fi

summary="$RESULTS/tasks/$BOUNDARY_TASK/summary.json"
task_state="$RESULTS/state/tasks/$BOUNDARY_TASK.json"
forbidden_result="$RESULTS/tasks/$FORBIDDEN_TASK"
[[ ! -e "$forbidden_result" ]] \
  || fail "obsolete 5090 queue promoted task040"

export SUMMARY_PATH="$summary"
export TASK_STATE_PATH="$task_state"
if [[ -n "$BOUNDARY_VALIDATE_CMD" ]]; then
  bash -lc "$BOUNDARY_VALIDATE_CMD" \
    || fail "task039 validation command failed"
else
  if ! "$PYTHON" -X utf8 - \
    "$REPO" \
    "$MANIFEST" \
    "$RESULTS" \
    "$CACHE_DIR" \
    "$GRAPH_DIR" \
    "$PYTHON" \
    "$BOUNDARY_TASK" \
    "$CACHE_BUNDLE_ID" \
    "$CACHE_BUNDLE_DIGEST" \
    "$CODE_SHA" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
results_root = Path(sys.argv[3]).resolve()
cache_dir = Path(sys.argv[4]).resolve()
graph_dir = Path(sys.argv[5]).resolve()
python_executable = sys.argv[6]
boundary_task = sys.argv[7]
sys.path.insert(0, str(repo))

from scripts.remote.windows import protocol_v3_runner as runner

manifest = runner.load_manifest(
    manifest_path,
    require_production_matrix=True,
)
task = next(
    (task for task in manifest.tasks if task.task_id == boundary_task),
    None,
)
if task is None:
    raise SystemExit("task039 is absent from canonical manifest")
run_inputs = runner.ProtocolV3Inputs(
    results_root=results_root,
    cache_dir=cache_dir,
    graph_dir=graph_dir,
    cache_bundle_id=sys.argv[8],
    cache_bundle_digest=sys.argv[9],
    code_sha=sys.argv[10],
    python_executable=python_executable,
)
probe = results_root / ".staging" / task.task_id / "__TRANSITION__"
invocation = runner.build_production_invocation(
    task,
    manifest,
    run_inputs,
    probe,
)
config_hash = runner.derive_config_hash(
    task,
    manifest,
    run_inputs,
    invocation_config=invocation.config,
)
result_dir = results_root / task.output_dir
runner.validate_result(
    result_dir,
    task,
    manifest,
    run_inputs,
    expected_config_hash=config_hash,
    required_artifacts=invocation.required_artifacts,
    required_metrics=invocation.required_metrics,
)
state_path = results_root / "state" / "tasks" / f"{task.task_id}.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
if (
    state.get("state") != "completed"
    or state.get("task_id") != task.task_id
    or state.get("config_hash") != config_hash
    or state.get("returncode") != 0
):
    raise SystemExit("task039 completed state failed transition validation")
print(f"BOUNDARY_RESULT_OK task={task.task_id}")
PY
  then
    fail "task039 full result validation failed"
  fi
fi
log "BOUNDARY_RESULT_OK task=$BOUNDARY_TASK"

[[ ! -e "$forbidden_result" ]] \
  || fail "task040 appeared during task039 validation"
[[ -s "$OLD_FINALIZER_PID_FILE" ]] \
  || fail "old 5090 finalizer PID file is missing"
old_finalizer_pid="$(tr -d '[:space:]' < "$OLD_FINALIZER_PID_FILE")"
[[ "$old_finalizer_pid" =~ ^[0-9]+$ ]] \
  || fail "old finalizer PID is invalid"

if process_alive "$old_finalizer_pid"; then
  old_finalizer_command="$(ps -p "$old_finalizer_pid" -o command= 2>/dev/null || true)"
  [[ "$old_finalizer_command" == *"$OLD_FINALIZER_COMMAND_MARKER"* ]] \
    || fail "refusing to stop unexpected old finalizer command"
  old_finalizer_pgid="$(ps -p "$old_finalizer_pid" -o pgid= | tr -d '[:space:]')"
  [[ "$old_finalizer_pgid" =~ ^[0-9]+$ ]] \
    || fail "cannot read old finalizer process group"
  own_pgid="$(ps -p "$$" -o pgid= | tr -d '[:space:]')"
  [[ "$old_finalizer_pgid" != "$own_pgid" ]] \
    || fail "refusing to stop transition process group"
  kill -TERM -- "-$old_finalizer_pgid" 2>/dev/null \
    || fail "cannot signal old finalizer process group"
  stop_deadline="$($PYTHON - "$STOP_WAIT_SECONDS" <<'PY'
import sys
import time
print(time.monotonic() + float(sys.argv[1]))
PY
)"
  while process_alive "$old_finalizer_pid"; do
    stop_expired="$($PYTHON - "$stop_deadline" <<'PY'
import sys
import time
print(int(time.monotonic() >= float(sys.argv[1])))
PY
)"
    [[ "$stop_expired" == "0" ]] || break
    sleep 0.1
  done
  if process_alive "$old_finalizer_pid"; then
    kill -KILL -- "-$old_finalizer_pgid" 2>/dev/null \
      || fail "cannot kill old finalizer process group"
    sleep 0.1
  fi
fi
process_alive "$old_finalizer_pid" \
  && fail "old 5090 finalizer remains alive pid=$old_finalizer_pid"
log "OLD_PROCESSES_STOPPED runner=$old_runner_pid finalizer=$old_finalizer_pid"

[[ ! -e "$forbidden_result" ]] \
  || fail "task040 appeared before repartition marker publication"

[[ -s "$LAUNCHER" ]] || fail "shard18 launcher is missing"
[[ -s "$FINALIZER" ]] || fail "shard18 finalizer is missing"
for pid_file in "$SHARD_PID_FILE" "$FINALIZER_PID_FILE"; do
  if [[ -s "$pid_file" ]]; then
    existing_pid="$(tr -d '[:space:]' < "$pid_file")"
    process_alive "$existing_pid" \
      && fail "new shard18 process is already running pid=$existing_pid"
  fi
done

"$PYTHON" - \
  "$READY_MARKER" \
  "$BOUNDARY_STATUS" \
  "$summary" \
  "$task_state" \
  "$BOUNDARY_TASK" \
  "$FORBIDDEN_TASK" \
  "$old_runner_pid" \
  "$old_finalizer_pid" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
state_path = Path(sys.argv[4])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {
    "boundary_status_sha256": digest(status_path),
    "boundary_task": sys.argv[5],
    "forbidden_task": sys.argv[6],
    "old_finalizer_pid": int(sys.argv[8]),
    "old_runner_pid": int(sys.argv[7]),
    "plan": "27_18",
    "schema_version": 1,
    "status": "ready",
    "boundary_summary_sha256": digest(summary_path),
    "boundary_state_sha256": digest(state_path),
}
if marker_path.exists():
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"existing ready marker mismatch: {key}")
else:
    payload = {
        **expected,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
PY

remote_ready="$RELAY_CONTROL_DIR/$READY_NAME"
"$SSH_BIN" \
  "${SSH_OPTIONS[@]}" \
  "$RELAY" \
  "set -eu; \
  control='$RELAY_CONTROL_DIR'; \
  final='$remote_ready'; \
  mkdir -p \"\$control\"; \
  tmp=\"\$control/.${READY_NAME}.\$\$.tmp\"; \
  trap 'rm -f \"\$tmp\"' EXIT; \
  cat > \"\$tmp\"; \
  if [ -e \"\$final\" ]; then \
    test -f \"\$final\"; test ! -L \"\$final\"; \
    cmp \"\$tmp\" \"\$final\"; rm -f \"\$tmp\"; \
  else \
    mv \"\$tmp\" \"\$final\"; \
  fi; \
  sync" < "$READY_MARKER" \
  || fail "relay boundary-ready marker publication failed"
log "BOUNDARY_READY_PUBLISHED destination=$RELAY:$remote_ready"

nohup "$SETSID_BIN" bash "$LAUNCHER" > "$SHARD_LOG" 2>&1 < /dev/null &
shard_pid=$!
printf '%s\n' "$shard_pid" > "$SHARD_PID_FILE"
sleep "$LAUNCH_GRACE_SECONDS"
process_alive "$shard_pid" || {
  tail -n 40 "$SHARD_LOG" | tee -a "$TRANSITION_LOG"
  fail "5090 shard18 runner exited during launch"
}
log "STARTED 5090 shard18 runner pid=$shard_pid"

queue_wait_iterations=0
while true; do
  process_alive "$shard_pid" \
    || fail "5090 shard18 runner exited before writing its queue contract"
  read -r queue_contract queue_state queue_current < <(
    read_new_queue_state
  )
  case "$queue_contract" in
    ready)
      process_alive "$shard_pid" \
        || fail "5090 shard18 runner exited after writing its queue contract"
      log "NEW_QUEUE_OK total=18 state=$queue_state current=$queue_current"
      break
      ;;
    stale)
      ;;
    invalid)
      fail "new 5090 queue contract is invalid state=$queue_state current=$queue_current"
      ;;
    *)
      fail "cannot classify new 5090 queue contract"
      ;;
  esac
  queue_wait_iterations=$((queue_wait_iterations + 1))
  if (( queue_wait_iterations >= QUEUE_MAX_WAIT_ITERATIONS )); then
    fail "new 5090 queue contract wait limit reached"
  fi
  sleep "$QUEUE_POLL_SECONDS"
done

nohup "$SETSID_BIN" bash "$FINALIZER" \
  > "$FINALIZER_CONSOLE" 2>&1 < /dev/null &
finalizer_pid=$!
printf '%s\n' "$finalizer_pid" > "$FINALIZER_PID_FILE"
sleep "$LAUNCH_GRACE_SECONDS"
process_alive "$finalizer_pid" || {
  tail -n 40 "$FINALIZER_CONSOLE" | tee -a "$TRANSITION_LOG"
  fail "5090 shard18 finalizer exited during launch"
}
log "STARTED 5090 shard18 finalizer pid=$finalizer_pid"
log "TRANSITION_SUCCESS plan=27_18"
