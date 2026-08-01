#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/native100_orange}
REPO=${REPO:-/root/autodl-tmp/native100_repo}
CONTROL_ROOT=${CONTROL_ROOT:-$RUN_ROOT/control}
RESULTS_ROOT=${RESULTS_ROOT:-$RUN_ROOT/results}
STATUS_FILE=${STATUS_FILE:-$RESULTS_ROOT/queue_status.json}
EXIT_CODE_FILE=${EXIT_CODE_FILE:-$CONTROL_ROOT/runner.exitcode}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.native100_compare_v1.json}
BASE_MANIFEST=${BASE_MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/distributed_native100_5080_5090_v1.json}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
PGREP_BIN=${PGREP_BIN:-/usr/bin/pgrep}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
WAIT_SECONDS=${WAIT_SECONDS:-60}
WAIT_ONCE=${WAIT_ONCE:-0}
LOG=${LOG:-$CONTROL_ROOT/shutdown_waiter.log}

mkdir -p "$CONTROL_ROOT"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "NATIVE100_SHUTDOWN_BLOCKED $*"
  exit 1
}

log "NATIVE100_SHUTDOWN_WAIT status=$STATUS_FILE"
while true; do
  if [[ -s "$EXIT_CODE_FILE" && ! -L "$EXIT_CODE_FILE" ]]; then
    break
  fi
  if [[ "$WAIT_ONCE" == "1" ]]; then
    fail "queue is not complete"
  fi
  sleep "$WAIT_SECONDS"
done

[[ -x "$PYTHON" ]] || fail "Python runtime is missing"
[[ -x "$PGREP_BIN" ]] || fail "process checker is missing"
[[ -x "$SHUTDOWN_BIN" ]] || fail "shutdown command is missing"

if ! "$PYTHON" - "$REPO" "$MANIFEST" "$BASE_MANIFEST" "$PLAN" \
  "$STATUS_FILE" "$EXIT_CODE_FILE" "$RESULTS_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

repo_name, manifest_name, base_name, plan_name, status_name, exit_name, results_name = (
    sys.argv[1:]
)
repo = Path(repo_name)
sys.path.insert(0, str(repo))

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    validate_completed_task_artifacts,
)
from scripts.reproduction.igstgnn_table4.secure_io import (
    read_verified_regular_file,
    verify_secure_directory,
)

MATRIX_SHA256 = "e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246"
BASE_SHA256 = "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
PLAN_SHA256 = "d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e"
HOST_ID = "native100_rtx5090_orange"
CLAIM_SCOPE = "same_table_native_recipe_100_epoch_supplement"


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path, *, purpose, max_bytes):
    payload = read_verified_regular_file(
        path,
        purpose=purpose,
        max_bytes=max_bytes,
        require_nonempty=True,
    )
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )


manifest_path = Path(manifest_name)
base_path = Path(base_name)
plan_path = Path(plan_name)
status_path = Path(status_name)
exit_path = Path(exit_name)
results_root = verify_secure_directory(
    results_name,
    purpose="Native 100 Orange result root",
)

manifest = load_native100_compare_manifest(
    manifest_path,
    expected_sha256=MATRIX_SHA256,
    base_manifest_path=base_path,
    expected_base_manifest_sha256=BASE_SHA256,
)
plan_bytes = read_verified_regular_file(
    plan_path,
    purpose="Native 100 dual-host plan",
    max_bytes=1024 * 1024,
    require_nonempty=True,
)
if hashlib.sha256(plan_bytes).hexdigest() != PLAN_SHA256:
    raise SystemExit("native100 plan SHA-256 mismatch")
plan = json.loads(
    plan_bytes.decode("utf-8"),
    object_pairs_hook=strict_object,
    parse_constant=lambda item: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {item}")
    ),
)
host = next((item for item in plan.get("hosts", []) if item.get("host_id") == HOST_ID), None)
if host is None:
    raise SystemExit("native100 Orange host is absent from the plan")
selected_ids = host.get("task_ids")
if not isinstance(selected_ids, list) or len(selected_ids) != 6:
    raise SystemExit("native100 Orange plan must select exactly six tasks")

status = read_json(
    status_path,
    purpose="Native 100 Orange queue status",
    max_bytes=1024 * 1024,
)
required = {
    "schema_version": 1,
    "state": "completed",
    "protocol_id": "igstgnn_table4_official_v1",
    "matrix_id": "igstgnn_table4_bist_ndr_native100_18_v1",
    "manifest_sha256": MATRIX_SHA256,
    "base_manifest_sha256": BASE_SHA256,
    "selected_task_count": 6,
    "selected_task_ids": selected_ids,
    "verified_task_count": 6,
    "completed_count": 6,
    "completed": selected_ids,
    "skipped_count": 0,
    "skipped": [],
    "failed_count": 0,
    "failed": [],
    "current": None,
}
for key, expected in required.items():
    if status.get(key) != expected:
        raise SystemExit(f"invalid completed queue field: {key}")
if not isinstance(status.get("finished_at"), str) or not status["finished_at"]:
    raise SystemExit("completed queue is missing finished_at")

exit_payload = read_verified_regular_file(
    exit_path,
    purpose="Native 100 Orange runner exit code",
    max_bytes=32,
    require_nonempty=True,
).decode("ascii")
if exit_payload != "0\n":
    raise SystemExit("native100 runner exit code is not zero")

if {path.name for path in results_root.iterdir()} != {"queue_status.json", "tasks"}:
    raise SystemExit("native100 Orange result root allowlist mismatch")
tasks_root = verify_secure_directory(
    results_root / "tasks",
    purpose="Native 100 Orange tasks root",
)
task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
if {path.name for path in tasks_root.iterdir()} != set(selected_ids):
    raise SystemExit("native100 Orange task directory set mismatch")
for task_id in selected_ids:
    validate_completed_task_artifacts(
        tasks_root / task_id,
        expected_task=task_by_id[task_id],
        expected_claim_scope=CLAIM_SCOPE,
    )
print("NATIVE100_ORANGE_POSTFLIGHT_OK tasks=6", flush=True)
PY
then
  fail "completion postflight failed"
fi

if "$PGREP_BIN" -af 'native100_compare_(matrix_)?runner' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "native100 training process remains active"
fi

sync
log "NATIVE100_SHUTDOWN_SUCCESS shutting_down"
"$SHUTDOWN_BIN" -h now
