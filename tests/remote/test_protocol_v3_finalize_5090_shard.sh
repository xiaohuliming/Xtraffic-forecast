#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
finalizer="$repo_root/scripts/remote/linux/protocol_v3_finalize_5090_shard.sh"
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

bin_root="$tmp_root/bin"
mkdir -p "$bin_root"

cat > "$bin_root/ssh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ssh %s\n' "$*" >> "$ACTION_LOG"
if [[ "$*" == *"df -PB1"* ]]; then
  printf '%s %s %s\n' \
    "${FAKE_REMOTE_FREE:-1000000000}" \
    "${FAKE_REMOTE_INCOMING_BYTES:-0}" \
    "${FAKE_REMOTE_FINAL_EXISTS:-0}"
elif [[ "$*" == *"root='"*"marker_name='"* ]]; then
  [[ "$*" == *"sha256sum -c ../metadata/transfer.sha256"* ]]
  [[ "$*" == *'cmp "$tmp/actual.files" "$tmp/expected.files"'* ]]
  [[ "$*" == *'cmp "$tmp/actual.dirs" "$tmp/expected.dirs"'* ]]
  [[ "$*" == *'postflight.json transfer.files transfer.sha256'* ]]
  exit "${FAKE_VERIFY_RC:-0}"
fi
SH
chmod +x "$bin_root/ssh"

cat > "$bin_root/rsync" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'rsync %s\n' "$*" >> "$ACTION_LOG"
exit "${FAKE_RSYNC_RC:-0}"
SH
chmod +x "$bin_root/rsync"

cat > "$bin_root/shutdown" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'shutdown %s\n' "$*" >> "$ACTION_LOG"
touch "$SHUTDOWN_MARKER"
SH
chmod +x "$bin_root/shutdown"

cat > "$tmp_root/postflight.py" <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RESULTS_ROOT"])
report = Path(os.environ["POSTFLIGHT_REPORT"])
paths = ["queue_status.json", "tasks/example/summary.json"]
records = []
for relative in paths:
    path = root / relative
    records.append(
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
queue = json.loads((root / "queue_status.json").read_text(encoding="utf-8"))
actual_skips = queue["skipped_valid_tasks"]
task_numbers = [
    5,
    6,
    11,
    12,
    17,
    18,
    *range(31, 40),
    *range(43, 46),
]
task_ids = [f"task{number:03d}" for number in task_numbers]
expected_skip_ids = task_ids[:15]
actual_skip_ids = task_ids[:actual_skips]
tasks = [
    {
        "task_id": task_id,
        "state": (
            "skipped_valid"
            if task_id in actual_skip_ids
            else "completed"
        ),
    }
    for task_id in task_ids
]
files_digest = hashlib.sha256(
    json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
report.write_text(
    json.dumps(
        {
            "status": "verified",
            "shard": "5090",
            "verified_at": queue["updated_at"],
            "source_completed_at": queue["updated_at"],
            "validated_tasks": 18,
            "expected_skipped_valid_tasks": 15,
            "minimum_skipped_valid_tasks": 15,
            "actual_skipped_valid_tasks": actual_skips,
            "expected_skipped_task_ids": expected_skip_ids,
            "actual_skipped_task_ids": actual_skip_ids,
            "task_state_counts": {
                "completed": 18 - actual_skips,
                "skipped_valid": actual_skips,
            },
            "queue_status": queue,
            "tasks": tasks,
            "files": records,
            "export_paths": paths,
            "file_count": len(records),
            "total_bytes": sum(item["size_bytes"] for item in records),
            "files_digest": files_digest,
        }
    ),
    encoding="utf-8",
)
PY

make_case() {
  local name=$1
  local queue_state=$2
  local queue_total=$3
  local queue_completed=$4
  local queue_skipped=$5
  local root="$tmp_root/$name"
  mkdir -p "$root/results/tasks/example"
  printf '%s\n' "validated" > "$root/results/tasks/example/summary.json"
  printf '%s\n' \
    "{\"state\":\"$queue_state\",\"updated_at\":\"2026-07-20T12:00:00Z\",\"total_tasks\":$queue_total,\"completed_tasks\":$queue_completed,\"skipped_valid_tasks\":$queue_skipped,\"current_task\":null}" \
    > "$root/results/queue_status.json"
}

run_finalizer() {
  local name=$1
  local root="$tmp_root/$name"
  export RESULTS_ROOT="$root/results"
  export POSTFLIGHT_REPORT="$root/postflight.json"
  export ACTION_LOG="$root/actions.log"
  export SHUTDOWN_MARKER="$root/shutdown.called"
  SHARD_PROCESS_REQUIRED=0 \
  FINALIZER_ONCE=1 \
  POSTFLIGHT_CMD="python3 '$tmp_root/postflight.py'" \
  POSTFLIGHT_REPORT="$POSTFLIGHT_REPORT" \
  TRANSFER_FILE_LIST="$root/files.txt" \
  TRANSFER_MANIFEST="$root/transfer.sha256" \
  TRANSFER_MARKER="$root/complete.json" \
  SSH_BIN="$bin_root/ssh" \
  RSYNC_BIN="$bin_root/rsync" \
  SHUTDOWN_BIN="$bin_root/shutdown" \
  PYTHON=python3 \
  SPACE_MARGIN_BYTES=1 \
  RELAY=root@relay.test \
  RELAY_BASE=/srv/results/shards \
  LOG="$root/finalizer.log" \
  bash "$finalizer"
}

make_case success completed 18 18 15
run_finalizer success
test -f "$tmp_root/success/shutdown.called"
grep -q '^shutdown -h now$' "$tmp_root/success/actions.log"
grep -q -- '--files-from=' "$tmp_root/success/actions.log"
grep -q "rm -rf '/srv/results/shards/.rtx5090_18.incoming'" \
  "$tmp_root/success/actions.log"
grep -q 'FINALIZE_SUCCESS' "$tmp_root/success/finalizer.log"
while IFS= read -r action; do
  case "$action" in
    ssh\ *|rsync\ *)
      [[ "$action" == *"ServerAliveInterval=5"* ]]
      [[ "$action" == *"ServerAliveCountMax=3"* ]]
      ;;
  esac
done < "$tmp_root/success/actions.log"
python3 - "$tmp_root/success/complete.json" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert marker["schema_version"] == 1
assert marker["status"] == "verified"
assert marker["shard"] == "5090"
assert marker["validated_tasks"] == 18
assert marker["minimum_skipped_valid_tasks"] == 15
assert marker["actual_skipped_valid_tasks"] == 15
assert len(marker["manifest_sha256"]) == 64
assert len(marker["file_list_sha256"]) == 64
assert len(marker["postflight_sha256"]) == 64
assert len(marker["postflight_files_digest"]) == 64
PY
printf '%s\n' "PASS verified shard transfers before shutdown"
printf '%s\n' "PASS every relay transport has keepalive failure bounds"

make_case safe_restart completed 18 18 17
run_finalizer safe_restart
test -f "$tmp_root/safe_restart/shutdown.called"
grep -q 'FINALIZE_SUCCESS' "$tmp_root/safe_restart/finalizer.log"
printf '%s\n' "PASS safe restart may increase skipped-valid count"

make_case verified_reentry completed 18 18 17
FAKE_REMOTE_FINAL_EXISTS=1 run_finalizer verified_reentry
test -f "$tmp_root/verified_reentry/shutdown.called"
test "$(grep -c '^rsync ' "$tmp_root/verified_reentry/actions.log" || true)" -eq 0
grep -q 'REMOTE_ALREADY_VERIFIED' \
  "$tmp_root/verified_reentry/finalizer.log"
printf '%s\n' "PASS existing destination receives full re-verification"

make_case conflicting_reentry completed 18 18 15
set +e
FAKE_REMOTE_FINAL_EXISTS=1 FAKE_VERIFY_RC=8 \
  run_finalizer conflicting_reentry
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/conflicting_reentry/shutdown.called"
grep -q 'full verification failed' \
  "$tmp_root/conflicting_reentry/finalizer.log"
printf '%s\n' "PASS invalid existing destination remains fail-closed"

make_case incomplete running 18 15 15
set +e
run_finalizer incomplete
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/incomplete/shutdown.called"
grep -q 'FINALIZE_BLOCKED queue state is running' \
  "$tmp_root/incomplete/finalizer.log"
printf '%s\n' "PASS incomplete shard remains on"

make_case postflight_failure completed 18 18 15
export RESULTS_ROOT="$tmp_root/postflight_failure/results"
export POSTFLIGHT_REPORT="$tmp_root/postflight_failure/postflight.json"
export ACTION_LOG="$tmp_root/postflight_failure/actions.log"
export SHUTDOWN_MARKER="$tmp_root/postflight_failure/shutdown.called"
set +e
SHARD_PROCESS_REQUIRED=0 \
FINALIZER_ONCE=1 \
POSTFLIGHT_CMD=false \
POSTFLIGHT_REPORT="$POSTFLIGHT_REPORT" \
SSH_BIN="$bin_root/ssh" \
RSYNC_BIN="$bin_root/rsync" \
SHUTDOWN_BIN="$bin_root/shutdown" \
PYTHON=python3 \
LOG="$tmp_root/postflight_failure/finalizer.log" \
bash "$finalizer"
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/postflight_failure/shutdown.called"
grep -q 'FINALIZE_BLOCKED postflight validation failed' \
  "$tmp_root/postflight_failure/finalizer.log"
printf '%s\n' "PASS postflight failure remains on"

make_case insufficient_space completed 18 18 15
set +e
FAKE_REMOTE_FREE=1 run_finalizer insufficient_space
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/insufficient_space/shutdown.called"
grep -q 'FINALIZE_BLOCKED relay capacity' \
  "$tmp_root/insufficient_space/finalizer.log"
printf '%s\n' "PASS insufficient relay space remains on"

make_case transfer_failure completed 18 18 15
set +e
FAKE_RSYNC_RC=9 run_finalizer transfer_failure
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/transfer_failure/shutdown.called"
grep -q 'FINALIZE_BLOCKED result rsync failed' \
  "$tmp_root/transfer_failure/finalizer.log"
printf '%s\n' "PASS transfer failure remains on"

make_case verify_failure completed 18 18 15
set +e
FAKE_VERIFY_RC=8 run_finalizer verify_failure
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/verify_failure/shutdown.called"
grep -q 'FINALIZE_BLOCKED remote exact tree or hash verification failed' \
  "$tmp_root/verify_failure/finalizer.log"
printf '%s\n' "PASS remote hash failure remains on"
