#!/usr/bin/env bash
# Verify, transfer, and shut down the protocol-v3 RTX 5090 shard.
set -euo pipefail

REPO=${REPO:-/root/xtraffic_v3_5090_clean}
OPS_ROOT=${OPS_ROOT:-/root/autodl-tmp/xtraffic_ops}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090}
SHARD_PID_FILE=${SHARD_PID_FILE:-/root/autodl-tmp/protocol_v3_5090_shard.pid}
SHARD_EXIT_CODE=${SHARD_EXIT_CODE:-/root/autodl-tmp/protocol_v3_5090_shard.exitcode}
POSTFLIGHT=${POSTFLIGHT:-$OPS_ROOT/scripts/remote/distributed/protocol_v3_shard_postflight.py}
POSTFLIGHT_CMD=${POSTFLIGHT_CMD:-}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-/root/autodl-tmp/protocol_v3_5090_shard_postflight.json}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-/root/autodl-tmp/protocol_v3_5090_shard_transfer.sha256}
TRANSFER_FILE_LIST=${TRANSFER_FILE_LIST:-/root/autodl-tmp/protocol_v3_5090_shard_files.txt}
TRANSFER_MARKER=${TRANSFER_MARKER:-/root/autodl-tmp/protocol_v3_5090_shard_complete.json}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/shards}
REMOTE_NAME=${REMOTE_NAME:-rtx5090_21}
SPACE_MARGIN_BYTES=${SPACE_MARGIN_BYTES:-2147483648}
POLL_SECONDS=${POLL_SECONDS:-60}
SPACE_RETRY_SECONDS=${SPACE_RETRY_SECONDS:-300}
FINALIZER_ONCE=${FINALIZER_ONCE:-0}
SHARD_PROCESS_REQUIRED=${SHARD_PROCESS_REQUIRED:-1}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
RSYNC_BIN=${RSYNC_BIN:-/usr/bin/rsync}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
LOG=${LOG:-/root/autodl-tmp/protocol_v3_5090_shard_finalize.log}

remote_incoming="$RELAY_BASE/.${REMOTE_NAME}.incoming"
remote_final="$RELAY_BASE/$REMOTE_NAME"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "FINALIZE_BLOCKED $*"
  exit 1
}

[[ "$RELAY_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || fail "unsafe RELAY_BASE"
[[ "$REMOTE_NAME" =~ ^[A-Za-z0-9._-]+$ ]] \
  || fail "unsafe REMOTE_NAME"
[[ -d "$RESULTS_ROOT" ]] \
  || fail "results root missing: $RESULTS_ROOT"
[[ -f "$RESULTS_ROOT/queue_status.json" ]] \
  || fail "queue status missing"

read_queue() {
  "$PYTHON" - "$RESULTS_ROOT/queue_status.json" <<'PY'
import json
import sys
from pathlib import Path

queue = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    queue.get("state"),
    queue.get("total_tasks"),
    queue.get("completed_tasks"),
    queue.get("skipped_valid_tasks"),
)
PY
}

shard_alive() {
  [[ -s "$SHARD_PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$SHARD_PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" \
    | grep -Fq "protocol_v3_launch_5090_shard.sh"
}

while true; do
  read -r queue_state queue_total queue_completed queue_skipped < <(
    read_queue
  )
  [[ "$queue_total" =~ ^[0-9]+$ ]] \
    || fail "queue total_tasks is not an integer"
  [[ "$queue_completed" =~ ^[0-9]+$ ]] \
    || fail "queue completed_tasks is not an integer"
  [[ "$queue_skipped" =~ ^[0-9]+$ ]] \
    || fail "queue skipped_valid_tasks is not an integer"
  if [[
    "$queue_state" == "completed"
    && "$queue_total" == "21"
    && "$queue_completed" == "21"
    && "$queue_skipped" -ge 2
    && "$queue_skipped" -le 21
  ]]; then
    break
  fi
  case "$queue_state" in
    preflight|running)
      ;;
    failed|interrupted)
      fail "queue entered terminal state $queue_state"
      ;;
    *)
      fail "unexpected queue state $queue_state"
      ;;
  esac
  if [[ "$SHARD_PROCESS_REQUIRED" == "1" ]]; then
    shard_alive || fail "5090 shard exited before completion"
  fi
  if [[ "$FINALIZER_ONCE" == "1" ]]; then
    fail "queue state is $queue_state with ${queue_completed}/${queue_total}"
  fi
  log "WAIT queue_state=$queue_state completed=${queue_completed}/${queue_total}"
  sleep "$POLL_SECONDS"
done

if [[ "$SHARD_PROCESS_REQUIRED" == "1" ]]; then
  for _ in $(seq 1 300); do
    shard_alive || break
    sleep 1
  done
  shard_alive && fail "5090 shard process did not exit after queue completion"
  [[ -s "$SHARD_EXIT_CODE" ]] || fail "shard exit code is missing"
  [[ "$(tr -d '[:space:]' < "$SHARD_EXIT_CODE")" == "0" ]] \
    || fail "shard runner exit code is not zero"
fi

export XTRAFFIC_REPO_ROOT="$REPO"
export PYTHONPATH="$OPS_ROOT:$REPO"
log "POSTFLIGHT_START"
if [[ -n "$POSTFLIGHT_CMD" ]]; then
  export POSTFLIGHT_REPORT
  bash -lc "$POSTFLIGHT_CMD" >> "$LOG" 2>&1 \
    || fail "postflight validation failed"
else
  "$PYTHON" -X utf8 "$POSTFLIGHT" \
    --shard 5090 \
    --manifest "$REPO/scripts/remote/windows/protocol_v3_manifest.json" \
    --repo-root "$REPO" \
    --results-root "$RESULTS_ROOT" \
    --cache-dir "$REPO/outputs/dist_net/region_data_v3_noleak_dedup" \
    --graph-dir "$REPO/outputs/region_graphs" \
    --cache-bundle-id "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4" \
    --cache-bundle-digest \
      "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5" \
    --code-sha "3a4952837aaf6fbda1403d50b610b1bf40c6098a" \
    --python "$PYTHON" \
    --report "$POSTFLIGHT_REPORT" >> "$LOG" 2>&1 \
    || fail "postflight validation failed"
fi
[[ -s "$POSTFLIGHT_REPORT" ]] || fail "postflight report is missing"
log "POSTFLIGHT_OK"

"$PYTHON" - \
  "$POSTFLIGHT_REPORT" \
  "$TRANSFER_FILE_LIST" \
  "$TRANSFER_MANIFEST" \
  "$TRANSFER_MARKER" \
  "$RESULTS_ROOT" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

report_path = Path(sys.argv[1])
file_list_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
marker_path = Path(sys.argv[4])
results_root = Path(sys.argv[5]).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "verified" or report.get("shard") != "5090":
    raise SystemExit("invalid postflight report identity")
if report.get("validated_tasks") != 21:
    raise SystemExit("invalid postflight validated_tasks")
if report.get("expected_skipped_valid_tasks") != 2:
    raise SystemExit("invalid inherited postflight skip count")
if report.get("minimum_skipped_valid_tasks") != 2:
    raise SystemExit("invalid minimum postflight skip count")
actual_skips = report.get("actual_skipped_valid_tasks")
if (
    isinstance(actual_skips, bool)
    or not isinstance(actual_skips, int)
    or not 2 <= actual_skips <= 21
):
    raise SystemExit("invalid actual postflight skip count")
expected_skip_ids = report.get("expected_skipped_task_ids")
actual_skip_ids = report.get("actual_skipped_task_ids")
if (
    not isinstance(expected_skip_ids, list)
    or len(expected_skip_ids) != 2
    or not isinstance(actual_skip_ids, list)
    or len(actual_skip_ids) != actual_skips
    or not set(expected_skip_ids).issubset(actual_skip_ids)
):
    raise SystemExit("invalid postflight skipped task identity")
tasks = report.get("tasks")
if not isinstance(tasks, list) or len(tasks) != 21:
    raise SystemExit("invalid postflight task records")
states = {"completed": 0, "skipped_valid": 0}
task_skip_ids = []
for task in tasks:
    if not isinstance(task, dict):
        raise SystemExit("invalid postflight task record")
    state = task.get("state")
    task_id = task.get("task_id")
    if state not in states or not isinstance(task_id, str):
        raise SystemExit("invalid postflight task state")
    states[state] += 1
    if state == "skipped_valid":
        task_skip_ids.append(task_id)
if sorted(task_skip_ids) != sorted(actual_skip_ids):
    raise SystemExit("postflight skipped task list differs from task states")
if report.get("task_state_counts") != states:
    raise SystemExit("postflight task state counts differ from task records")
queue = report.get("queue_status")
if (
    not isinstance(queue, dict)
    or queue.get("state") != "completed"
    or queue.get("total_tasks") != 21
    or queue.get("completed_tasks") != 21
    or queue.get("skipped_valid_tasks") != actual_skips
):
    raise SystemExit("invalid postflight queue summary")
if (
    report.get("verified_at") != queue.get("updated_at")
    or report.get("source_completed_at") != queue.get("updated_at")
):
    raise SystemExit("postflight completion timestamp is not source-stable")
records = report.get("files")
export_paths = report.get("export_paths")
if not isinstance(records, list) or not isinstance(export_paths, list):
    raise SystemExit("postflight report lacks export records")
by_path = {}
total_bytes = 0
for record in records:
    if not isinstance(record, dict):
        raise SystemExit("invalid postflight file record")
    relative = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise SystemExit("invalid postflight path, digest, or size")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != relative
        or relative in by_path
    ):
        raise SystemExit(f"unsafe export path: {relative!r}")
    source = (results_root / Path(*pure.parts)).resolve()
    try:
        source.relative_to(results_root)
    except ValueError as exc:
        raise SystemExit(f"export path escapes results root: {relative}") from exc
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"export source is missing or unsafe: {relative}")
    if source.stat().st_size != size:
        raise SystemExit(f"export source size changed: {relative}")
    actual_digest = sha256_file(source)
    if actual_digest != digest:
        raise SystemExit(f"export source hash changed: {relative}")
    by_path[relative] = digest
    total_bytes += size
ordered = sorted(by_path)
if export_paths != ordered:
    raise SystemExit("export path set differs from hashed file records")
if report.get("file_count") != len(records):
    raise SystemExit("postflight file_count differs from file records")
if report.get("total_bytes") != total_bytes:
    raise SystemExit("postflight total_bytes differs from file records")
files_digest = hashlib.sha256(
    json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
if report.get("files_digest") != files_digest:
    raise SystemExit("postflight files_digest differs from file records")
for destination in (
    file_list_path,
    manifest_path,
    marker_path,
):
    destination.parent.mkdir(parents=True, exist_ok=True)
file_tmp = file_list_path.with_suffix(file_list_path.suffix + ".tmp")
manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
file_tmp.write_text("".join(f"{path}\n" for path in ordered), encoding="utf-8")
manifest_tmp.write_text(
    "".join(f"{by_path[path]}  {path}\n" for path in ordered),
    encoding="utf-8",
)
os.replace(file_tmp, file_list_path)
os.replace(manifest_tmp, manifest_path)
manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
file_list_digest = hashlib.sha256(file_list_path.read_bytes()).hexdigest()
report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
marker = {
    "actual_skipped_valid_tasks": actual_skips,
    "file_count": len(records),
    "file_list_sha256": file_list_digest,
    "manifest_sha256": manifest_digest,
    "minimum_skipped_valid_tasks": 2,
    "postflight_files_digest": files_digest,
    "postflight_sha256": report_digest,
    "schema_version": 1,
    "shard": "5090",
    "source_bytes": total_bytes,
    "status": "verified",
    "validated_tasks": 21,
}
marker_tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
marker_tmp.write_text(
    json.dumps(
        marker,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(marker_tmp, marker_path)
PY

[[ -s "$TRANSFER_FILE_LIST" ]] || fail "transfer file list is empty"
[[ -s "$TRANSFER_MANIFEST" ]] || fail "transfer manifest is empty"
[[ -s "$TRANSFER_MARKER" ]] || fail "transfer completion marker is empty"
source_bytes="$(
  "$PYTHON" - "$POSTFLIGHT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(report["total_bytes"])
PY
)"
file_count="$(wc -l < "$TRANSFER_MANIFEST" | tr -d ' ')"
manifest_digest="$(sha256sum "$TRANSFER_MANIFEST" | awk '{print $1}')"
file_list_digest="$(sha256sum "$TRANSFER_FILE_LIST" | awk '{print $1}')"
report_digest="$(sha256sum "$POSTFLIGHT_REPORT" | awk '{print $1}')"
marker_digest="$(sha256sum "$TRANSFER_MARKER" | awk '{print $1}')"

verify_remote_tree() {
  local root=$1
  local marker_name=$2
  "$SSH_BIN" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 \
    "$RELAY" \
    "set -eu; \
    root='$root'; marker_name='$marker_name'; \
    test -d \"\$root\"; test ! -L \"\$root\"; \
    test -d \"\$root/results\"; test ! -L \"\$root/results\"; \
    test -d \"\$root/metadata\"; test ! -L \"\$root/metadata\"; \
    test -f \"\$root/\$marker_name\"; test ! -L \"\$root/\$marker_name\"; \
    test \"\$(sha256sum \"\$root/\$marker_name\" | awk '{print \$1}')\" \
      = '$marker_digest'; \
    test \"\$(sha256sum \"\$root/metadata/transfer.sha256\" | awk '{print \$1}')\" \
      = '$manifest_digest'; \
    test \"\$(sha256sum \"\$root/metadata/transfer.files\" | awk '{print \$1}')\" \
      = '$file_list_digest'; \
    test \"\$(sha256sum \"\$root/metadata/postflight.json\" | awk '{print \$1}')\" \
      = '$report_digest'; \
    test -z \"\$(find \"\$root\" -type l -print -quit)\"; \
    test -z \"\$(find \"\$root\" ! -type f ! -type d -print -quit)\"; \
    tmp=\$(mktemp -d); trap 'rm -rf \"\$tmp\"' EXIT; \
    printf '%s\\n' '_TRANSFER_COMPLETE.json' metadata results \
      | LC_ALL=C sort > \"\$tmp/expected.top\"; \
    find \"\$root\" -mindepth 1 -maxdepth 1 -printf '%f\\n' \
      | sed \"s/^\$marker_name$/_TRANSFER_COMPLETE.json/\" \
      | LC_ALL=C sort > \"\$tmp/actual.top\"; \
    cmp \"\$tmp/actual.top\" \"\$tmp/expected.top\"; \
    printf '%s\\n' postflight.json transfer.files transfer.sha256 \
      | LC_ALL=C sort > \"\$tmp/expected.metadata\"; \
    find \"\$root/metadata\" -mindepth 1 -maxdepth 1 -type f -printf '%f\\n' \
      | LC_ALL=C sort > \"\$tmp/actual.metadata\"; \
    cmp \"\$tmp/actual.metadata\" \"\$tmp/expected.metadata\"; \
    test -z \"\$(find \"\$root/metadata\" -mindepth 2 -print -quit)\"; \
    cd \"\$root/results\"; \
    sha256sum -c ../metadata/transfer.sha256 >/dev/null; \
    cut -c67- ../metadata/transfer.sha256 \
      | LC_ALL=C sort > \"\$tmp/manifest.files\"; \
    LC_ALL=C sort ../metadata/transfer.files > \"\$tmp/expected.files\"; \
    cmp \"\$tmp/manifest.files\" \"\$tmp/expected.files\"; \
    find . -type f -printf '%P\\n' \
      | LC_ALL=C sort > \"\$tmp/actual.files\"; \
    cmp \"\$tmp/actual.files\" \"\$tmp/expected.files\"; \
    awk -F/ '{ path=\"\"; for (i=1; i<NF; i++) { \
      path=(path ? path \"/\" : \"\") \$i; print path \
    }}' ../metadata/transfer.files \
      | LC_ALL=C sort -u > \"\$tmp/expected.dirs\"; \
    find . -mindepth 1 -type d -printf '%P\\n' \
      | LC_ALL=C sort > \"\$tmp/actual.dirs\"; \
    cmp \"\$tmp/actual.dirs\" \"\$tmp/expected.dirs\""
}

while true; do
  read -r remote_free incoming_bytes final_exists < <(
    "$SSH_BIN" \
      -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=15 \
      "$RELAY" \
      "set -eu; mkdir -p '$RELAY_BASE'; \
      free=\$(df -PB1 '$RELAY_BASE' | awk 'NR==2 {print \$4}'); \
      incoming=0; \
      if [ -d '$remote_incoming' ]; then \
        incoming=\$(du -sb '$remote_incoming' | awk '{print \$1}'); \
      fi; \
      final_exists=0; \
      if [ -e '$remote_final' ]; then final_exists=1; fi; \
      printf '%s %s %s\\n' \"\$free\" \"\$incoming\" \"\$final_exists\""
  )
  [[ "$remote_free" =~ ^[0-9]+$ ]] \
    || fail "cannot read relay free space"
  [[ "$incoming_bytes" =~ ^[0-9]+$ ]] \
    || fail "cannot read relay incoming size"
  [[ "$final_exists" =~ ^[01]$ ]] \
    || fail "cannot read relay final destination state"
  if [[ "$final_exists" == "1" ]]; then
    if verify_remote_tree "$remote_final" "_TRANSFER_COMPLETE.json"; then
      log "REMOTE_ALREADY_VERIFIED destination=$RELAY:$remote_final"
      sync
      log "FINALIZE_SUCCESS shutting_down"
      "$SHUTDOWN_BIN" -h now
      exit 0
    fi
    fail "relay destination exists but full verification failed"
  fi
  available_capacity=$((remote_free + incoming_bytes))
  required_capacity=$((source_bytes + SPACE_MARGIN_BYTES))
  if (( available_capacity >= required_capacity )); then
    break
  fi
  if [[ "$FINALIZER_ONCE" == "1" ]]; then
    fail "relay capacity $available_capacity is below required $required_capacity"
  fi
  log "WAIT_SPACE available=$available_capacity required=$required_capacity"
  sleep "$SPACE_RETRY_SECONDS"
done

log "TRANSFER_START bytes=$source_bytes files=$file_count"
"$SSH_BIN" \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  "$RELAY" \
  "set -eu; \
  rm -rf '$remote_incoming'; \
  mkdir -p '$remote_incoming/results' '$remote_incoming/metadata'"

"$RSYNC_BIN" \
  --archive \
  --relative \
  --partial \
  --human-readable \
  --info=stats2 \
  --files-from="$TRANSFER_FILE_LIST" \
  -e "$SSH_BIN -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$RESULTS_ROOT/" \
  "$RELAY:$remote_incoming/results/" \
  >> "$LOG" 2>&1 \
  || fail "result rsync failed"

"$RSYNC_BIN" \
  --archive \
  -e "$SSH_BIN -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$TRANSFER_MANIFEST" \
  "$RELAY:$remote_incoming/metadata/transfer.sha256" \
  >> "$LOG" 2>&1 \
  || fail "manifest rsync failed"
"$RSYNC_BIN" \
  --archive \
  -e "$SSH_BIN -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$TRANSFER_FILE_LIST" \
  "$RELAY:$remote_incoming/metadata/transfer.files" \
  >> "$LOG" 2>&1 \
  || fail "file list rsync failed"
"$RSYNC_BIN" \
  --archive \
  -e "$SSH_BIN -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$POSTFLIGHT_REPORT" \
  "$RELAY:$remote_incoming/metadata/postflight.json" \
  >> "$LOG" 2>&1 \
  || fail "postflight report rsync failed"
"$RSYNC_BIN" \
  --archive \
  -e "$SSH_BIN -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$TRANSFER_MARKER" \
  "$RELAY:$remote_incoming/_TRANSFER_COMPLETE.json.tmp" \
  >> "$LOG" 2>&1 \
  || fail "completion marker rsync failed"

log "REMOTE_VERIFY_START manifest=$manifest_digest"
verify_remote_tree "$remote_incoming" "_TRANSFER_COMPLETE.json.tmp" \
  || fail "remote exact tree or hash verification failed"
"$SSH_BIN" \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  "$RELAY" \
  "set -eu; \
  test ! -e '$remote_final'; \
  mv '$remote_incoming/_TRANSFER_COMPLETE.json.tmp' \
    '$remote_incoming/_TRANSFER_COMPLETE.json'; \
  sync; \
  mv '$remote_incoming' '$remote_final'; \
  sync" \
  || fail "remote atomic promotion failed"
verify_remote_tree "$remote_final" "_TRANSFER_COMPLETE.json" \
  || fail "promoted remote tree failed full verification"

log "REMOTE_VERIFY_OK destination=$RELAY:$remote_final"
sync
log "FINALIZE_SUCCESS shutting_down"
"$SHUTDOWN_BIN" -h now
