#!/usr/bin/env bash
# Wait for the native-100 Orange shard, publish it to the relay, then halt.
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/native100_orange}
REPO=${REPO:-/root/autodl-tmp/native100_repo}
CONTROL_ROOT=${CONTROL_ROOT:-$RUN_ROOT/control}
RESULTS_ROOT=${RESULTS_ROOT:-$RUN_ROOT/results}
EXIT_CODE_FILE=${EXIT_CODE_FILE:-$CONTROL_ROOT/runner.exitcode}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.native100_compare_v1.json}
BASE_MANIFEST=${BASE_MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
PLAN=${PLAN:-/root/autodl-tmp/native100_control/distributed_native100_5080_5090_v1.json}
VALIDATOR=${VALIDATOR:-/root/autodl-tmp/native100_control/igstgnn_table4_native100_5090_shutdown_waiter.sh}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
PGREP_BIN=${PGREP_BIN:-/usr/bin/pgrep}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
RSYNC_BIN=${RSYNC_BIN:-/usr/bin/rsync}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
REALPATH_BIN=${REALPATH_BIN:-/usr/bin/realpath}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-/root/.ssh/id_ed25519_xtraffic_native100}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/igstgnn_table4_official_v1/native100_5080_5090_v1/hosts}
HOST_ID=${HOST_ID:-native100_rtx5090_orange}
WAIT_SECONDS=${WAIT_SECONDS:-60}
WAIT_ONCE=${WAIT_ONCE:-0}
RESERVE_BYTES=${RESERVE_BYTES:-134217728}
EXPORT_ROOT=${EXPORT_ROOT:-$CONTROL_ROOT/export}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-$EXPORT_ROOT/postflight.json}
TRANSFER_FILE_LIST=${TRANSFER_FILE_LIST:-$EXPORT_ROOT/transfer.files}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-$EXPORT_ROOT/transfer.sha256}
TRANSFER_MARKER=${TRANSFER_MARKER:-$EXPORT_ROOT/_TRANSFER_COMPLETE.json}
LOG=${LOG:-$CONTROL_ROOT/finalize.log}

MATRIX_SHA256=e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246
BASE_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78
PLAN_SHA256=d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e
CLAIM_SCOPE=same_table_native_recipe_100_epoch_supplement

mkdir -p "$CONTROL_ROOT"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "NATIVE100_FINALIZE_BLOCKED $*"
  exit 1
}

[[ "$HOST_ID" == "native100_rtx5090_orange" ]] || fail "invalid host ID"
[[ "$RELAY_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "unsafe relay base"
for executable in "$PGREP_BIN" "$SSH_BIN" "$RSYNC_BIN" "$SHUTDOWN_BIN" "$REALPATH_BIN"; do
  [[ -x "$executable" && ! -L "$executable" ]] || fail "required executable is missing or unsafe"
done
resolved_python=$("$REALPATH_BIN" "$PYTHON")
[[ -f "$resolved_python" && ! -L "$resolved_python" && -x "$resolved_python" ]] \
  || fail "resolved Python runtime is missing or unsafe"
case "$resolved_python" in
  /root/miniconda3/bin/python*) ;;
  *) fail "resolved Python runtime escapes the bound environment" ;;
esac
PYTHON=$resolved_python
for source in "$MANIFEST" "$BASE_MANIFEST" "$PLAN" "$VALIDATOR"; do
  [[ -s "$source" && ! -L "$source" ]] || fail "bound control file is missing or unsafe"
done
[[ -s "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
  || fail "relay identity is missing or unsafe"
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MATRIX_SHA256" ]] \
  || fail "matrix manifest hash mismatch"
[[ "$(sha256sum "$BASE_MANIFEST" | awk '{print $1}')" == "$BASE_SHA256" ]] \
  || fail "base manifest hash mismatch"
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] \
  || fail "dual-host plan hash mismatch"

log "NATIVE100_FINALIZE_WAIT exit_code=$EXIT_CODE_FILE"
while true; do
  if [[ -s "$EXIT_CODE_FILE" && ! -L "$EXIT_CODE_FILE" ]]; then
    break
  fi
  if [[ "$WAIT_ONCE" == "1" ]]; then
    fail "queue is not complete"
  fi
  sleep "$WAIT_SECONDS"
done

# Reuse the separately tested completion validator, but replace shutdown with true.
RUN_ROOT="$RUN_ROOT" \
REPO="$REPO" \
PLAN="$PLAN" \
PYTHON="$PYTHON" \
PGREP_BIN="$PGREP_BIN" \
SHUTDOWN_BIN=/usr/bin/true \
WAIT_ONCE=1 \
bash "$VALIDATOR" >>"$LOG" 2>&1 \
  || fail "local completion postflight failed"
log "NATIVE100_LOCAL_POSTFLIGHT_OK tasks=6"

rm -rf "$EXPORT_ROOT"
mkdir -p "$EXPORT_ROOT"
"$PYTHON" - "$RESULTS_ROOT" "$POSTFLIGHT_REPORT" "$TRANSFER_FILE_LIST" \
  "$TRANSFER_MANIFEST" "$TRANSFER_MARKER" "$MATRIX_SHA256" "$BASE_SHA256" \
  "$PLAN_SHA256" "$HOST_ID" "$CLAIM_SCOPE" <<'PY'
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    results_name,
    report_name,
    list_name,
    manifest_name,
    marker_name,
    matrix_sha256,
    base_sha256,
    plan_sha256,
    host_id,
    claim_scope,
) = sys.argv[1:]
results = Path(results_name).resolve(strict=True)
report_path = Path(report_name)
file_list_path = Path(list_name)
transfer_manifest_path = Path(manifest_name)
marker_path = Path(marker_name)
selected_ids = [
    "n100007_bist_orange_seed42",
    "n100008_bist_orange_seed1",
    "n100009_bist_orange_seed2",
    "n100016_ndr_net_orange_seed42",
    "n100017_ndr_net_orange_seed1",
    "n100018_ndr_net_orange_seed2",
]
artifact_names = {
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if results.is_symlink() or {item.name for item in results.iterdir()} != {
    "queue_status.json",
    "tasks",
}:
    raise SystemExit("native100 result root allowlist mismatch")
tasks = results / "tasks"
if tasks.is_symlink() or {item.name for item in tasks.iterdir()} != set(selected_ids):
    raise SystemExit("native100 task directory allowlist mismatch")
relative_paths = ["queue_status.json"]
for task_id in selected_ids:
    task_dir = tasks / task_id
    if task_dir.is_symlink() or {item.name for item in task_dir.iterdir()} != artifact_names:
        raise SystemExit(f"native100 artifact allowlist mismatch: {task_id}")
    relative_paths.extend(
        f"tasks/{task_id}/{name}" for name in sorted(artifact_names)
    )
relative_paths = sorted(relative_paths)
records = []
for relative in relative_paths:
    source = results / relative
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise SystemExit(f"unsafe native100 artifact: {relative}")
    records.append(
        {
            "path": relative,
            "size_bytes": source.stat().st_size,
            "sha256": hash_file(source),
        }
    )
files_digest_payload = "".join(
    f"{item['sha256']}  {item['path']}\n" for item in records
).encode("utf-8")
files_digest = hashlib.sha256(files_digest_payload).hexdigest()
report = {
    "schema_version": 1,
    "status": "verified",
    "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "protocol_id": "igstgnn_table4_official_v1",
    "matrix_id": "igstgnn_table4_bist_ndr_native100_18_v1",
    "manifest_sha256": matrix_sha256,
    "base_manifest_sha256": base_sha256,
    "plan_id": "igstgnn_table4_native100_5080_5090_v1",
    "plan_sha256": plan_sha256,
    "host_id": host_id,
    "runtime_hostname": socket.gethostname(),
    "claim_scope": claim_scope,
    "validated_tasks": len(selected_ids),
    "task_ids": selected_ids,
    "file_count": len(records),
    "total_bytes": sum(item["size_bytes"] for item in records),
    "files_digest": files_digest,
    "files": records,
}
report_path.write_text(
    json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
file_list_path.write_text(
    "".join(f"{relative}\n" for relative in relative_paths),
    encoding="utf-8",
)
transfer_manifest_path.write_bytes(files_digest_payload)
marker = {
    key: report[key]
    for key in (
        "schema_version",
        "status",
        "protocol_id",
        "matrix_id",
        "manifest_sha256",
        "base_manifest_sha256",
        "plan_id",
        "plan_sha256",
        "host_id",
        "claim_scope",
        "validated_tasks",
        "file_count",
        "total_bytes",
        "files_digest",
    )
}
marker["postflight_sha256"] = hash_file(report_path)
marker["transfer_manifest_sha256"] = hash_file(transfer_manifest_path)
temporary = marker_path.with_suffix(".tmp")
temporary.write_text(
    json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker_path)
PY

[[ -s "$POSTFLIGHT_REPORT" && -s "$TRANSFER_FILE_LIST" \
  && -s "$TRANSFER_MANIFEST" && -s "$TRANSFER_MARKER" ]] \
  || fail "transfer metadata is incomplete"
log "NATIVE100_SOURCE_EXPORT_OK"

SSH_OPTIONS=(
  -i "$SSH_IDENTITY_FILE"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=6
)
printf -v RSYNC_SSH_COMMAND '%q ' "$SSH_BIN" "${SSH_OPTIONS[@]}"
RSYNC_SSH_COMMAND=${RSYNC_SSH_COMMAND% }
export RSYNC_RSH="$RSYNC_SSH_COMMAND"

remote_incoming="$RELAY_BASE/.$HOST_ID.incoming"
remote_final="$RELAY_BASE/$HOST_ID"
source_bytes=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' \
  "$POSTFLIGHT_REPORT")
required_bytes=$((source_bytes + RESERVE_BYTES))
"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; mkdir -p '$RELAY_BASE'; \
   test ! -e '$remote_final'; test ! -e '$remote_incoming'; \
   free=\$(df -PB1 '$RELAY_BASE' | awk 'NR==2 {print \$4}'); \
   test \"\$free\" -ge '$required_bytes'; \
   mkdir -p '$remote_incoming/results' '$remote_incoming/metadata'" \
  || fail "relay preparation or capacity gate failed"

"$RSYNC_BIN" -a --files-from="$TRANSFER_FILE_LIST" \
  "$RESULTS_ROOT/" "$RELAY:$remote_incoming/results/" \
  || fail "native100 result transfer failed"
"$RSYNC_BIN" -a \
  "$POSTFLIGHT_REPORT" \
  "$TRANSFER_FILE_LIST" \
  "$TRANSFER_MANIFEST" \
  "$TRANSFER_MARKER" \
  "$MANIFEST" \
  "$BASE_MANIFEST" \
  "$PLAN" \
  "$RELAY:$remote_incoming/metadata/" \
  || fail "native100 metadata transfer failed"

postflight_name=$(basename "$POSTFLIGHT_REPORT")
file_list_name=$(basename "$TRANSFER_FILE_LIST")
transfer_manifest_name=$(basename "$TRANSFER_MANIFEST")
marker_name=$(basename "$TRANSFER_MARKER")
manifest_name=$(basename "$MANIFEST")
base_manifest_name=$(basename "$BASE_MANIFEST")
plan_name=$(basename "$PLAN")

"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; root='$remote_incoming'; \
   test -d \"\$root/results\"; test ! -L \"\$root/results\"; \
   test -d \"\$root/metadata\"; test ! -L \"\$root/metadata\"; \
   test -z \"\$(find \"\$root\" -type l -print -quit)\"; \
   cd \"\$root/results\"; \
   sha256sum -c \"../metadata/$transfer_manifest_name\" >/dev/null; \
   tmp=\$(mktemp -d); trap 'rm -rf \"\$tmp\"' EXIT; \
   cut -c67- \"../metadata/$transfer_manifest_name\" | LC_ALL=C sort > \"\$tmp/expected\"; \
   find . -type f -printf '%P\\n' | LC_ALL=C sort > \"\$tmp/actual\"; \
   cmp \"\$tmp/expected\" \"\$tmp/actual\"; \
   cd \"\$root/metadata\"; \
   printf '%s\\n' '$postflight_name' '$file_list_name' '$transfer_manifest_name' \
     '$marker_name' '$manifest_name' '$base_manifest_name' '$plan_name' \
     | LC_ALL=C sort > \"\$tmp/expected_meta\"; \
   find . -maxdepth 1 -type f -printf '%f\\n' | LC_ALL=C sort > \"\$tmp/actual_meta\"; \
   cmp \"\$tmp/expected_meta\" \"\$tmp/actual_meta\"; \
   test \"\$(sha256sum '$manifest_name' | awk '{print \$1}')\" = '$MATRIX_SHA256'; \
   test \"\$(sha256sum '$base_manifest_name' | awk '{print \$1}')\" = '$BASE_SHA256'; \
   test \"\$(sha256sum '$plan_name' | awk '{print \$1}')\" = '$PLAN_SHA256'; \
   cd '$RELAY_BASE'; mv '.$HOST_ID.incoming' '$HOST_ID'" \
  || fail "relay verification or atomic promotion failed"

"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; root='$remote_final'; test -d \"\$root\"; \
   cd \"\$root/results\"; \
   sha256sum -c \"../metadata/$transfer_manifest_name\" >/dev/null; \
   test -s \"../metadata/$marker_name\"" \
  || fail "promoted relay tree failed final verification"

sync
log "NATIVE100_RELAY_VERIFIED destination=$remote_final"
log "NATIVE100_FINALIZE_SUCCESS shutting_down"
"$SHUTDOWN_BIN" -h now
