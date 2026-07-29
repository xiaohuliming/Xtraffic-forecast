#!/usr/bin/env bash
# Validate one distributed Table 4 host, transfer it, verify it remotely, and halt.
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/table4/repo}
RESULTS_BASE=${RESULTS_BASE:-/root/autodl-tmp/table4/results}
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/distributed_4090d_9gpu_v1.json}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
MANIFEST_SHA256=${MANIFEST_SHA256:-d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78}
HOST_ID=${HOST_ID:?HOST_ID is required}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-/root/autodl-tmp/table4/control/$HOST_ID/postflight.json}
POSTFLIGHT_CMD=${POSTFLIGHT_CMD:-}
TRANSFER_FILE_LIST=${TRANSFER_FILE_LIST:-/root/autodl-tmp/table4/control/$HOST_ID/transfer.files}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-/root/autodl-tmp/table4/control/$HOST_ID/transfer.sha256}
TRANSFER_MARKER=${TRANSFER_MARKER:-/root/autodl-tmp/table4/control/$HOST_ID/transfer_complete.json}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/igstgnn_table4_official_v1/distributed_4090d_9gpu_v1/hosts}
REMOTE_NAME=${REMOTE_NAME:-$HOST_ID}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-}
SPACE_MARGIN_BYTES=${SPACE_MARGIN_BYTES:-2147483648}
SPACE_RETRY_SECONDS=${SPACE_RETRY_SECONDS:-300}
FINALIZER_ONCE=${FINALIZER_ONCE:-0}
PYTHON=${PYTHON:-/root/autodl-tmp/table4/env/bin/python}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
RSYNC_BIN=${RSYNC_BIN:-/usr/bin/rsync}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/sbin/shutdown}
SHUTDOWN_DEFER_FILE=${SHUTDOWN_DEFER_FILE:-/root/autodl-tmp/table4/control/$HOST_ID/defer_shutdown_for_ndr_extension}
LOG=${LOG:-/root/autodl-tmp/table4/control/$HOST_ID/finalize.log}

mkdir -p "$(dirname "$LOG")"

SSH_OPTIONS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=6
)
if [[ -n "$SSH_IDENTITY_FILE" ]]; then
  SSH_OPTIONS=(-i "$SSH_IDENTITY_FILE" "${SSH_OPTIONS[@]}")
fi
RSYNC_SSH_OPTIONS=("$SSH_BIN" "${SSH_OPTIONS[@]}")
printf -v RSYNC_SSH_COMMAND '%q ' "${RSYNC_SSH_OPTIONS[@]}"
RSYNC_SSH_COMMAND=${RSYNC_SSH_COMMAND% }

remote_incoming="$RELAY_BASE/.${REMOTE_NAME}.incoming"
remote_final="$RELAY_BASE/$REMOTE_NAME"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "FINALIZE_BLOCKED $*"
  exit 1
}

finish_successfully() {
  sync
  if [[ -e "$SHUTDOWN_DEFER_FILE" || -L "$SHUTDOWN_DEFER_FILE" ]]; then
    [[ -f "$SHUTDOWN_DEFER_FILE" && ! -L "$SHUTDOWN_DEFER_FILE" ]] \
      || fail "shutdown defer sentinel is unsafe"
    log "FINALIZE_SUCCESS shutdown_deferred sentinel=$SHUTDOWN_DEFER_FILE"
    return
  fi
  log "FINALIZE_SUCCESS shutting_down"
  "$SHUTDOWN_BIN" -h now
}

[[ "$HOST_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe HOST_ID"
[[ "$REMOTE_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe REMOTE_NAME"
[[ "$RELAY_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "unsafe RELAY_BASE"
[[ -d "$RESULTS_BASE" ]] || fail "results base is missing"
[[ -f "$PLAN" && ! -L "$PLAN" ]] || fail "plan is missing or unsafe"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail "manifest is missing or unsafe"
if [[ -n "$SSH_IDENTITY_FILE" ]]; then
  [[ -f "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
    || fail "relay identity file is missing or unsafe"
fi

export PYTHONPATH="$REPO"
export RESULTS_BASE POSTFLIGHT_REPORT PLAN HOST_ID
log "POSTFLIGHT_START host=$HOST_ID"
if [[ -n "$POSTFLIGHT_CMD" ]]; then
  bash -lc "$POSTFLIGHT_CMD" >> "$LOG" 2>&1 \
    || fail "postflight validation failed"
else
  "$PYTHON" -X utf8 -m \
    scripts.reproduction.igstgnn_table4.distributed_host_postflight \
    --plan "$PLAN" \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --host-id "$HOST_ID" \
    --results-base "$RESULTS_BASE" \
    --report "$POSTFLIGHT_REPORT" >> "$LOG" 2>&1 \
    || fail "postflight validation failed"
fi
[[ -s "$POSTFLIGHT_REPORT" && ! -L "$POSTFLIGHT_REPORT" ]] \
  || fail "postflight report is missing or unsafe"
log "POSTFLIGHT_OK host=$HOST_ID"

"$PYTHON" - \
  "$POSTFLIGHT_REPORT" \
  "$PLAN" \
  "$MANIFEST_SHA256" \
  "$HOST_ID" \
  "$RESULTS_BASE" \
  "$TRANSFER_FILE_LIST" \
  "$TRANSFER_MANIFEST" \
  "$TRANSFER_MARKER" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

(
    report_name,
    plan_name,
    manifest_sha256,
    host_id,
    results_name,
    file_list_name,
    transfer_manifest_name,
    marker_name,
) = sys.argv[1:]
report_path = Path(report_name)
plan_path = Path(plan_name)
results_root = Path(results_name).resolve()
file_list_path = Path(file_list_name)
transfer_manifest_path = Path(transfer_manifest_name)
marker_path = Path(marker_name)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


report = json.loads(report_path.read_text(encoding="utf-8"))
plan_digest = sha256_file(plan_path)
if (
    report.get("schema_version") != 1
    or report.get("status") != "verified"
    or report.get("protocol_id") != "igstgnn_table4_official_v1"
    or report.get("matrix_id") != "igstgnn_table4_public_repro_39_v2"
    or report.get("plan_id") != "igstgnn_table4_4090d_9gpu_v1"
    or report.get("plan_sha256") != plan_digest
    or report.get("manifest_sha256") != manifest_sha256
    or report.get("host_id") != host_id
    or report.get("gpu_sku") != "NVIDIA GeForce RTX 4090 D"
    or report.get("execution_layout") != "dual_host_gpu_sku_homogeneous"
):
    raise SystemExit("invalid postflight report identity")
validated_shards = report.get("validated_shards")
validated_tasks = report.get("validated_tasks")
if (
    isinstance(validated_shards, bool)
    or not isinstance(validated_shards, int)
    or validated_shards <= 0
    or isinstance(validated_tasks, bool)
    or not isinstance(validated_tasks, int)
    or validated_tasks <= 0
):
    raise SystemExit("invalid postflight validated counts")
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
        or size <= 0
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
    except ValueError as error:
        raise SystemExit(f"export path escapes results root: {relative}") from error
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"export source is missing or unsafe: {relative}")
    if source.stat().st_size != size or sha256_file(source) != digest:
        raise SystemExit(f"export source changed after postflight: {relative}")
    by_path[relative] = digest
    total_bytes += size
ordered = sorted(by_path)
if export_paths != ordered:
    raise SystemExit("postflight export paths are not canonical")
if report.get("file_count") != len(records):
    raise SystemExit("postflight file_count mismatch")
if report.get("total_bytes") != total_bytes:
    raise SystemExit("postflight total_bytes mismatch")
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
    raise SystemExit("postflight files_digest mismatch")

for destination in (file_list_path, transfer_manifest_path, marker_path):
    destination.parent.mkdir(parents=True, exist_ok=True)
file_tmp = file_list_path.with_suffix(file_list_path.suffix + ".tmp")
manifest_tmp = transfer_manifest_path.with_suffix(
    transfer_manifest_path.suffix + ".tmp"
)
file_tmp.write_text(
    "".join(f"{relative}\n" for relative in ordered),
    encoding="utf-8",
)
manifest_tmp.write_text(
    "".join(f"{by_path[relative]}  {relative}\n" for relative in ordered),
    encoding="utf-8",
)
os.replace(file_tmp, file_list_path)
os.replace(manifest_tmp, transfer_manifest_path)
marker = {
    "schema_version": 1,
    "status": "verified",
    "protocol_id": report["protocol_id"],
    "matrix_id": report["matrix_id"],
    "plan_id": report["plan_id"],
    "plan_sha256": plan_digest,
    "manifest_sha256": manifest_sha256,
    "host_id": host_id,
    "gpu_sku": report["gpu_sku"],
    "execution_layout": report["execution_layout"],
    "validated_shards": validated_shards,
    "validated_tasks": validated_tasks,
    "file_count": len(records),
    "source_bytes": total_bytes,
    "files_digest": files_digest,
    "file_list_sha256": sha256_file(file_list_path),
    "transfer_manifest_sha256": sha256_file(transfer_manifest_path),
    "postflight_sha256": sha256_file(report_path),
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
[[ -s "$TRANSFER_MARKER" ]] || fail "transfer marker is empty"

source_bytes="$("$PYTHON" - "$POSTFLIGHT_REPORT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["total_bytes"])
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
  "$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
    "set -eu; \
    root='$root'; marker_name='$marker_name'; \
    test -d \"\$root\"; test ! -L \"\$root\"; \
    test -d \"\$root/results\"; test ! -L \"\$root/results\"; \
    test -d \"\$root/metadata\"; test ! -L \"\$root/metadata\"; \
    test -f \"\$root/\$marker_name\"; test ! -L \"\$root/\$marker_name\"; \
    test \"\$(sha256sum \"\$root/\$marker_name\" | awk '{print \$1}')\" = '$marker_digest'; \
    test \"\$(sha256sum \"\$root/metadata/transfer.sha256\" | awk '{print \$1}')\" = '$manifest_digest'; \
    test \"\$(sha256sum \"\$root/metadata/transfer.files\" | awk '{print \$1}')\" = '$file_list_digest'; \
    test \"\$(sha256sum \"\$root/metadata/postflight.json\" | awk '{print \$1}')\" = '$report_digest'; \
    test -z \"\$(find \"\$root\" -type l -print -quit)\"; \
    test -z \"\$(find \"\$root\" ! -type f ! -type d -print -quit)\"; \
    tmp=\$(mktemp -d); trap 'rm -rf \"\$tmp\"' EXIT; \
    cd \"\$root/results\"; \
    sha256sum -c ../metadata/transfer.sha256 >/dev/null; \
    cut -c67- ../metadata/transfer.sha256 | LC_ALL=C sort > \"\$tmp/expected.files\"; \
    find . -type f -printf '%P\\n' | LC_ALL=C sort > \"\$tmp/actual.files\"; \
    cmp \"\$tmp/actual.files\" \"\$tmp/expected.files\"; \
    printf '%s\\n' postflight.json transfer.files transfer.sha256 \
      | LC_ALL=C sort > \"\$tmp/expected.metadata\"; \
    find \"\$root/metadata\" -mindepth 1 -maxdepth 1 -type f -printf '%f\\n' \
      | LC_ALL=C sort > \"\$tmp/actual.metadata\"; \
    cmp \"\$tmp/actual.metadata\" \"\$tmp/expected.metadata\""
}

while true; do
  read -r remote_free incoming_bytes final_exists < <(
    "$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
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
  [[ "$remote_free" =~ ^[0-9]+$ ]] || fail "cannot read relay free space"
  [[ "$incoming_bytes" =~ ^[0-9]+$ ]] || fail "cannot read incoming size"
  [[ "$final_exists" =~ ^[01]$ ]] || fail "cannot read final destination state"
  if [[ "$final_exists" == "1" ]]; then
    if verify_remote_tree "$remote_final" "_TRANSFER_COMPLETE.json"; then
      log "REMOTE_ALREADY_VERIFIED destination=$RELAY:$remote_final"
      finish_successfully
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
"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; rm -rf '$remote_incoming'; \
  mkdir -p '$remote_incoming/results' '$remote_incoming/metadata'"

"$RSYNC_BIN" \
  --archive \
  --relative \
  --partial \
  --human-readable \
  --files-from="$TRANSFER_FILE_LIST" \
  -e "$RSYNC_SSH_COMMAND" \
  "$RESULTS_BASE/" \
  "$RELAY:$remote_incoming/results/" >> "$LOG" 2>&1 \
  || fail "result rsync failed"
"$RSYNC_BIN" --archive -e "$RSYNC_SSH_COMMAND" \
  "$TRANSFER_MANIFEST" "$RELAY:$remote_incoming/metadata/transfer.sha256" \
  >> "$LOG" 2>&1 || fail "manifest rsync failed"
"$RSYNC_BIN" --archive -e "$RSYNC_SSH_COMMAND" \
  "$TRANSFER_FILE_LIST" "$RELAY:$remote_incoming/metadata/transfer.files" \
  >> "$LOG" 2>&1 || fail "file list rsync failed"
"$RSYNC_BIN" --archive -e "$RSYNC_SSH_COMMAND" \
  "$POSTFLIGHT_REPORT" "$RELAY:$remote_incoming/metadata/postflight.json" \
  >> "$LOG" 2>&1 || fail "postflight rsync failed"
"$RSYNC_BIN" --archive -e "$RSYNC_SSH_COMMAND" \
  "$TRANSFER_MARKER" "$RELAY:$remote_incoming/_TRANSFER_COMPLETE.json.tmp" \
  >> "$LOG" 2>&1 || fail "marker rsync failed"

log "REMOTE_VERIFY_START"
verify_remote_tree "$remote_incoming" "_TRANSFER_COMPLETE.json.tmp" \
  || fail "remote exact tree or hash verification failed"
"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; test ! -e '$remote_final'; \
  mv '$remote_incoming/_TRANSFER_COMPLETE.json.tmp' \
    '$remote_incoming/_TRANSFER_COMPLETE.json'; \
  sync; mv '$remote_incoming' '$remote_final'; sync" \
  || fail "remote atomic promotion failed"
verify_remote_tree "$remote_final" "_TRANSFER_COMPLETE.json" \
  || fail "promoted remote tree failed full verification"

log "REMOTE_VERIFY_OK destination=$RELAY:$remote_final"
finish_successfully
