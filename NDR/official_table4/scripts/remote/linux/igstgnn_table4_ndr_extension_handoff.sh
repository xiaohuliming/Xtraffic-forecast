#!/usr/bin/env bash
# Wait for the original 39-task host export, run the NDR extension, export, and halt.
set -euo pipefail

HOST_ID=${HOST_ID:?HOST_ID is required}
ROOT=${ROOT:-/root/autodl-tmp/table4}
REPO=${REPO:-$ROOT/ndr_repo}
PYTHON=${PYTHON:-$ROOT/env/bin/python}
SPLITS_ROOT=${SPLITS_ROOT:-$ROOT/data/official_splits}
RESULTS_BASE=${RESULTS_BASE:-$ROOT/ndr_extension/results}
CONTROL_DIR=${CONTROL_DIR:-$ROOT/ndr_extension/control/$HOST_ID}
PLAN=${PLAN:-$REPO/config/reproduction/igstgnn_table4/distributed_ndr_extension_9gpu_v1.json}
MANIFEST=${MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.ndr_extension_v1.json}
MANIFEST_SHA256=${MANIFEST_SHA256:-2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4}
PROFILE=${PROFILE:-$REPO/config/reproduction/igstgnn_table4/ndr_extension_profile_v1.json}
BASE_MANIFEST=${BASE_MANIFEST:-$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json}
BASE_MANIFEST_SHA256=${BASE_MANIFEST_SHA256:-d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78}
ORIGINAL_CONTROL=${ORIGINAL_CONTROL:-$ROOT/control/$HOST_ID}
ORIGINAL_TRANSFER_MARKER=${ORIGINAL_TRANSFER_MARKER:-$ORIGINAL_CONTROL/transfer_complete.json}
ORIGINAL_FINALIZE_LOG=${ORIGINAL_FINALIZE_LOG:-$ORIGINAL_CONTROL/finalize.log}
SHUTDOWN_DEFER_FILE=${SHUTDOWN_DEFER_FILE:-$ORIGINAL_CONTROL/defer_shutdown_for_ndr_extension}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-$CONTROL_DIR/postflight.json}
TRANSFER_FILE_LIST=${TRANSFER_FILE_LIST:-$CONTROL_DIR/transfer.files}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-$CONTROL_DIR/transfer.sha256}
TRANSFER_MARKER=${TRANSFER_MARKER:-$CONTROL_DIR/transfer_complete.json}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/igstgnn_table4_official_v1/ndr_extension_9gpu_v1/hosts}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-$ORIGINAL_CONTROL/relay_ed25519}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
RSYNC_BIN=${RSYNC_BIN:-/usr/bin/rsync}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
REALPATH_BIN=${REALPATH_BIN:-/usr/bin/realpath}
WAIT_SECONDS=${WAIT_SECONDS:-60}
HANDOFF_WAIT_ONCE=${HANDOFF_WAIT_ONCE:-0}
LOG=${LOG:-$CONTROL_DIR/handoff.log}

mkdir -p "$CONTROL_DIR"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "NDR_HANDOFF_BLOCKED $*"
  exit 1
}

[[ "$HOST_ID" =~ ^table4_4090d[45]$ ]] || fail "invalid HOST_ID"
[[ -d "$REPO" && ! -L "$REPO" ]] || fail "NDR repository is missing or unsafe"
[[ -x "$PYTHON" ]] || fail "Python runtime is missing or unsafe"
[[ -x "$REALPATH_BIN" ]] || fail "realpath utility is missing or unsafe"
resolved_root=$("$REALPATH_BIN" "$ROOT")
resolved_python=$("$REALPATH_BIN" "$PYTHON")
[[ -f "$resolved_python" && ! -L "$resolved_python" && -x "$resolved_python" ]] \
  || fail "resolved Python runtime is missing or unsafe"
case "$resolved_python" in
  "$resolved_root"/env/bin/*) ;;
  *) fail "resolved Python runtime escapes the bound environment" ;;
esac
[[ -d "$SPLITS_ROOT" && ! -L "$SPLITS_ROOT" ]] \
  || fail "official split root is missing or unsafe"
for source in "$PLAN" "$MANIFEST" "$PROFILE" "$BASE_MANIFEST"; do
  [[ -f "$source" && ! -L "$source" ]] || fail "bound config is missing or unsafe"
done
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] \
  || fail "NDR extension manifest hash mismatch"
[[ "$(sha256sum "$BASE_MANIFEST" | awk '{print $1}')" == "$BASE_MANIFEST_SHA256" ]] \
  || fail "base manifest hash mismatch"
[[ -f "$SHUTDOWN_DEFER_FILE" && ! -L "$SHUTDOWN_DEFER_FILE" ]] \
  || fail "shutdown defer sentinel is missing or unsafe"
[[ -f "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
  || fail "relay identity file is missing or unsafe"

log "NDR_HANDOFF_WAIT_ORIGINAL host=$HOST_ID"
while true; do
  if [[ -s "$ORIGINAL_TRANSFER_MARKER" && ! -L "$ORIGINAL_TRANSFER_MARKER" ]] \
    && [[ -s "$ORIGINAL_FINALIZE_LOG" && ! -L "$ORIGINAL_FINALIZE_LOG" ]] \
    && grep -Fq "FINALIZE_SUCCESS shutdown_deferred" "$ORIGINAL_FINALIZE_LOG"; then
    break
  fi
  if [[ "$HANDOFF_WAIT_ONCE" == "1" ]]; then
    fail "original host export is not complete"
  fi
  sleep "$WAIT_SECONDS"
done

if pgrep -af \
  'igstgnn_table4\.(distributed_host_runner|matrix_runner|task_runner|recovery_task_runner)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "original Table 4 process remains active after finalization"
fi
log "NDR_HANDOFF_ORIGINAL_VERIFIED host=$HOST_ID"

export PYTHONPATH="$REPO"
if [[ "$HOST_ID" == "table4_4090d4" ]]; then
  probe_task=ndr004_contra_costa_seed42
  probe_gpu=3
else
  probe_task=ndr007_orange_seed42
  probe_gpu=4
fi
CUDA_VISIBLE_DEVICES="$probe_gpu" \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"$PYTHON" -X utf8 -m \
  scripts.reproduction.igstgnn_table4.ndr_extension_runner \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --base-manifest "$BASE_MANIFEST" \
  --base-manifest-sha256 "$BASE_MANIFEST_SHA256" \
  --task-id "$probe_task" \
  --splits-root "$SPLITS_ROOT" \
  --device cuda:0 \
  --smoke >> "$LOG" 2>&1 \
  || fail "NDR extension forward-backward probe failed"
log "NDR_PROBE_OK host=$HOST_ID task=$probe_task gpu=$probe_gpu"

"$PYTHON" -X utf8 -m \
  scripts.reproduction.igstgnn_table4.ndr_extension_distributed \
  run-host \
  --plan "$PLAN" \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --base-manifest "$BASE_MANIFEST" \
  --base-manifest-sha256 "$BASE_MANIFEST_SHA256" \
  --host-id "$HOST_ID" \
  --splits-root "$SPLITS_ROOT" \
  --results-base "$RESULTS_BASE" \
  --control-dir "$CONTROL_DIR" \
  --python "$PYTHON" >> "$LOG" 2>&1 \
  || fail "NDR extension host runner failed"
log "NDR_HOST_RUN_OK host=$HOST_ID"

"$PYTHON" -X utf8 -m \
  scripts.reproduction.igstgnn_table4.ndr_extension_distributed \
  postflight \
  --plan "$PLAN" \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --base-manifest "$BASE_MANIFEST" \
  --base-manifest-sha256 "$BASE_MANIFEST_SHA256" \
  --host-id "$HOST_ID" \
  --results-base "$RESULTS_BASE" \
  --report "$POSTFLIGHT_REPORT" >> "$LOG" 2>&1 \
  || fail "NDR extension source-host postflight failed"
[[ -s "$POSTFLIGHT_REPORT" && ! -L "$POSTFLIGHT_REPORT" ]] \
  || fail "NDR extension postflight report is missing or unsafe"
log "NDR_POSTFLIGHT_OK host=$HOST_ID"

"$PYTHON" - "$POSTFLIGHT_REPORT" "$RESULTS_BASE" \
  "$TRANSFER_FILE_LIST" "$TRANSFER_MANIFEST" "$TRANSFER_MARKER" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

report_name, results_name, list_name, manifest_name, marker_name = sys.argv[1:]
report_path = Path(report_name)
results_root = Path(results_name).resolve()
file_list = Path(list_name)
transfer_manifest = Path(manifest_name)
marker = Path(marker_name)
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    report.get("status") != "verified"
    or report.get("matrix_id") != "igstgnn_table4_ndr_extension_9_v1"
    or report.get("claim_scope") != "same_protocol_project_method_extension"
):
    raise SystemExit("invalid NDR extension postflight identity")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = report["files"]
ordered = sorted(record["path"] for record in records)
if report["export_paths"] != ordered:
    raise SystemExit("postflight export paths are not canonical")
by_path = {record["path"]: record for record in records}
for relative in ordered:
    source = (results_root / relative).resolve()
    source.relative_to(results_root)
    record = by_path[relative]
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size != record["size_bytes"]
        or sha256_file(source) != record["sha256"]
    ):
        raise SystemExit(f"postflight source changed: {relative}")
file_list.parent.mkdir(parents=True, exist_ok=True)
file_list.write_text("".join(f"{relative}\n" for relative in ordered), encoding="utf-8")
transfer_manifest.write_text(
    "".join(f"{by_path[relative]['sha256']}  {relative}\n" for relative in ordered),
    encoding="utf-8",
)
value = {
    "schema_version": 1,
    "status": "verified",
    "protocol_id": report["protocol_id"],
    "matrix_id": report["matrix_id"],
    "plan_id": report["plan_id"],
    "plan_sha256": report["plan_sha256"],
    "manifest_sha256": report["manifest_sha256"],
    "base_manifest_sha256": report["base_manifest_sha256"],
    "host_id": report["host_id"],
    "claim_scope": report["claim_scope"],
    "validated_tasks": report["validated_tasks"],
    "file_count": report["file_count"],
    "total_bytes": report["total_bytes"],
    "files_digest": report["files_digest"],
    "postflight_sha256": sha256_file(report_path),
    "file_list_sha256": sha256_file(file_list),
    "transfer_manifest_sha256": sha256_file(transfer_manifest),
}
temporary = marker.with_suffix(".tmp")
temporary.write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
PY

[[ -s "$TRANSFER_FILE_LIST" && -s "$TRANSFER_MANIFEST" && -s "$TRANSFER_MARKER" ]] \
  || fail "NDR extension transfer metadata is incomplete"

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
printf -v RSYNC_SSH_COMMAND '%q ' "$SSH_BIN" "${SSH_OPTIONS[@]}"
RSYNC_SSH_COMMAND=${RSYNC_SSH_COMMAND% }
export RSYNC_RSH="$RSYNC_SSH_COMMAND"

remote_incoming="$RELAY_BASE/.$HOST_ID.incoming"
remote_final="$RELAY_BASE/$HOST_ID"
"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; mkdir -p '$RELAY_BASE'; \
   test ! -e '$remote_final'; \
   mkdir -p '$remote_incoming/results' '$remote_incoming/metadata'" \
  || fail "cannot prepare relay incoming directory"

"$RSYNC_BIN" -a --files-from="$TRANSFER_FILE_LIST" \
  "$RESULTS_BASE/" "$RELAY:$remote_incoming/results/" \
  || fail "NDR extension result transfer failed"
"$RSYNC_BIN" -a \
  "$POSTFLIGHT_REPORT" \
  "$TRANSFER_FILE_LIST" \
  "$TRANSFER_MANIFEST" \
  "$TRANSFER_MARKER" \
  "$MANIFEST" \
  "$PROFILE" \
  "$PLAN" \
  "$BASE_MANIFEST" \
  "$RELAY:$remote_incoming/metadata/" \
  || fail "NDR extension metadata transfer failed"

marker_name=$(basename "$TRANSFER_MARKER")
postflight_name=$(basename "$POSTFLIGHT_REPORT")
transfer_list_name=$(basename "$TRANSFER_FILE_LIST")
transfer_manifest_name=$(basename "$TRANSFER_MANIFEST")
manifest_name=$(basename "$MANIFEST")
profile_name=$(basename "$PROFILE")
plan_name=$(basename "$PLAN")
base_manifest_name=$(basename "$BASE_MANIFEST")

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
   printf '%s\\n' \
     '$postflight_name' '$transfer_list_name' '$transfer_manifest_name' \
     '$marker_name' '$manifest_name' '$profile_name' '$plan_name' '$base_manifest_name' \
     | LC_ALL=C sort > \"\$tmp/expected_meta\"; \
   find . -maxdepth 1 -type f -printf '%f\\n' | LC_ALL=C sort > \"\$tmp/actual_meta\"; \
   cmp \"\$tmp/expected_meta\" \"\$tmp/actual_meta\"; \
   cd '$RELAY_BASE'; mv '.$HOST_ID.incoming' '$HOST_ID'" \
  || fail "relay verification or atomic promotion failed"

"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; root='$remote_final'; test -d \"\$root\"; \
   cd \"\$root/results\"; sha256sum -c \"../metadata/$transfer_manifest_name\" >/dev/null" \
  || fail "promoted relay tree failed final verification"
log "NDR_RELAY_VERIFIED host=$HOST_ID destination=$remote_final"

rm -f "$SHUTDOWN_DEFER_FILE"
sync
log "NDR_HANDOFF_SUCCESS shutting_down"
"$SHUTDOWN_BIN" -h now
