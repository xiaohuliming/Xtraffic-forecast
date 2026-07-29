#!/usr/bin/env bash
# Halt the five-GPU host after its already-promoted original Table 4 export is reverified.
set -euo pipefail

HOST_ID=${HOST_ID:-table4_4090d5}
ROOT=${ROOT:-/root/autodl-tmp/table4}
CONTROL_DIR=${CONTROL_DIR:-$ROOT/control/$HOST_ID}
TRANSFER_MARKER=${TRANSFER_MARKER:-$CONTROL_DIR/transfer_complete.json}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-$CONTROL_DIR/transfer.sha256}
TRANSFER_FILE_LIST=${TRANSFER_FILE_LIST:-$CONTROL_DIR/transfer.files}
POSTFLIGHT_REPORT=${POSTFLIGHT_REPORT:-$CONTROL_DIR/postflight.json}
FINALIZE_LOG=${FINALIZE_LOG:-$CONTROL_DIR/finalize.log}
SHUTDOWN_DEFER_FILE=${SHUTDOWN_DEFER_FILE:-$CONTROL_DIR/defer_shutdown_for_ndr_extension}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-$CONTROL_DIR/relay_ed25519}
RELAY=${RELAY:-root@8.134.12.5}
RELAY_BASE=${RELAY_BASE:-/root/xtraffic_results/igstgnn_table4_official_v1/distributed_4090d_9gpu_v1/hosts}
SSH_BIN=${SSH_BIN:-/usr/bin/ssh}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
WAIT_SECONDS=${WAIT_SECONDS:-60}
WAIT_ONCE=${WAIT_ONCE:-0}
LOG=${LOG:-$CONTROL_DIR/original_shutdown_waiter.log}

mkdir -p "$CONTROL_DIR"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

fail() {
  log "ORIGINAL_SHUTDOWN_BLOCKED $*"
  exit 1
}

[[ "$HOST_ID" == "table4_4090d5" ]] || fail "invalid HOST_ID"
[[ -f "$SHUTDOWN_DEFER_FILE" && ! -L "$SHUTDOWN_DEFER_FILE" ]] \
  || fail "shutdown defer sentinel is missing or unsafe"
[[ "$RELAY_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "unsafe RELAY_BASE"

log "ORIGINAL_SHUTDOWN_WAIT export=$TRANSFER_MARKER"
while true; do
  if [[ -s "$TRANSFER_MARKER" && ! -L "$TRANSFER_MARKER" ]] \
    && [[ -s "$TRANSFER_MANIFEST" && ! -L "$TRANSFER_MANIFEST" ]] \
    && [[ -s "$TRANSFER_FILE_LIST" && ! -L "$TRANSFER_FILE_LIST" ]] \
    && [[ -s "$POSTFLIGHT_REPORT" && ! -L "$POSTFLIGHT_REPORT" ]] \
    && [[ -s "$FINALIZE_LOG" && ! -L "$FINALIZE_LOG" ]] \
    && grep -Fq "FINALIZE_SUCCESS shutdown_deferred" "$FINALIZE_LOG"; then
    break
  fi
  if [[ "$WAIT_ONCE" == "1" ]]; then
    fail "original export is not complete"
  fi
  sleep "$WAIT_SECONDS"
done

python3 - "$TRANSFER_MARKER" "$TRANSFER_MANIFEST" "$TRANSFER_FILE_LIST" \
  "$POSTFLIGHT_REPORT" "$HOST_ID" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

marker_name, manifest_name, file_list_name, report_name, host_id = sys.argv[1:]
marker_path = Path(marker_name)
manifest_path = Path(manifest_name)
file_list_path = Path(file_list_name)
report_path = Path(report_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


marker = json.loads(marker_path.read_text(encoding="utf-8"))
required = {
    "schema_version": 1,
    "status": "verified",
    "protocol_id": "igstgnn_table4_official_v1",
    "matrix_id": "igstgnn_table4_public_repro_39_v2",
    "plan_id": "igstgnn_table4_4090d_9gpu_v1",
    "host_id": host_id,
    "gpu_sku": "NVIDIA GeForce RTX 4090 D",
    "execution_layout": "dual_host_gpu_sku_homogeneous",
    "validated_shards": 5,
    "validated_tasks": 13,
    "file_count": 70,
}
for key, expected in required.items():
    if marker.get(key) != expected:
        raise SystemExit(f"invalid transfer marker field: {key}")
for key in (
    "plan_sha256",
    "manifest_sha256",
    "files_digest",
    "file_list_sha256",
    "transfer_manifest_sha256",
    "postflight_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(marker.get(key, ""))) is None:
        raise SystemExit(f"invalid transfer marker digest: {key}")
if marker["transfer_manifest_sha256"] != sha256_file(manifest_path):
    raise SystemExit("transfer manifest changed after finalization")
if marker["file_list_sha256"] != sha256_file(file_list_path):
    raise SystemExit("transfer file list changed after finalization")
if marker["postflight_sha256"] != sha256_file(report_path):
    raise SystemExit("postflight report changed after finalization")
if sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line) != 70:
    raise SystemExit("unexpected transfer manifest entry count")
PY

if pgrep -af \
  'igstgnn_table4\.(distributed_host_runner|matrix_runner|task_runner|recovery_task_runner)' \
  | grep -v 'pgrep -af' >/dev/null; then
  fail "original Table 4 process remains active after finalization"
fi

[[ -f "$SSH_IDENTITY_FILE" && ! -L "$SSH_IDENTITY_FILE" ]] \
  || fail "relay identity file is missing or unsafe"
marker_digest=$(sha256sum "$TRANSFER_MARKER" | awk '{print $1}')
manifest_digest=$(sha256sum "$TRANSFER_MANIFEST" | awk '{print $1}')
file_list_digest=$(sha256sum "$TRANSFER_FILE_LIST" | awk '{print $1}')
report_digest=$(sha256sum "$POSTFLIGHT_REPORT" | awk '{print $1}')
remote_final="$RELAY_BASE/$HOST_ID"

SSH_OPTIONS=(
  -i "$SSH_IDENTITY_FILE"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=6
)

"$SSH_BIN" "${SSH_OPTIONS[@]}" "$RELAY" \
  "set -eu; root='$remote_final'; \
   test -d \"\$root\"; test ! -L \"\$root\"; \
   test -d \"\$root/results\"; test ! -L \"\$root/results\"; \
   test -d \"\$root/metadata\"; test ! -L \"\$root/metadata\"; \
   test -f \"\$root/_TRANSFER_COMPLETE.json\"; \
   test ! -L \"\$root/_TRANSFER_COMPLETE.json\"; \
   test \"\$(sha256sum \"\$root/_TRANSFER_COMPLETE.json\" | awk '{print \$1}')\" = '$marker_digest'; \
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
   cmp \"\$tmp/expected.files\" \"\$tmp/actual.files\"; \
   printf '%s\\n' postflight.json transfer.files transfer.sha256 \
     | LC_ALL=C sort > \"\$tmp/expected.metadata\"; \
   find \"\$root/metadata\" -mindepth 1 -maxdepth 1 -type f -printf '%f\\n' \
     | LC_ALL=C sort > \"\$tmp/actual.metadata\"; \
   cmp \"\$tmp/expected.metadata\" \"\$tmp/actual.metadata\"" \
  || fail "promoted original relay tree failed full verification"

rm -f "$SHUTDOWN_DEFER_FILE"
sync
log "ORIGINAL_SHUTDOWN_SUCCESS shutting_down"
"$SHUTDOWN_BIN" -h now
