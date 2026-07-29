#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$ROOT_DIR/scripts/remote/linux/igstgnn_table4_original_export_shutdown_waiter.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/root/control/table4_4090d5" "$TMP/bin"
touch "$TMP/root/control/table4_4090d5/defer_shutdown_for_ndr_extension"
cat > "$TMP/bin/shutdown" <<EOF
#!/usr/bin/env bash
touch "$TMP/shutdown.called"
EOF
chmod +x "$TMP/bin/shutdown"

set +e
HOST_ID=table4_4090d5 \
ROOT="$TMP/root" \
WAIT_ONCE=1 \
SHUTDOWN_BIN="$TMP/bin/shutdown" \
PYTHON=python3 \
bash "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e

[[ "$rc" != "0" ]]
[[ ! -e "$TMP/shutdown.called" ]]
[[ -f "$TMP/root/control/table4_4090d5/defer_shutdown_for_ndr_extension" ]]
grep -Fq "ORIGINAL_SHUTDOWN_BLOCKED original export is not complete" \
  "$TMP/root/control/table4_4090d5/original_shutdown_waiter.log"
echo "verified: incomplete original export preserves defer sentinel and power"

success_root="$TMP/success"
success_control="$success_root/control/table4_4090d5"
mkdir -p "$success_control"
touch "$success_control/defer_shutdown_for_ndr_extension"
touch "$success_control/relay_ed25519"
printf '%s\n' "2026-07-29T00:00:00Z FINALIZE_SUCCESS shutdown_deferred" \
  > "$success_control/finalize.log"
for index in $(seq -w 1 70); do
  printf 'file%s\n' "$index" >> "$success_control/transfer.files"
  printf '%064d  file%s\n' 0 "$index" >> "$success_control/transfer.sha256"
done
printf '%s\n' '{}' > "$success_control/postflight.json"
python3 - "$success_control" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sha = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
marker = {
    "schema_version": 1,
    "status": "verified",
    "protocol_id": "igstgnn_table4_official_v1",
    "matrix_id": "igstgnn_table4_public_repro_39_v2",
    "plan_id": "igstgnn_table4_4090d_9gpu_v1",
    "plan_sha256": "1" * 64,
    "manifest_sha256": "2" * 64,
    "host_id": "table4_4090d5",
    "gpu_sku": "NVIDIA GeForce RTX 4090 D",
    "execution_layout": "dual_host_gpu_sku_homogeneous",
    "validated_shards": 5,
    "validated_tasks": 13,
    "file_count": 70,
    "source_bytes": 70,
    "files_digest": "3" * 64,
    "file_list_sha256": sha("transfer.files"),
    "transfer_manifest_sha256": sha("transfer.sha256"),
    "postflight_sha256": sha("postflight.json"),
}
(root / "transfer_complete.json").write_text(
    json.dumps(marker, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
cat > "$TMP/bin/ssh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$TMP/bin/ssh"

HOST_ID=table4_4090d5 \
ROOT="$success_root" \
WAIT_ONCE=1 \
SSH_BIN="$TMP/bin/ssh" \
SHUTDOWN_BIN="$TMP/bin/shutdown" \
PYTHON=python3 \
bash "$SCRIPT" >/dev/null

[[ -e "$TMP/shutdown.called" ]]
[[ ! -e "$success_control/defer_shutdown_for_ndr_extension" ]]
grep -Fq "ORIGINAL_SHUTDOWN_SUCCESS shutting_down" \
  "$success_control/original_shutdown_waiter.log"
echo "verified: promoted original export removes defer sentinel and shuts down"
